// SPDX-License-Identifier: AGPL-3.0-only
//
// The persistent decode kernel, the grid-barrier baseline: one co-resident block per SM on each card, serving the
// resident ring's tickets until the exit ticket. A ticket applies the kv pool's commands and runs one decode cycle:
// the schedule's phases in stream order, each task between two grid barriers, the cards exchanging fp32 partials and
// ranked candidates over nvlink and both computing the same verdict. The kernel waits for nothing per cycle but the
// ticket itself and, right before it ranks the lm_head's rows, the masks the host fills while the body streams. A
// logits ticket runs the same verify pass and hands the lm_head's logits to the host where a cycle ranks them.
#include "decode/machine.h"
#include "decode/profile.h"
#include "decode/tasks.h"
#include "decode/walk.h"
#include "decode/wire.h"
#include "decode/workspace.h"
#include "deltanet/layout.h"
#include "model.h"
#include "prims/budget.h"
#include "prims/memory.h"
#include "prims/order.h"
#include "prims/thread.h"
#include "ring/ring.h"

namespace engine::device::decode {

[[nodiscard]] __attribute__((device, always_inline)) inline auto global_thread() -> unsigned {
	return prims::block() * static_cast<unsigned>(decode_budget.threads) + prims::thread_in_block();
}

inline constexpr unsigned grid_threads = sms * static_cast<unsigned>(decode_budget.threads);

[[nodiscard]] __attribute__((device, always_inline)) inline auto now() -> unsigned long long {
	return static_cast<unsigned long long>(__nvvm_read_ptx_sreg_globaltimer());
}

/** Zeroes `bytes` (whole chunks) from `offset` of the workspace, spread over every thread of the grid. */
__attribute__((device)) inline void zero(Machine const& machine, Count offset, Count bytes) {
	auto const written = prims::policy<Eviction::Normal>();
	auto* chunks = machine.local<Chunk>(offset);
	for (Count chunk = global_thread(); chunk < bytes / 16; chunk += grid_threads) {
		prims::store(chunks + chunk, Chunk{}, written);
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// Start: clear every flag the cards and SMs publish epochs on, and meet the peer card, which may store into this
// card's flags only after it has seen this card ready.

__attribute__((device)) inline void start(Machine const& machine, Grid grid) {
	auto const& exchange = workspace.exchange;
	zero(machine, exchange.partial_flags, schedule::exchange_depth * schedule::exchange_tiles * schedule::flag_bytes);
	zero(machine, exchange.candidate_flags, schedule::exchange_depth * schedule::flag_bytes);
	zero(machine, exchange.fixup_flags, prims::round_up(static_cast<int>(sms * schedule::flag_bytes), 16));
	grid.sync(prims::remote_backoff);
	if (prims::block() == 0 && prims::thread_in_block() == 0) {
		prims::EpochWriter<prims::Scope::System>{.flag = &at<Control>(machine.params.peer_control)->peer_ready.word}.publish(machine.params.launch);
		prims::EpochReader<prims::Scope::System>{.flag = &at<Control>(machine.params.control)->peer_ready.word}.await(machine.params.launch, prims::remote_backoff);
	}
	grid.sync(prims::remote_backoff);
}

// ---------------------------------------------------------------------------------------------------------------------
// The kv pool's commands.

__attribute__((device)) inline void copy_chunks(Chunk const* from, Chunk* to, Count chunks) {
	auto const once = prims::policy<Eviction::First>();
	auto const written = prims::policy<Eviction::Normal>();
	for (Count chunk = global_thread(); chunk < chunks; chunk += grid_threads) {
		prims::store(to + chunk, prims::load(from + chunk, once), written);
	}
}

__attribute__((device)) inline void apply(Machine const& machine, Command const& command) {
	auto const written = prims::policy<Eviction::Normal>();
	auto const once = prims::policy<Eviction::First>();
	auto* const pool = at<unsigned char>(machine.params.pool);
	auto const block_chunks = [&](unsigned index) {
		return reinterpret_cast<Chunk*>(pool + Count{index} * model::kv_block_bytes);
	};
	auto* const states = at<unsigned char>(machine.params.states) + Count{command.slot} * deltanet::slot_image_bytes;
	constexpr Count chunks = model::kv_block_bytes / 16;
	constexpr Count row_chunks = attn::row_bytes / 16;
	constexpr Count plane_chunks = attn::plane_bytes / 16;
	switch (command.action) {
	case Action::ZeroBlock: {
		Chunk* const target = block_chunks(command.first);
		for (Count chunk = global_thread(); chunk < chunks; chunk += grid_threads) {
			prims::store(target + chunk, Chunk{}, written);
		}
		break;
	}
	case Action::CopyPrefix: {
		Chunk const* const source = block_chunks(command.first);
		Chunk* const target = block_chunks(command.second);
		for (Count chunk = global_thread(); chunk < chunks; chunk += grid_threads) {
			Count const position = (chunk % plane_chunks) / row_chunks;
			prims::store(target + chunk, position < command.third ? prims::load(source + chunk, once) : Chunk{}, written);
		}
		break;
	}
	case Action::SetTableEntry:
		if (global_thread() == 0) {
			prims::store_word(machine.local<unsigned>(workspace.tables) + Count{command.slot} * model::kv_table_blocks + command.second, command.first,
			                  written);
		}
		break;
	case Action::SaveSnapshotPart: {
		deltanet::Extent const part = deltanet::snapshot_part(command.second);
		copy_chunks(reinterpret_cast<Chunk const*>(states + part.offset), block_chunks(command.first), part.bytes / 16);
		break;
	}
	case Action::LoadSnapshotPart: {
		deltanet::Extent const part = deltanet::snapshot_part(command.second);
		copy_chunks(block_chunks(command.first), reinterpret_cast<Chunk*>(states + part.offset), part.bytes / 16);
		break;
	}
	case Action::Replay:
		// Not reached: a replay needs the grid's barriers, and `apply_commands` runs it itself.
		expect(false);
		break;
	}
}

/** The ticket's commands in order, every SM at once within a wave, a grid barrier after each wave. */
__attribute__((device)) inline void apply_commands(Machine const& machine, Shared& shared, Grid grid) {
	Descriptor const& descriptor = machine.descriptor();
	expect(descriptor.commands <= max_commands);
	for (unsigned index = 0; index < descriptor.commands; ++index) {
		Command const command = descriptor.command[index];
		if (index > 0 && command.wave != descriptor.command[index - 1].wave) {
			grid.sync(prims::remote_backoff);
		}
		if (command.action == Action::Replay) {
			materialize(machine, shared, grid, command.slot, command.first);
		} else {
			apply(machine, command);
		}
	}
	if (descriptor.commands > 0) {
		grid.sync(prims::remote_backoff);
	}
}

static_assert(deltanet::snapshot_part(0).bytes % 16 == 0 && deltanet::slot_image_bytes % 16 == 0, "snapshots copy in whole chunks");

// ---------------------------------------------------------------------------------------------------------------------
// The cycle: the walk, interpreted.

struct Runner {
	Machine const* machine;
	Shared* shared;
	Epochs* epochs;
	Grid grid;

	__attribute__((device)) void phase(Step /*step*/, Pass /*pass*/, Count /*image_phase*/, Count /*row_tiles*/) const {}

	__attribute__((device)) void barrier() const {
		grid.sync(prims::remote_backoff);
	}

	__attribute__((device)) void task(Point const& point) const {
		Machine const& m = *machine;
		switch (point.task) {
		case Task::Embed:
			embed(m);
			break;
		case Task::Norm:
			norm_task(m, point);
			break;
		case Task::PromptRow:
			prompt_row(m);
			break;
		case Task::Gemv:
			if (point.row_tiles == row_tiles) {
				gemv_task<static_cast<int>(row_tiles)>(m, *shared, *epochs, point);
			} else {
				gemv_task<static_cast<int>(schedule::step_row_tiles)>(m, *shared, *epochs, point);
			}
			break;
		case Task::DeltanetCore:
			deltanet_core(m, shared->deltanet, point);
			break;
		case Task::AttentionCore:
			if (point.pass.kind == PassKind::Step) {
				attention_core(m, shared->narrow, point);
			} else {
				attention_core(m, shared->wide, point);
			}
			break;
		case Task::AttentionMerge:
			attention_merge(m, point);
			break;
		case Task::RankSegments:
			rank_segments(m, shared->rank, point, epochs->ticket);
			break;
		case Task::RankGroups:
			rank_groups(m, shared->rank, point);
			break;
		case Task::RankFinal:
			rank_final(m, shared->rank, point, *epochs);
			break;
		case Task::Verify:
			verify(m, shared->verify, *epochs);
			break;
		case Task::MtpGather:
			mtp_gather(m, point);
			break;
		case Task::MtpNorm:
			mtp_norm(m, point);
			break;
		case Task::Draft:
			draft(m, shared->verify, point, *epochs);
			break;
		case Task::Logits:
			hand_logits(m, epochs->ticket);
			break;
		}
	}
};

// ---------------------------------------------------------------------------------------------------------------------
// The profile build's ticket: the same walk under a visitor that times every phase on every block and counts it into
// the phase's slot of the card's profile, then the profile copied to the host. Nothing here runs in any other build.

/** Each image phase's weight bytes on each card, as the schedule places them. */
struct PhaseWeights {
	unsigned long long bytes[model::cards][schedule::image_phases];
};

[[nodiscard]] consteval auto plan_phase_weights() -> PhaseWeights {
	auto weights = PhaseWeights{};
	for (Count card = 0; card < model::cards; ++card) {
		auto const layout = schedule::image(card);
		for (Count phase = 0; phase < schedule::image_phases; ++phase) {
			weights.bytes[card][phase] = schedule::phase_bytes(layout, layout.phases[phase]);
		}
	}
	return weights;
}

inline constexpr PhaseWeights phase_weights = plan_phase_weights();

/** Bytes a phase streams by its shape: its weights on this card, and for a core the state or kv its rows read. */
[[nodiscard]] __attribute__((device)) inline auto streamed(Machine const& machine, Step step, Pass pass, Count image_phase) -> unsigned long long {
	unsigned long long bytes = phase_weights.bytes[machine.params.card][image_phase];
	if (step.op == Op::DeltanetCore || step.op == Op::AttentionCore) {
		PassRows const rows = pass_rows(machine, pass);
		for (unsigned live = 0; live < rows.cycle.live; ++live) {
			attn::SlotWork const& work = rows.cycle.work[live];
			bytes += step.op == Op::DeltanetCore ? schedule::deltanet_state_bytes_per_layer_per_slot
			                                     : Count{work.position + work.rows} * schedule::kv_bytes_per_position_per_layer;
		}
	}
	return bytes;
}

/**
 * Raises a 64-bit counter on this card to at least `value`; relaxed, and order-free like a sum, so concurrent raises
 * from every block land in any order.
 */
__attribute__((device, always_inline)) inline void raise(unsigned long long* counter, unsigned long long value) {
	static_cast<void>(__nvvm_atom_max_gen_ull(counter, value));
}

/** No phase is open: before a walk's first phase and after its end. */
inline constexpr Count no_phase = profile_phases;

/** The kernel's own visitor, every phase timed on each block by its thread 0 and counted into the phase's slot. */
struct Timed {
	Runner const* runner;
	Control* control;
	Count slot;
	unsigned long long entered;

	__attribute__((device)) void phase(Step step, Pass pass, Count image_phase, Count phase_tiles) {
		close();
		slot = profile_slot(pass, image_phase);
		entered = prims::cycles();
		if (prims::block() == 0 && prims::thread_in_block() == 0) {
			PhaseCounters& counters = control->profile.phase[slot];
			prims::accumulate(&counters.visits, 1);
			prims::accumulate(&counters.bytes, streamed(*runner->machine, step, pass, image_phase));
		}
		runner->phase(step, pass, image_phase, phase_tiles);
	}

	__attribute__((device)) void task(Point const& point) const {
		runner->task(point);
	}

	__attribute__((device)) void barrier() const {
		unsigned long long const since = prims::cycles();
		runner->barrier();
		if (prims::thread_in_block() == 0) {
			prims::accumulate(&control->profile.phase[slot].barrier, prims::cycles() - since);
		}
	}

	/**
	 * Ends the open phase on this block: its time in the phase, and the waits its warps noted, into the phase's
	 * counters. A phase ends right after the barrier closing its last task, which ordered every warp's notes before it.
	 */
	__attribute__((device)) void close() {
		if (slot == no_phase) {
			return;
		}
		if (prims::thread_in_block() == 0) {
			unsigned long long const spent = prims::cycles() - entered;
			PhaseCounters& counters = control->profile.phase[slot];
			prims::accumulate(&counters.cycles, spent);
			raise(&counters.slowest, spent);
			raise(&counters.fastest_complement, ~spent);
			BlockWaits& waits = control->waits[prims::block()];
			prims::accumulate(&counters.peer, prims::load_relaxed<prims::Scope::Gpu>(&waits.peer));
			prims::accumulate(&counters.masks, prims::load_relaxed<prims::Scope::Gpu>(&waits.masks));
			prims::store_relaxed<prims::Scope::Gpu>(&waits.peer, 0);
			prims::store_relaxed<prims::Scope::Gpu>(&waits.masks, 0);
		}
		slot = no_phase;
	}
};

/**
 * A profile build's ticket: the walk under `Timed`, block 0's cycles and nanoseconds over it (the SM clock's rate), and
 * then, once every block has counted, the card's profile copied by block 0 into the host's buffer before it answers.
 */
__attribute__((device)) inline void walk_timed(Machine const& machine, Runner const& runner, Grid grid) {
	Descriptor const& descriptor = machine.descriptor();
	if (descriptor.work == Work::Apply) {
		return;
	}
	auto* const control = at<Control>(machine.params.control);
	Timed timed = {.runner = &runner, .control = control, .slot = no_phase, .entered = 0};
	unsigned long long const first_cycle = prims::cycles();
	unsigned long long const first_nanosecond = now();
	switch (descriptor.work) {
	case Work::Apply:
		break;
	case Work::Cycle:
		walk_cycle(descriptor.depth, timed);
		break;
	case Work::PromptEnd:
		walk_prompt_end(descriptor.depth, timed);
		break;
	case Work::Logits:
		walk_logits(timed);
		break;
	}
	timed.close();
	unsigned long long const cycles = prims::cycles() - first_cycle;
	unsigned long long const nanoseconds = now() - first_nanosecond;
	grid.sync(prims::remote_backoff);
	if (prims::block() != 0) {
		return;
	}
	if (prims::thread_in_block() == 0) {
		prims::accumulate(&control->profile.tickets, 1);
		prims::accumulate(&control->profile.ticket_cycles, cycles);
		prims::accumulate(&control->profile.ticket_nanoseconds, nanoseconds);
	}
	prims::sync_block();
	auto const once = prims::policy<Eviction::First>();
	auto const written = prims::policy<Eviction::Normal>();
	/* SAFETY: the profile is whole chunks, in the control block and in the host's buffer alike. */
	auto const* from = reinterpret_cast<Chunk const*>(&control->profile);
	auto* to = at<Chunk>(prims::load_relaxed<prims::Scope::Gpu>(&control->profile_buffer.word));
	for (Count chunk = prims::thread_in_block(); chunk < sizeof(Profile) / 16; chunk += static_cast<Count>(decode_budget.threads)) {
		prims::store(to + chunk, prims::load(from + chunk, once), written);
	}
	prims::sync_block();
}

static_assert(ring::Layout<Descriptor, Result>::descriptor % 16 == 0, "the descriptor copies in whole chunks");
static_assert(sizeof(Result) % 4 == 0 && sizeof(SlotOutcome) % 4 == 0, "the result copies in whole words");

/** Block 0 waits for the ticket and copies its command and descriptor into the workspace. */
__attribute__((device)) inline void receive(Machine const& machine, unsigned long long ticket) {
	auto* const slot = at<Ring>(machine.params.ring) + ring::index(ticket, ring_slots);
	if (prims::thread_in_block() == 0) {
		prims::EpochReader<prims::Scope::System>{.flag = &slot->request.ticket}.await(ticket, prims::remote_backoff);
		auto const once = prims::policy<Eviction::First>();
		/* SAFETY: the command is the first byte of its 32-bit word in the request half (little-endian). */
		unsigned const word = prims::load_word(reinterpret_cast<unsigned const*>(&slot->request.command), once);
		prims::store_word(machine.local<unsigned>(workspace.command), word & 0xffU, prims::policy<Eviction::Last>());
	}
	prims::sync_block();
	auto const once = prims::policy<Eviction::First>();
	auto const reused = prims::policy<Eviction::Last>();
	/* SAFETY: the descriptor is chunk-aligned in the slot (checked above) and in the workspace. */
	auto const* from = reinterpret_cast<Chunk const*>(&slot->request.descriptor);
	auto* to = machine.local<Chunk>(workspace.descriptor);
	for (Count chunk = prims::thread_in_block(); chunk < sizeof(Descriptor) / 16; chunk += static_cast<Count>(decode_budget.threads)) {
		prims::store(to + chunk, prims::load(from + chunk, once), reused);
	}
}

/** Block 0's thread 0 answers the ticket: the outcomes and timings, then the released ticket. */
__attribute__((device)) inline void answer(Machine const& machine, unsigned long long ticket, unsigned long long started) {
	auto* const slot = at<Ring>(machine.params.ring) + ring::index(ticket, ring_slots);
	auto const written = prims::policy<Eviction::Normal>();
	auto const reused = prims::policy<Eviction::Last>();
	/* SAFETY: the result's fields are 32-bit words in mapped memory, the outcomes the same layout in the workspace. */
	auto* const words = reinterpret_cast<unsigned*>(&slot->response.result);
	unsigned long long const finished = now();
	prims::store_word(words, static_cast<unsigned>(started), written);
	prims::store_word(words + 1, static_cast<unsigned>(started >> 32U), written);
	prims::store_word(words + 2, static_cast<unsigned>(finished), written);
	prims::store_word(words + 3, static_cast<unsigned>(finished >> 32U), written);
	auto const* outcomes = machine.local<unsigned const>(workspace.outcomes);
	constexpr Count outcome_words = model::slots * sizeof(SlotOutcome) / 4;
	static_assert(__builtin_offsetof(Result, slots) == 16);
	for (Count index = 0; index < outcome_words; ++index) {
		prims::store_word(words + 4 + index, prims::load_word(outcomes + index, reused), written);
	}
	prims::store_release<prims::Scope::System>(&slot->response.ticket, ticket);
}

}

namespace decode = engine::device::decode;
namespace prims = engine::device::prims;

extern "C" __attribute__((device, used)) constexpr prims::Budget engine_budget_decode_loop = decode::decode_budget;

/**
 * The resident decode loop on one card: started once per card with that card's memories and the peer's, it serves
 * tickets 1, 2, 3, ... of the card's ring until the exit ticket. Both cards run it at once; each waits for the other at
 * start and at every exchange.
 */
extern "C" __attribute__((global, launch_bounds(decode::decode_budget.threads, decode::decode_budget.blocks_per_sm))) void
decode_loop(decode::Params params) {
	using namespace engine::device::decode;
	Machine const machine = {.params = params};
	auto& shared = prims::shared_layout<decode_budget, Shared>();
	Grid const grid = machine.grid();
	start(machine, grid);
	Epochs epochs = {.ticket = 0, .fixups = 0, .partials = 0, .candidates = 0};
	Runner runner = {.machine = &machine, .shared = &shared, .epochs = &epochs, .grid = grid};
	for (;;) {
		++epochs.ticket;
		if (prims::block() == 0) {
			receive(machine, epochs.ticket);
		}
		grid.sync(prims::remote_backoff);
		unsigned long long const started = now();
		auto const command = static_cast<engine::device::ring::Command>(prims::load_word(machine.local<unsigned const>(workspace.command), prims::policy<Eviction::Last>()));
		if (command == engine::device::ring::Command::Exit) {
			if (prims::block() == 0 && prims::thread_in_block() == 0) {
				auto* const slot = at<Ring>(params.ring) + engine::device::ring::index(epochs.ticket, ring_slots);
				prims::store_release<prims::Scope::System>(&slot->response.ticket, epochs.ticket);
			}
			return;
		}
		apply_commands(machine, shared, grid);
		if constexpr (engine::device::profiler == engine::device::Profiler::On) {
			walk_timed(machine, runner, grid);
		} else {
			Descriptor const& descriptor = machine.descriptor();
			switch (descriptor.work) {
			case Work::Apply:
				break;
			case Work::Cycle:
				walk_cycle(descriptor.depth, runner);
				break;
			case Work::PromptEnd:
				walk_prompt_end(descriptor.depth, runner);
				break;
			case Work::Logits:
				walk_logits(runner);
				break;
			}
		}
		if (prims::block() == 0 && prims::thread_in_block() == 0) {
			answer(machine, epochs.ticket, started);
		}
	}
}
