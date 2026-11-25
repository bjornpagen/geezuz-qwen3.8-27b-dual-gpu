// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.model;
import engine.schedule;
import engine.testing;

namespace {

namespace model = engine::model;
namespace schedule = engine::schedule;

using model::Count;
using model::DraftDepth;
using model::RowBudget;
using model::Stream;
using schedule::Op;
using schedule::Program;

/** An element of a fixed-size table the model description or the schedule holds, bounds-checked. */
[[nodiscard]] constexpr auto at(auto const& table, std::size_t index) -> decltype(auto) {
	return std::span{table}[index];
}

constexpr auto card0 = schedule::image(0);
constexpr auto card1 = schedule::image(1);

constexpr auto programs = std::array{
    schedule::cycle<DraftDepth::One, RowBudget::ThreeTiles>,   schedule::cycle<DraftDepth::Two, RowBudget::ThreeTiles>,
    schedule::cycle<DraftDepth::Three, RowBudget::ThreeTiles>, schedule::cycle<DraftDepth::Four, RowBudget::ThreeTiles>,
    schedule::cycle<DraftDepth::One, RowBudget::FourTiles>,    schedule::cycle<DraftDepth::Two, RowBudget::FourTiles>,
    schedule::cycle<DraftDepth::Three, RowBudget::FourTiles>,  schedule::cycle<DraftDepth::Four, RowBudget::FourTiles>,
};

[[nodiscard]] auto phases(Program const& program) -> std::span<schedule::Phase const> {
	return std::span{program.phases}.first(program.phase_count);
}

[[nodiscard]] auto image_phase(schedule::Phase const& phase) -> schedule::ImagePhase const& {
	return at(card0.phases, phase.image_phase);
}

/** The placements a phase reads, ring then side. */
[[nodiscard]] auto placements_of(schedule::CardImage const& layout, schedule::ImagePhase const& phase) -> std::vector<schedule::Placement> {
	auto result = std::vector<schedule::Placement>{};
	for (auto const range : {phase.ring, phase.side}) {
		for (auto index = range.begin; index < range.begin + range.length; ++index) {
			result.push_back(at(layout.placements, index));
		}
	}
	return result;
}

// ---------------------------------------------------------------------------------------------------------------------
// Every byte, once per pass.

/** Which copy of a tensor: the stream and layer, as the model description counts copies. */
using Copy = std::tuple<Count, Stream, Count>;

/** The tensor copies a pass must read, from the model description's scopes and layer kinds alone. */
[[nodiscard]] auto copies_read_by(schedule::PassKind pass) -> std::set<Copy> {
	auto copies = std::set<Copy>{};
	for (auto tensor = Count{0}; tensor < schedule::tensor_count; ++tensor) {
		auto const& entry = at(model::inventory, tensor);
		auto const verify = pass == schedule::PassKind::Verify;
		switch (entry.scope) {
		case model::Scope::Model:
			if (verify && entry.split != model::Split::Host) {
				copies.emplace(tensor, Stream::Head, 0);
			}
			break;
		case model::Scope::Mtp:
			if (!verify) {
				copies.emplace(tensor, Stream::Mtp, 0);
			}
			break;
		case model::Scope::DecoderLayer:
		case model::Scope::LinearAttentionLayer:
		case model::Scope::FullAttentionLayer:
			if (verify) {
				for (auto layer = Count{0}; layer < model::trunk_layers; ++layer) {
					if (model::in_layer(entry.scope, at(model::trunk_mixers, layer))) {
						copies.emplace(tensor, Stream::Body, layer);
					}
				}
			} else if (model::in_layer(entry.scope, model::mtp_mixer)) {
				copies.emplace(tensor, Stream::Mtp, 0);
			}
			break;
		}
	}
	return copies;
}

/** How many times each 128-row block of each copy is read, per part, over one pass of a program. */
struct Coverage {
	std::map<Copy, std::vector<Count>> tiles;
	std::map<Copy, std::vector<Count>> scales;
	std::map<Copy, Count> whole;
	std::map<Copy, Count> bytes;
};

[[nodiscard]] auto coverage(Program const& program, schedule::Pass pass) -> Coverage {
	auto result = Coverage{};
	for (auto const& phase : phases(program)) {
		if (phase.pass.kind != pass.kind || phase.pass.draft != pass.draft) {
			continue;
		}
		for (auto const& placement : placements_of(card0, image_phase(phase))) {
			auto const& read = placement.read;
			for (auto which = Count{0}; which < read.tensor_count; ++which) {
				auto const tensor = at(read.tensor, which);
				auto const copy = Copy{tensor, placement.instance.stream, placement.instance.layer};
				auto const shape = model::card_shape(at(model::inventory, tensor), 0);
				auto const block_count = model::divide_up(at(shape.extents, 0), schedule::tile_rows);
				auto const chosen = schedule::blocks(read.select, tensor, 0);
				result.bytes[copy] += schedule::read_bytes(read, tensor, 0);
				if (read.part == schedule::Part::Whole) {
					result.whole[copy] += 1;
					continue;
				}
				auto& counts = read.part == schedule::Part::Tiles ? result.tiles[copy] : result.scales[copy];
				counts.resize(block_count);
				for (auto block = Count{0}; block < block_count; ++block) {
					counts.at(block) += schedule::holds(chosen, block) ? 1 : 0;
				}
			}
		}
	}
	return result;
}

[[nodiscard]] auto every_block_once(std::vector<Count> const& counts) -> bool {
	return std::ranges::all_of(counts, [](Count count) {
		return count == 1;
	});
}

/** The verify pass reads every trunk and head copy exactly once, and its bytes are the card's bytes of the copy. */
[[nodiscard]] auto verify_pass_reads_every_byte_once(Program const& program) -> bool {
	auto const covered = coverage(program, {.kind = schedule::PassKind::Verify, .draft = 0});
	auto const expected = copies_read_by(schedule::PassKind::Verify);
	auto read = std::set<Copy>{};
	for (auto const& [copy, bytes] : covered.bytes) {
		read.insert(copy);
	}
	return read == expected && std::ranges::all_of(expected, [&](Copy const& copy) {
		       auto const& entry = at(model::inventory, std::get<0>(copy));
		       auto const tiled = covered.tiles.contains(copy) && every_block_once(covered.tiles.at(copy));
		       auto const whole = covered.whole.contains(copy) && covered.whole.at(copy) == 1;
		       auto const scaled = covered.scales.contains(copy) && every_block_once(covered.scales.at(copy));
		       auto const scales_right = entry.format == model::Format::E4m3 ? scaled : !covered.scales.contains(copy);
		       return tiled != whole && scales_right && covered.bytes.at(copy) == model::card_bytes(entry, 0);
	       });
}

/**
 * Every mtp pass reads every mtp copy exactly once, and the lm_head blocks of the draft vocabulary exactly once and no
 * other lm_head block.
 */
[[nodiscard]] auto mtp_passes_read_every_byte_once(Program const& program) -> bool {
	auto const expected = copies_read_by(schedule::PassKind::CatchUp);
	auto const lm_head = Copy{schedule::tensors.lm_head, Stream::Head, 0};
	auto const draft_blocks = model::draft_vocab_per_card / schedule::tile_rows;
	auto holds = true;
	for (auto draft = Count{1}; draft <= model::drafts(program.depth); ++draft) {
		auto const kind = draft == 1 ? schedule::PassKind::CatchUp : schedule::PassKind::Step;
		auto const covered = coverage(program, {.kind = kind, .draft = draft});
		auto read = std::set<Copy>{};
		for (auto const& [copy, bytes] : covered.bytes) {
			read.insert(copy);
		}
		auto with_view = expected;
		with_view.insert(lm_head);
		holds = holds && read == with_view;
		for (auto const& copy : expected) {
			auto const& entry = at(model::inventory, std::get<0>(copy));
			auto const tiled = covered.tiles.contains(copy) && every_block_once(covered.tiles.at(copy));
			auto const whole = covered.whole.contains(copy) && covered.whole.at(copy) == 1;
			holds = holds && tiled != whole && covered.bytes.at(copy) == model::card_bytes(entry, 0);
		}
		auto const& view = covered.tiles.at(lm_head);
		for (auto block = Count{0}; block < view.size(); ++block) {
			holds = holds && view.at(block) == (block < draft_blocks ? Count{1} : Count{0});
		}
	}
	return holds;
}

[[nodiscard]] auto every_program_reads_every_byte_once() -> bool {
	return std::ranges::all_of(programs, [](Program const& program) {
		return verify_pass_reads_every_byte_once(program) && mtp_passes_read_every_byte_once(program);
	});
}

[[nodiscard]] auto passes_follow_the_depth() -> bool {
	return std::ranges::all_of(programs, [](Program const& program) {
		auto drafts = std::vector<Count>{};
		for (auto const& phase : phases(program)) {
			if (phase.step.op == Op::Draft) {
				drafts.push_back(phase.pass.draft);
			}
		}
		auto expected = std::vector<Count>(model::drafts(program.depth));
		std::ranges::iota(expected, Count{1});
		return drafts == expected;
	});
}

// ---------------------------------------------------------------------------------------------------------------------
// Dataflow.

/** A value the model's arithmetic passes from phase to phase. */
enum class Value : unsigned char {
	Residual,
	Tokens,
	Logits,
	MixerInput,
	CoreOutput,
	Early,
	Attended,
	Activation,
};

/** The values one op reads and writes, written from the model's math, independently of the generator. */
struct Flow {
	std::vector<Value> reads;
	std::vector<Value> writes;
};

[[nodiscard]] auto flow(Op op) -> Flow {
	switch (op) {
	case Op::Embed:
		return {.reads = {}, .writes = {Value::Residual}};
	case Op::DeltanetIn:
	case Op::AttentionIn:
		return {.reads = {Value::Residual}, .writes = {Value::MixerInput}};
	case Op::DeltanetCore:
	case Op::AttentionCore:
		return {.reads = {Value::MixerInput}, .writes = {Value::CoreOutput}};
	case Op::DeltanetZ:
	case Op::AttentionGate:
		return {.reads = {Value::Residual}, .writes = {Value::Early}};
	case Op::AttentionMerge:
		return {.reads = {Value::CoreOutput, Value::Early}, .writes = {Value::Attended}};
	case Op::DeltanetOut:
		return {.reads = {Value::CoreOutput, Value::Early, Value::Residual}, .writes = {Value::Residual}};
	case Op::AttentionOut:
		return {.reads = {Value::Attended, Value::Residual}, .writes = {Value::Residual}};
	case Op::MlpIn:
		return {.reads = {Value::Residual}, .writes = {Value::Activation}};
	case Op::MlpOut:
		return {.reads = {Value::Activation, Value::Residual}, .writes = {Value::Residual}};
	case Op::Head:
	case Op::DraftHead:
		return {.reads = {Value::Residual}, .writes = {Value::Logits}};
	case Op::Verify:
	case Op::Draft:
		return {.reads = {Value::Logits}, .writes = {Value::Tokens}};
	case Op::MtpIn:
		return {.reads = {Value::Tokens, Value::Residual}, .writes = {Value::Residual}};
	}
	std::unreachable();
}

/** Each phase's dependencies, closed transitively. */
[[nodiscard]] auto ancestors(Program const& program) -> std::vector<std::set<Count>> {
	auto result = std::vector<std::set<Count>>(program.phase_count);
	for (auto index = Count{0}; index < program.phase_count; ++index) {
		auto const& phase = at(program.phases, index);
		for (auto const dependency : std::span{phase.dependencies}.first(phase.dependency_count)) {
			result.at(index).insert(dependency);
			if (dependency < index) {
				result.at(index).insert_range(result.at(dependency));
			}
		}
	}
	return result;
}

/** Every dependency precedes its phase in stream order: no consumer is scheduled before its producer. */
[[nodiscard]] auto dependencies_precede() -> bool {
	return std::ranges::all_of(programs, [](Program const& program) {
		for (auto index = Count{0}; index < program.phase_count; ++index) {
			auto const& phase = at(program.phases, index);
			for (auto const dependency : std::span{phase.dependencies}.first(phase.dependency_count)) {
				if (dependency >= index) {
					return false;
				}
			}
		}
		return true;
	});
}

/**
 * Walking the stream in order, every value a phase reads was last written by a phase it depends on, every read value
 * was written this cycle (the embedding's tokens come from the last cycle), and every dependency wrote a value the
 * phase reads: the dependencies are exactly the dataflow.
 */
[[nodiscard]] auto dependencies_are_the_dataflow() -> bool {
	return std::ranges::all_of(programs, [](Program const& program) {
		auto const closed = ancestors(program);
		auto writer = std::map<Value, Count>{};
		for (auto index = Count{0}; index < program.phase_count; ++index) {
			auto const& phase = at(program.phases, index);
			auto const step = flow(phase.step.op);
			auto direct = std::set<Count>{};
			for (auto const value : step.reads) {
				if (!writer.contains(value) || !closed.at(index).contains(writer.at(value))) {
					return false;
				}
				direct.insert(writer.at(value));
			}
			auto declared = std::set<Count>{};
			for (auto const dependency : std::span{phase.dependencies}.first(phase.dependency_count)) {
				declared.insert(dependency);
			}
			if (declared != direct) {
				return false;
			}
			for (auto const value : step.writes) {
				writer.insert_or_assign(value, index);
			}
		}
		return true;
	});
}

/** The input-split tensors a phase streams: their per-card products are partial sums. */
[[nodiscard]] auto streams_partial_sums(schedule::ImagePhase const& phase) -> bool {
	auto partial = false;
	for (auto index = phase.ring.begin; index < phase.ring.begin + phase.ring.length; ++index) {
		auto const& entry = at(model::inventory, at(at(card0.placements, index).read.tensor, 0));
		partial = partial || model::split_axis(entry.split) == 1;
	}
	return partial;
}

[[nodiscard]] auto streams_vocabulary(schedule::ImagePhase const& phase) -> bool {
	auto vocabulary = false;
	for (auto index = phase.ring.begin; index < phase.ring.begin + phase.ring.length; ++index) {
		vocabulary = vocabulary || at(at(card0.placements, index).read.tensor, 0) == schedule::tensors.lm_head;
	}
	return vocabulary;
}

/**
 * A phase exchanges partial sums exactly when it streams a tensor split on its input axis, and candidates exactly
 * when it streams vocabulary rows; a partial exchange carries the whole residual in output tiles.
 */
[[nodiscard]] auto exchanges_follow_the_splits() -> bool {
	return std::ranges::all_of(card0.phases, [](schedule::ImagePhase const& phase) {
		auto const kind = schedule::exchange(phase.step.op);
		auto const partial = streams_partial_sums(phase);
		auto const candidates = streams_vocabulary(phase);
		auto const& streamed = at(model::inventory, at(at(card0.placements, phase.ring.begin).read.tensor, 0));
		auto const tiles_right =
		    !partial || at(model::card_shape(streamed, 0).extents, 0) == schedule::exchange_tiles * schedule::tile_rows;
		return (kind == schedule::Exchange::Partials) == partial && (kind == schedule::Exchange::Candidates) == candidates && tiles_right;
	});
}

/**
 * Each exchange of a kind depends on the one before it, so a receive slot is rewritten only after both cards
 * consumed what it held two exchanges earlier.
 */
[[nodiscard]] auto exchange_slots_are_never_overrun() -> bool {
	return std::ranges::all_of(programs, [](Program const& program) {
		auto const closed = ancestors(program);
		for (auto const kind : {schedule::Exchange::Partials, schedule::Exchange::Candidates}) {
			auto previous = std::optional<Count>{};
			for (auto index = Count{0}; index < program.phase_count; ++index) {
				if (schedule::exchange(at(program.phases, index).step.op) != kind) {
					continue;
				}
				if (previous.has_value() && !closed.at(index).contains(*previous)) {
					return false;
				}
				previous = index;
			}
		}
		return true;
	});
}

[[nodiscard]] auto exchange_counts() -> bool {
	return std::ranges::all_of(programs, [](Program const& program) {
		auto const per_pass = Count{3};
		auto const main = model::trunk_layers * 2;
		return schedule::exchanges(program, schedule::Exchange::Partials) == main + per_pass * model::drafts(program.depth) &&
		       schedule::exchanges(program, schedule::Exchange::Candidates) == 1 + model::drafts(program.depth);
	});
}

/**
 * Every serial core is followed in stream order by a weight phase that does not depend on it and that the core's
 * consumer needs: the dependency-free weights stream while the core runs.
 */
[[nodiscard]] auto weights_stream_during_every_core() -> bool {
	return std::ranges::all_of(programs, [](Program const& program) {
		auto const closed = ancestors(program);
		for (auto index = Count{0}; index < program.phase_count; ++index) {
			auto const op = at(program.phases, index).step.op;
			if (op != Op::DeltanetCore && op != Op::AttentionCore) {
				continue;
			}
			auto const next = index + 1;
			auto const& early = image_phase(at(program.phases, next));
			auto const independent = !closed.at(next).contains(index);
			auto const streams = early.ring.length > 0;
			auto needed = false;
			for (auto later = next + 1; later < program.phase_count; ++later) {
				needed = needed || (closed.at(later).contains(index) && closed.at(later).contains(next) &&
				                    std::ranges::contains(std::span{at(program.phases, later).dependencies}, next));
			}
			if (!independent || !streams || !needed) {
				return false;
			}
		}
		return true;
	});
}

// ---------------------------------------------------------------------------------------------------------------------
// The image.

[[nodiscard]] auto same_key(schedule::Placement const& left, schedule::Placement const& right) -> bool {
	return left.read.tensor_count == right.read.tensor_count && std::ranges::equal(left.read.tensor, right.read.tensor) &&
	       left.read.part == right.read.part && left.read.select == right.read.select && left.instance.stream == right.instance.stream &&
	       left.instance.layer == right.instance.layer;
}

/** Both cards stream the same placements, in the same order, at the same offsets and sizes. */
[[nodiscard]] auto both_cards_stream_alike() -> bool {
	auto same = card0.bytes == card1.bytes && card0.stream_bytes == card1.stream_bytes;
	for (auto index = Count{0}; index < schedule::placements; ++index) {
		auto const& left = at(card0.placements, index);
		auto const& right = at(card1.placements, index);
		same = same && same_key(left, right) && left.offset == right.offset && left.bytes == right.bytes;
	}
	for (auto index = Count{0}; index < schedule::image_phases; ++index) {
		auto const& left = at(card0.phases, index);
		auto const& right = at(card1.phases, index);
		same = same && left.step.op == right.step.op && left.ring.begin == right.ring.begin && left.side.begin == right.side.begin;
	}
	return same;
}

/**
 * The tiles fill the image's front without a gap, the side data follows chunk-aligned without overlap, every placement
 * is distinct, and together they are the card's resident weights.
 */
[[nodiscard]] auto image_tiles_the_weights() -> bool {
	auto extents = std::vector<std::pair<Count, Count>>{};
	auto total = Count{0};
	auto sound = true;
	for (auto const& placement : card0.placements) {
		extents.emplace_back(placement.offset, placement.offset + placement.bytes);
		total += placement.bytes;
		auto const tiles = placement.read.part == schedule::Part::Tiles;
		sound = sound && (tiles ? placement.offset + placement.bytes <= card0.stream_bytes
		                        : placement.offset >= card0.side_offset && placement.offset % schedule::side_alignment == 0);
	}
	std::ranges::sort(extents);
	auto tile_end = Count{0};
	for (auto index = Count{0}; index < extents.size(); ++index) {
		auto const [begin, end] = extents.at(index);
		if (begin < card0.stream_bytes) {
			sound = sound && begin == tile_end;
			tile_end = end;
		}
		sound = sound && (index == 0 || extents.at(index - 1).second <= begin);
	}
	for (auto left = Count{0}; left < schedule::placements; ++left) {
		for (auto right = left + 1; right < schedule::placements; ++right) {
			sound = sound && !same_key(at(card0.placements, left), at(card0.placements, right));
		}
	}
	return sound && tile_end == card0.stream_bytes && total == model::weight_bytes_per_card && card0.bytes >= total;
}

/** The draft lm_head reads the front of the lm_head's tiles: a view, not a copy. */
[[nodiscard]] auto draft_head_is_a_prefix_of_the_head() -> bool {
	auto const head = std::ranges::find_if(card0.phases, [](auto const& phase) {
		return phase.step.op == Op::Head;
	});
	auto const draft = std::ranges::find_if(card0.phases, [](auto const& phase) {
		return phase.step.op == Op::DraftHead;
	});
	auto const& first = at(card0.placements, head->ring.begin);
	auto const& view = at(card0.placements, draft->ring.begin);
	return draft->ring.length == 1 && first.offset == view.offset && view.bytes == model::draft_head_bytes_per_card &&
	       at(first.read.tensor, 0) == schedule::tensors.lm_head;
}

// ---------------------------------------------------------------------------------------------------------------------
// Partition.

/** The split of `units` over the SMs covers each unit once, in SM order, with lengths at most one apart. */
[[nodiscard]] auto shares_are_balanced(Count units) -> bool {
	auto next = Count{0};
	auto shortest = units;
	auto longest = Count{0};
	for (auto sm = Count{0}; sm < schedule::sms; ++sm) {
		auto const part = schedule::share(units, sm);
		if (part.begin != next) {
			return false;
		}
		next += part.length;
		shortest = std::min(shortest, part.length);
		longest = std::max(longest, part.length);
	}
	return next == units && longest - shortest <= 1 && longest == model::divide_up(units, schedule::sms);
}

/** The finisher of a span holds its last unit, found by searching every SM's share. */
[[nodiscard]] auto finisher_holds_the_last_unit(Count units, Count first, Count span) -> bool {
	auto const last = first + span - 1;
	auto const holder = std::ranges::find_if(std::views::iota(Count{0}, schedule::sms), [&](Count sm) {
		auto const part = schedule::share(units, sm);
		return part.begin <= last && last < part.begin + part.length;
	});
	return *holder == schedule::finisher(units, first, span);
}

[[nodiscard]] auto partition_is_balanced_for_every_count(std::uint64_t seed) -> bool {
	auto generate = engine::testing::generator(seed);
	auto sound = true;
	for (auto units = Count{0}; units <= 4 * schedule::sms; ++units) {
		sound = sound && shares_are_balanced(units);
	}
	auto counts = std::uniform_int_distribution<Count>{1, 200'000};
	for (auto trial = 0; trial < 2'000; ++trial) {
		auto const units = counts(generate);
		sound = sound && shares_are_balanced(units);
		auto const last = std::uniform_int_distribution<Count>{0, units - 1}(generate);
		auto const span = std::uniform_int_distribution<Count>{1, last + 1}(generate);
		sound = sound && finisher_holds_the_last_unit(units, last + 1 - span, span);
	}
	return sound;
}

/**
 * Every phase's SM extents tile its run of the image in SM order, and every output tile's finisher holds its last
 * unit; nothing depends on the row budget, so the k-split is the same at every row count.
 */
[[nodiscard]] auto extents_tile_every_phase() -> bool {
	return std::ranges::all_of(card0.phases, [](schedule::ImagePhase const& phase) {
		auto const run = schedule::run(card0, phase);
		auto next = run.offset;
		for (auto sm = Count{0}; sm < schedule::sms; ++sm) {
			auto const extent = schedule::extent(card0, phase, sm);
			if (extent.length > 0 && extent.begin != next) {
				return false;
			}
			next += extent.length;
		}
		auto unit = Count{0};
		for (auto index = phase.ring.begin; index < phase.ring.begin + phase.ring.length; ++index) {
			auto const& placement = at(card0.placements, index);
			auto const span = schedule::units_per_output(placement);
			for (auto output = Count{0}; output < placement.bytes / schedule::unit_bytes(placement) / span; ++output) {
				if (!finisher_holds_the_last_unit(run.units, unit, span)) {
					return false;
				}
				unit += span;
			}
		}
		return next == run.offset + run.units * run.unit_bytes && unit == run.units;
	});
}

// ---------------------------------------------------------------------------------------------------------------------
// Shape and geometry.

/** A cycle's shape spends the row budget: each slot one committed row and the most drafts that fit, capped. */
[[nodiscard]] auto shapes_spend_the_budget() -> bool {
	auto sound = true;
	for (auto const budget : model::row_budgets) {
		for (auto live = Count{1}; live <= model::slots; ++live) {
			auto const drafts = model::drafts(schedule::shape(live, budget));
			auto const fits = live * (1 + drafts) <= model::rows(budget);
			auto const deepest = drafts == model::drafts(model::max_draft_depth) || live * (2 + drafts) > model::rows(budget);
			sound = sound && fits && deepest;
		}
	}
	return sound;
}

/** The workspace buffers are aligned, in order, and disjoint, each the size its geometry says. */
[[nodiscard]] auto workspace_buffers_are_disjoint() -> bool {
	return std::ranges::all_of(model::row_budgets, [](RowBudget budget) {
		auto const space = schedule::workspace(budget);
		auto const flag = schedule::flag_bytes;
		auto const buffers = std::array{
		    std::pair{space.partials, schedule::exchange_depth * schedule::exchange_tiles * schedule::partial_tile_bytes(budget)},
		    std::pair{space.partial_flags, schedule::exchange_depth * schedule::exchange_tiles * flag},
		    std::pair{space.candidates, schedule::exchange_depth * schedule::candidate_slot_bytes(budget)},
		    std::pair{space.candidate_flags, schedule::exchange_depth * flag},
		    std::pair{space.fixups, schedule::sms * schedule::fixup_slot_bytes(budget)},
		    std::pair{space.fixup_flags, schedule::sms * flag},
		    std::pair{space.counters, schedule::max_cycle_phases * flag},
		    std::pair{space.attention_records, schedule::attention_record_bytes(budget)},
		    std::pair{space.replay_rings, model::slots * schedule::replay_ring_bytes_per_slot},
		    std::pair{space.rope_table, schedule::rope_table_bytes},
		};
		auto end = Count{0};
		for (auto const& [offset, bytes] : buffers) {
			if (offset < end || offset % schedule::workspace_alignment != 0) {
				return false;
			}
			end = offset + bytes;
		}
		return end == space.bytes && end <= model::workspace_bytes;
	});
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "every pass of every cycle shape reads each card byte of its tensors exactly once",
	        .holds = every_program_reads_every_byte_once(),
	    },
	    engine::testing::Case{
	        .name = "a cycle runs one mtp pass per draft, yielding drafts 1 to d in order",
	        .holds = passes_follow_the_depth(),
	    },
	    engine::testing::Case{
	        .name = "every dependency precedes its phase in stream order",
	        .holds = dependencies_precede(),
	    },
	    engine::testing::Case{
	        .name = "the dependencies are exactly the model's dataflow, walking the stream",
	        .holds = dependencies_are_the_dataflow(),
	    },
	    engine::testing::Case{
	        .name = "a phase exchanges partials exactly when it streams an input-split tensor, candidates for vocabulary rows",
	        .holds = exchanges_follow_the_splits(),
	    },
	    engine::testing::Case{
	        .name = "each exchange depends on the previous one of its kind, so two receive slots are never overrun",
	        .holds = exchange_slots_are_never_overrun(),
	    },
	    engine::testing::Case{
	        .name = "a cycle exchanges partials twice per trunk layer and three times per mtp pass, candidates once per pass",
	        .holds = exchange_counts(),
	    },
	    engine::testing::Case{
	        .name = "every serial core is followed by weights it does not depend on and its consumer needs",
	        .holds = weights_stream_during_every_core(),
	    },
	    engine::testing::Case{
	        .name = "both cards lay out and stream the same placements at the same offsets",
	        .holds = both_cards_stream_alike(),
	    },
	    engine::testing::Case{
	        .name = "the image is the tiles without gaps, then aligned side data, holding each resident weight byte once",
	        .holds = image_tiles_the_weights(),
	    },
	    engine::testing::Case{
	        .name = "the draft lm_head streams the front of the lm_head's tiles",
	        .holds = draft_head_is_a_prefix_of_the_head(),
	    },
	    engine::testing::Case{
	        .name = "the split over the SMs is contiguous and balanced to one unit, seed 0x5c4ed01e",
	        .holds = partition_is_balanced_for_every_count(0x5c4e'd01e),
	    },
	    engine::testing::Case{
	        .name = "every phase's SM extents tile its run, and every output tile's finisher holds its last unit",
	        .holds = extents_tile_every_phase(),
	    },
	    engine::testing::Case{
	        .name = "a cycle's shape gives each live slot its committed row and the most drafts the budget allows",
	        .holds = shapes_spend_the_budget(),
	    },
	    engine::testing::Case{
	        .name = "the workspace buffers are aligned, disjoint and within the card's workspace",
	        .holds = workspace_buffers_are_disjoint(),
	    },
	};
	return engine::testing::report(cases);
}
