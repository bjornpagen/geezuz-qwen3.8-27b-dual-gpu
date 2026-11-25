// SPDX-License-Identifier: AGPL-3.0-only
//
// The cross-card and host microbenchmarks: the nvlink flag ping-pong between one warp per card, partials pushed into
// the peer's receive slots (and the same bytes pulled, for comparison), the cost of a system-scope release beside
// other traffic, row fetches from mapped pinned host memory, and a resident mailbox answering the host through the
// engine's ring protocol.
#include "prims/budget.h"
#include "prims/memory.h"
#include "prims/order.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "ring/ring.h"
#include "ubench/device.h"
#include "ubench/shapes.h"

namespace engine::device::ubench::link {

/** Publishes `round` on a flag in the other card's memory, ordered as `flavor` orders it. */
template<Flavor flavor>
__attribute__((device)) inline void publish_flag(unsigned long long* flag, unsigned long long round) {
	using namespace engine::device::prims;
	if constexpr (flavor == Flavor::ReleaseAcquire) {
		store_release<Scope::System>(flag, round);
	} else if constexpr (flavor == Flavor::Relaxed) {
		store_relaxed<Scope::System>(flag, round);
	} else {
		fence_sc<Scope::System>();
		store_relaxed<Scope::System>(flag, round);
	}
}

/** Waits for `round` on a flag in this card's memory, observed as `flavor` observes it. */
template<Flavor flavor, Wait wait>
__attribute__((device)) inline void await_flag(unsigned long long const* flag, unsigned long long round) {
	using namespace engine::device::prims;
	constexpr Backoff backoff = backoff_for<wait>(remote_backoff);
	unsigned sleep = backoff.first_ns;
	auto const seen = [&] {
		if constexpr (flavor == Flavor::ReleaseAcquire) {
			return load_acquire<Scope::System>(flag);
		} else {
			return load_relaxed<Scope::System>(flag);
		}
	};
	while (seen() < round) {
		nanosleep(sleep);
		sleep = sleep * 2U < backoff.ceiling_ns ? sleep * 2U : backoff.ceiling_ns;
	}
	if constexpr (flavor == Flavor::Fenced) {
		fence_sc<Scope::System>();
	}
}

/** The initiating card: publishes round r on the peer's flag, then waits for the peer to publish r on its own. */
template<Flavor flavor, Wait wait>
__attribute__((device)) inline void flag_ping(FlagParams params) {
	using namespace engine::device::prims;
	if (lane() != 0) {
		return;
	}
	unsigned long long const start = nanoseconds();
	for (unsigned long long round = 1; round <= static_cast<unsigned long long>(flag_iterations); ++round) {
		publish_flag<flavor>(at<unsigned long long>(params.peer_flag), round);
		await_flag<flavor, wait>(at<unsigned long long const>(params.own_flag), round);
	}
	*at<unsigned long long>(params.nanoseconds) = nanoseconds() - start;
}

/** The answering card: waits for round r on its own flag and publishes r on the initiator's. */
template<Flavor flavor, Wait wait>
__attribute__((device)) inline void flag_pong(FlagParams params) {
	using namespace engine::device::prims;
	if (lane() != 0) {
		return;
	}
	unsigned long long const start = nanoseconds();
	for (unsigned long long round = 1; round <= static_cast<unsigned long long>(flag_iterations); ++round) {
		await_flag<flavor, wait>(at<unsigned long long const>(params.own_flag), round);
		publish_flag<flavor>(at<unsigned long long>(params.peer_flag), round);
	}
	*at<unsigned long long>(params.nanoseconds) = nanoseconds() - start;
}

/**
 * Sender block s waits for the receiver's go, then pushes its share of every item into the peer's receive slots with
 * weak 16-byte stores and, once the block's stores are done, releases the item on its own flag in the peer's memory:
 * one release per item.
 */
template<class Shape>
__attribute__((device)) inline void push(PushParams params) {
	using namespace engine::device::prims;
	using Block = NamedBarrier<1, push_threads / 32>;
	unsigned const sender = block();
	if (thread_in_block() == 0) {
		EpochReader<Scope::System>{.flag = at<unsigned long long const>(params.go)}.await(1, backoff_for<Wait::Spin>(remote_backoff));
	}
	Block::sync();
	PeerOutbox const outbox = {
	    .slots = at<Chunk>(params.receive) + static_cast<unsigned long long>(sender) * Shape::sender_chunks,
	    .flag = {.flag = at<unsigned long long>(params.flags) + sender},
	};
	for (unsigned item = 1; item <= static_cast<unsigned>(items); ++item) {
		for (unsigned chunk = thread_in_block(); chunk < static_cast<unsigned>(Shape::sender_chunks); chunk += push_threads) {
			outbox.push(chunk, {.word = {item, chunk, sender, 0}});
		}
		Block::sync();
		if (thread_in_block() == 0) {
			outbox.publish(item);
		}
	}
}

/** Thread s watches sender s's flag; the block times from every sender's first item to every sender's last. */
template<class Shape>
__attribute__((device)) inline void receive(ReceiveParams params) {
	using namespace engine::device::prims;
	constexpr Backoff spin = backoff_for<Wait::Spin>(remote_backoff);
	unsigned const sender = thread_in_block();
	EpochReader<Scope::System> const reader = {.flag = at<unsigned long long const>(params.flags) + sender};
	if (sender == 0) {
		EpochWriter<Scope::System>{.flag = at<unsigned long long>(params.go)}.publish(1);
	}
	if (sender < static_cast<unsigned>(Shape::senders)) {
		reader.await(1, spin);
	}
	sync_block();
	unsigned long long const first = nanoseconds();
	if (sender < static_cast<unsigned>(Shape::senders)) {
		reader.await(static_cast<unsigned long long>(items), spin);
	}
	sync_block();
	if (sender == 0) {
		*at<ReceiveReading>(params.reading) = {.first_ns = first, .last_ns = nanoseconds()};
	}
}

/** The same bytes pulled: block s loads its share of every item from the peer's memory. */
template<class Shape>
__attribute__((device)) inline void pull(PullParams params) {
	using namespace engine::device::prims;
	Chunk const* const share = at<Chunk const>(params.source) + static_cast<unsigned long long>(block()) * Shape::sender_chunks;
	auto const once = policy<Eviction::First>();
	unsigned sink = 0;
	for (int item = 0; item < items; ++item) {
		for (unsigned chunk = thread_in_block(); chunk < static_cast<unsigned>(Shape::sender_chunks); chunk += push_threads) {
			sink ^= load(share + chunk, once).word[0];
		}
	}
	count(at<unsigned>(params.sink) + block(), sink);
}

/**
 * Block 0's lane 0 times system-scope releases, each after one weak store, as a tile's publication is; the other
 * blocks meanwhile store to the peer or to mapped host memory until block 0 tells them to stop.
 */
template<Background background>
__attribute__((device)) inline void release(ReleaseParams params) {
	using namespace engine::device::prims;
	auto const once = policy<Eviction::First>();
	unsigned long long* const stop = at<unsigned long long>(params.stop);
	if (block() == 0) {
		if (lane() != 0) {
			return;
		}
		Chunk* const scratch = at<Chunk>(params.scratch);
		unsigned long long* const flag = at<unsigned long long>(params.flag);
		unsigned long long const start = cycles();
		for (unsigned long long item = 1; item <= static_cast<unsigned long long>(release_iterations); ++item) {
			store(scratch + item % static_cast<unsigned long long>(sms), {.word = {static_cast<unsigned>(item), 0, 0, 0}}, once);
			store_release<Scope::System>(flag, item);
		}
		*at<unsigned long long>(params.cycles) = cycles() - start;
		store_release<Scope::Gpu>(stop, 1);
		return;
	}
	if constexpr (background != Background::Idle) {
		Chunk* const target = at<Chunk>(background == Background::PeerStores ? params.peer : params.host) + static_cast<unsigned long long>(block()) * background_chunks;
		for (unsigned round = 0;; ++round) {
			if (round % background_chunks == 0 && load_acquire<Scope::Gpu>(stop) != 0) {
				return;
			}
			store(target + round % background_chunks, {.word = {round, block(), lane(), 0}}, once);
		}
	}
}

/** Fetches `rows` 10 KiB rows from the mapped table per repeat, timing each repeat across the block. */
template<class Shape>
__attribute__((device)) inline void fetch(FetchParams params) {
	using namespace engine::device::prims;
	Chunk const* const table = at<Chunk const>(params.table);
	auto const once = policy<Eviction::First>();
	unsigned sink = 0;
	unsigned long long best = ~0ULL;
	unsigned long long total = 0;
	for (unsigned repeat = 0; repeat < static_cast<unsigned>(fetch_repeats); ++repeat) {
		Chunk const* const rows = table + static_cast<unsigned long long>(repeat) * Shape::rows * table_row_chunks;
		sync_block();
		unsigned long long const start = nanoseconds();
		for (unsigned chunk = thread_in_block(); chunk < static_cast<unsigned>(Shape::rows * table_row_chunks); chunk += fetch_budget.threads) {
			sink ^= load_constant(rows + chunk, once).word[0];
		}
		// Every thread has used what it loaded, so the barrier closes the fetch.
		sync_block();
		unsigned long long const took = nanoseconds() - start;
		best = took < best ? took : best;
		total += took;
	}
	if (thread_in_block() == 0) {
		*at<FetchReading>(params.reading) = {.best_ns = best, .total_ns = total, .sink = sink};
	}
}

/**
 * The resident mailbox: serves tickets 1, 2, 3, ... through the ring's slots in mapped host memory, answering each
 * `Run` with the card's time on seeing it, until the exit ticket. It spins, because its job is to time the path.
 */
__attribute__((device)) inline void mailbox(MailboxParams params) {
	using namespace engine::device::prims;
	using Slot = ring::Slot<MailboxRequest, MailboxAnswer>;
	using Layout = ring::Layout<MailboxRequest, MailboxAnswer>;
	static_assert(Layout::descriptor + sizeof(MailboxRequest) <= sizeof(Chunk), "a request's head fits one chunk");
	if (lane() != 0) {
		return;
	}
	Slot* const slots = at<Slot>(params.slots);
	auto const fresh = policy<Eviction::First>();
	for (unsigned long long ticket = 1;; ++ticket) {
		Slot& slot = slots[ring::index(ticket, mailbox_capacity)];
		while (load_acquire<Scope::System>(&slot.request.ticket) != ticket) {
		}
		unsigned long long const seen = nanoseconds();
		/* SAFETY: the request's ticket, command and descriptor share the slot's first 16 bytes. */
		Chunk const head = load(reinterpret_cast<Chunk const*>(&slot.request), fresh);
		auto const command = static_cast<ring::Command>(head.word[Layout::command / 4] & 0xffU);
		if (command == ring::Command::Run) {
			/* SAFETY: the answer is a 64-bit value at the result offset, 8-byte aligned in the response line. */
			unsigned* const words = reinterpret_cast<unsigned*>(&slot.response.result);
			store_word(words, static_cast<unsigned>(seen), fresh);
			store_word(words + 1, static_cast<unsigned>(seen >> 32U), fresh);
		}
		store_release<Scope::System>(&slot.response.ticket, ticket);
		if (command != ring::Command::Run) {
			return;
		}
	}
}

}

namespace ub = engine::device::ubench;
namespace link = engine::device::ubench::link;

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_flag_ping_release_backoff = link::flag_budget;
extern "C" __attribute__((global, launch_bounds(link::flag_budget.threads, link::flag_budget.blocks_per_sm))) void ubench_flag_ping_release_backoff(link::FlagParams params) {
	link::flag_ping<link::Flavor::ReleaseAcquire, ub::Wait::Backoff>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_flag_pong_release_backoff = link::flag_budget;
extern "C" __attribute__((global, launch_bounds(link::flag_budget.threads, link::flag_budget.blocks_per_sm))) void ubench_flag_pong_release_backoff(link::FlagParams params) {
	link::flag_pong<link::Flavor::ReleaseAcquire, ub::Wait::Backoff>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_flag_ping_release_spin = link::flag_budget;
extern "C" __attribute__((global, launch_bounds(link::flag_budget.threads, link::flag_budget.blocks_per_sm))) void ubench_flag_ping_release_spin(link::FlagParams params) {
	link::flag_ping<link::Flavor::ReleaseAcquire, ub::Wait::Spin>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_flag_pong_release_spin = link::flag_budget;
extern "C" __attribute__((global, launch_bounds(link::flag_budget.threads, link::flag_budget.blocks_per_sm))) void ubench_flag_pong_release_spin(link::FlagParams params) {
	link::flag_pong<link::Flavor::ReleaseAcquire, ub::Wait::Spin>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_flag_ping_relaxed_spin = link::flag_budget;
extern "C" __attribute__((global, launch_bounds(link::flag_budget.threads, link::flag_budget.blocks_per_sm))) void ubench_flag_ping_relaxed_spin(link::FlagParams params) {
	link::flag_ping<link::Flavor::Relaxed, ub::Wait::Spin>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_flag_pong_relaxed_spin = link::flag_budget;
extern "C" __attribute__((global, launch_bounds(link::flag_budget.threads, link::flag_budget.blocks_per_sm))) void ubench_flag_pong_relaxed_spin(link::FlagParams params) {
	link::flag_pong<link::Flavor::Relaxed, ub::Wait::Spin>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_flag_ping_fenced_spin = link::flag_budget;
extern "C" __attribute__((global, launch_bounds(link::flag_budget.threads, link::flag_budget.blocks_per_sm))) void ubench_flag_ping_fenced_spin(link::FlagParams params) {
	link::flag_ping<link::Flavor::Fenced, ub::Wait::Spin>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_flag_pong_fenced_spin = link::flag_budget;
extern "C" __attribute__((global, launch_bounds(link::flag_budget.threads, link::flag_budget.blocks_per_sm))) void ubench_flag_pong_fenced_spin(link::FlagParams params) {
	link::flag_pong<link::Flavor::Fenced, ub::Wait::Spin>(params);
}

// Pushes and pulls: rows per item by sending blocks.
using PushR1C4 = link::Push<1, 4>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r1_c4 = PushR1C4::budget;
extern "C" __attribute__((global, launch_bounds(PushR1C4::budget.threads, PushR1C4::budget.blocks_per_sm))) void ubench_push_r1_c4(link::PushParams params) {
	link::push<PushR1C4>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r1_c4 = PushR1C4::budget;
extern "C" __attribute__((global, launch_bounds(PushR1C4::budget.threads, PushR1C4::budget.blocks_per_sm))) void ubench_pull_r1_c4(link::PullParams params) {
	link::pull<PushR1C4>(params);
}

using PushR1C16 = link::Push<1, 16>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r1_c16 = PushR1C16::budget;
extern "C" __attribute__((global, launch_bounds(PushR1C16::budget.threads, PushR1C16::budget.blocks_per_sm))) void ubench_push_r1_c16(link::PushParams params) {
	link::push<PushR1C16>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r1_c16 = PushR1C16::budget;
extern "C" __attribute__((global, launch_bounds(PushR1C16::budget.threads, PushR1C16::budget.blocks_per_sm))) void ubench_pull_r1_c16(link::PullParams params) {
	link::pull<PushR1C16>(params);
}

using PushR1C64 = link::Push<1, 64>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r1_c64 = PushR1C64::budget;
extern "C" __attribute__((global, launch_bounds(PushR1C64::budget.threads, PushR1C64::budget.blocks_per_sm))) void ubench_push_r1_c64(link::PushParams params) {
	link::push<PushR1C64>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r1_c64 = PushR1C64::budget;
extern "C" __attribute__((global, launch_bounds(PushR1C64::budget.threads, PushR1C64::budget.blocks_per_sm))) void ubench_pull_r1_c64(link::PullParams params) {
	link::pull<PushR1C64>(params);
}

using PushR8C4 = link::Push<8, 4>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r8_c4 = PushR8C4::budget;
extern "C" __attribute__((global, launch_bounds(PushR8C4::budget.threads, PushR8C4::budget.blocks_per_sm))) void ubench_push_r8_c4(link::PushParams params) {
	link::push<PushR8C4>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r8_c4 = PushR8C4::budget;
extern "C" __attribute__((global, launch_bounds(PushR8C4::budget.threads, PushR8C4::budget.blocks_per_sm))) void ubench_pull_r8_c4(link::PullParams params) {
	link::pull<PushR8C4>(params);
}

using PushR8C16 = link::Push<8, 16>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r8_c16 = PushR8C16::budget;
extern "C" __attribute__((global, launch_bounds(PushR8C16::budget.threads, PushR8C16::budget.blocks_per_sm))) void ubench_push_r8_c16(link::PushParams params) {
	link::push<PushR8C16>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r8_c16 = PushR8C16::budget;
extern "C" __attribute__((global, launch_bounds(PushR8C16::budget.threads, PushR8C16::budget.blocks_per_sm))) void ubench_pull_r8_c16(link::PullParams params) {
	link::pull<PushR8C16>(params);
}

using PushR8C64 = link::Push<8, 64>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r8_c64 = PushR8C64::budget;
extern "C" __attribute__((global, launch_bounds(PushR8C64::budget.threads, PushR8C64::budget.blocks_per_sm))) void ubench_push_r8_c64(link::PushParams params) {
	link::push<PushR8C64>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r8_c64 = PushR8C64::budget;
extern "C" __attribute__((global, launch_bounds(PushR8C64::budget.threads, PushR8C64::budget.blocks_per_sm))) void ubench_pull_r8_c64(link::PullParams params) {
	link::pull<PushR8C64>(params);
}

using PushR24C4 = link::Push<24, 4>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r24_c4 = PushR24C4::budget;
extern "C" __attribute__((global, launch_bounds(PushR24C4::budget.threads, PushR24C4::budget.blocks_per_sm))) void ubench_push_r24_c4(link::PushParams params) {
	link::push<PushR24C4>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r24_c4 = PushR24C4::budget;
extern "C" __attribute__((global, launch_bounds(PushR24C4::budget.threads, PushR24C4::budget.blocks_per_sm))) void ubench_pull_r24_c4(link::PullParams params) {
	link::pull<PushR24C4>(params);
}

using PushR24C16 = link::Push<24, 16>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r24_c16 = PushR24C16::budget;
extern "C" __attribute__((global, launch_bounds(PushR24C16::budget.threads, PushR24C16::budget.blocks_per_sm))) void ubench_push_r24_c16(link::PushParams params) {
	link::push<PushR24C16>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r24_c16 = PushR24C16::budget;
extern "C" __attribute__((global, launch_bounds(PushR24C16::budget.threads, PushR24C16::budget.blocks_per_sm))) void ubench_pull_r24_c16(link::PullParams params) {
	link::pull<PushR24C16>(params);
}

using PushR24C64 = link::Push<24, 64>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r24_c64 = PushR24C64::budget;
extern "C" __attribute__((global, launch_bounds(PushR24C64::budget.threads, PushR24C64::budget.blocks_per_sm))) void ubench_push_r24_c64(link::PushParams params) {
	link::push<PushR24C64>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r24_c64 = PushR24C64::budget;
extern "C" __attribute__((global, launch_bounds(PushR24C64::budget.threads, PushR24C64::budget.blocks_per_sm))) void ubench_pull_r24_c64(link::PullParams params) {
	link::pull<PushR24C64>(params);
}

using PushR32C4 = link::Push<32, 4>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r32_c4 = PushR32C4::budget;
extern "C" __attribute__((global, launch_bounds(PushR32C4::budget.threads, PushR32C4::budget.blocks_per_sm))) void ubench_push_r32_c4(link::PushParams params) {
	link::push<PushR32C4>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r32_c4 = PushR32C4::budget;
extern "C" __attribute__((global, launch_bounds(PushR32C4::budget.threads, PushR32C4::budget.blocks_per_sm))) void ubench_pull_r32_c4(link::PullParams params) {
	link::pull<PushR32C4>(params);
}

using PushR32C16 = link::Push<32, 16>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r32_c16 = PushR32C16::budget;
extern "C" __attribute__((global, launch_bounds(PushR32C16::budget.threads, PushR32C16::budget.blocks_per_sm))) void ubench_push_r32_c16(link::PushParams params) {
	link::push<PushR32C16>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r32_c16 = PushR32C16::budget;
extern "C" __attribute__((global, launch_bounds(PushR32C16::budget.threads, PushR32C16::budget.blocks_per_sm))) void ubench_pull_r32_c16(link::PullParams params) {
	link::pull<PushR32C16>(params);
}

using PushR32C64 = link::Push<32, 64>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_push_r32_c64 = PushR32C64::budget;
extern "C" __attribute__((global, launch_bounds(PushR32C64::budget.threads, PushR32C64::budget.blocks_per_sm))) void ubench_push_r32_c64(link::PushParams params) {
	link::push<PushR32C64>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_pull_r32_c64 = PushR32C64::budget;
extern "C" __attribute__((global, launch_bounds(PushR32C64::budget.threads, PushR32C64::budget.blocks_per_sm))) void ubench_pull_r32_c64(link::PullParams params) {
	link::pull<PushR32C64>(params);
}

using ReceiveC4 = link::Receive<4>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_receive_c4 = ReceiveC4::budget;
extern "C" __attribute__((global, launch_bounds(ReceiveC4::budget.threads, ReceiveC4::budget.blocks_per_sm))) void ubench_receive_c4(link::ReceiveParams params) {
	link::receive<ReceiveC4>(params);
}

using ReceiveC16 = link::Receive<16>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_receive_c16 = ReceiveC16::budget;
extern "C" __attribute__((global, launch_bounds(ReceiveC16::budget.threads, ReceiveC16::budget.blocks_per_sm))) void ubench_receive_c16(link::ReceiveParams params) {
	link::receive<ReceiveC16>(params);
}

using ReceiveC64 = link::Receive<64>;
extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_receive_c64 = ReceiveC64::budget;
extern "C" __attribute__((global, launch_bounds(ReceiveC64::budget.threads, ReceiveC64::budget.blocks_per_sm))) void ubench_receive_c64(link::ReceiveParams params) {
	link::receive<ReceiveC64>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_release_idle = link::release_budget;
extern "C" __attribute__((global, launch_bounds(link::release_budget.threads, link::release_budget.blocks_per_sm))) void ubench_release_idle(link::ReleaseParams params) {
	link::release<link::Background::Idle>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_release_peer_stores = link::release_budget;
extern "C" __attribute__((global, launch_bounds(link::release_budget.threads, link::release_budget.blocks_per_sm))) void ubench_release_peer_stores(link::ReleaseParams params) {
	link::release<link::Background::PeerStores>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_release_host_writes = link::release_budget;
extern "C" __attribute__((global, launch_bounds(link::release_budget.threads, link::release_budget.blocks_per_sm))) void ubench_release_host_writes(link::ReleaseParams params) {
	link::release<link::Background::HostWrites>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_fetch_r1 = link::fetch_budget;
extern "C" __attribute__((global, launch_bounds(link::fetch_budget.threads, link::fetch_budget.blocks_per_sm))) void ubench_fetch_r1(link::FetchParams params) {
	link::fetch<link::Fetch<1>>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_fetch_r8 = link::fetch_budget;
extern "C" __attribute__((global, launch_bounds(link::fetch_budget.threads, link::fetch_budget.blocks_per_sm))) void ubench_fetch_r8(link::FetchParams params) {
	link::fetch<link::Fetch<8>>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_fetch_r24 = link::fetch_budget;
extern "C" __attribute__((global, launch_bounds(link::fetch_budget.threads, link::fetch_budget.blocks_per_sm))) void ubench_fetch_r24(link::FetchParams params) {
	link::fetch<link::Fetch<24>>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_fetch_r32 = link::fetch_budget;
extern "C" __attribute__((global, launch_bounds(link::fetch_budget.threads, link::fetch_budget.blocks_per_sm))) void ubench_fetch_r32(link::FetchParams params) {
	link::fetch<link::Fetch<32>>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_mailbox = link::mailbox_budget;
extern "C" __attribute__((global, launch_bounds(link::mailbox_budget.threads, link::mailbox_budget.blocks_per_sm))) void ubench_mailbox(link::MailboxParams params) {
	link::mailbox(params);
}
