// SPDX-License-Identifier: AGPL-3.0-only
//
// The cross-card and host kernels as the host launches them. A kernel that writes the other card's memory takes that
// card's owner: the address it was allocated at is valid on this card too, because the machine maps each card's
// memory into the other's.
export module engine.ubench_device:link;

import std;
import engine.driver;
import engine.prims;
import engine.ptx;
import :shapes;
import :images;
import :names;
import :stream;
import :sync;

namespace engine::ubench_device {

namespace ub = engine::device::ubench;
namespace link = engine::device::ubench::link;

export using Flag = driver::DeviceArray<unsigned long long, 1>;
export using Nanoseconds = driver::DeviceArray<unsigned long long, 1>;

/** Which end of the flag ping-pong a kernel is: the one that starts each round, or the one that answers it. */
export enum class FlagEnd : std::uint8_t { Ping, Pong };

export template<FlagEnd end, ub::Wait wait>
struct FlagKernel {
	static constexpr ptx::Image image = link_image;
	static constexpr EntryName name = entry_name("ubench_flag_", end == FlagEnd::Ping ? "ping" : "pong", "_", wait == ub::Wait::Backoff ? "backoff" : "spin");
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = link::flag_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = link::FlagParams;

	/** `own` lives on the launching card, `peer` on the other card. */
	[[nodiscard]] static auto params(Flag const& own, Flag const& peer, Nanoseconds const& nanoseconds) -> Params {
		return {.own_flag = own.address(), .peer_flag = peer.address(), .nanoseconds = nanoseconds.address()};
	}
};

export using ReceiveSlots = driver::DeviceArray<ub::Word4, link::receive_chunks>;
export using SenderFlags = driver::DeviceArray<unsigned long long, link::max_senders>;
export using PullSinks = driver::DeviceArray<unsigned, link::max_senders>;

export template<class Shape>
struct PushKernel {
	static constexpr ptx::Image image = link_image;
	static constexpr EntryName name = entry_name("ubench_push_r", Shape::rows, "_c", Shape::senders);
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = {.x = static_cast<std::uint32_t>(Shape::senders), .y = 1, .z = 1};
	static constexpr device::prims::Budget budget = Shape::budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = link::PushParams;

	/** `receive` and `flags` live on the receiving card, `go` on the sending one. */
	[[nodiscard]] static auto params(ReceiveSlots const& receive, SenderFlags const& flags, Flag const& go) -> Params {
		return {.receive = receive.address(), .flags = flags.address(), .go = go.address()};
	}
};

export template<class Shape>
struct PullKernel {
	static constexpr ptx::Image image = link_image;
	static constexpr EntryName name = entry_name("ubench_pull_r", Shape::rows, "_c", Shape::senders);
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = {.x = static_cast<std::uint32_t>(Shape::senders), .y = 1, .z = 1};
	static constexpr device::prims::Budget budget = Shape::budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = link::PullParams;

	/** `source` lives on the other card, `sinks` on the pulling one. */
	[[nodiscard]] static auto params(ReceiveSlots const& source, PullSinks const& sinks) -> Params {
		return {.source = source.address(), .sink = sinks.address()};
	}
};

export using ReceiveReading = driver::DeviceArray<link::ReceiveReading, 1>;

export template<int senders>
struct ReceiveKernel {
	using Shape = link::Receive<senders>;
	static constexpr ptx::Image image = link_image;
	static constexpr EntryName name = entry_name("ubench_receive_c", senders);
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = Shape::budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = link::ReceiveParams;

	/** `flags` and `reading` live on the receiving card, `go` on the sending one. */
	[[nodiscard]] static auto params(SenderFlags const& flags, Flag const& go, ReceiveReading const& reading) -> Params {
		return {.flags = flags.address(), .go = go.address(), .reading = reading.address()};
	}
};

/** The exchange sweep: rows per item by sending blocks. */
export template<int rows>
using PushRow = std::tuple<link::Push<rows, 4>, link::Push<rows, 16>, link::Push<rows, 64>>;
export using PushSweep = std::tuple<PushRow<1>, PushRow<8>, PushRow<24>, PushRow<32>>;

export using ReleaseScratch = driver::DeviceArray<ub::Word4, ub::sms>;
export using BackgroundSlots = driver::DeviceArray<ub::Word4, static_cast<std::size_t>(ub::sms) * link::background_chunks>;
export using BackgroundHost = driver::MappedArray<ub::Word4, static_cast<std::size_t>(ub::sms) * link::background_chunks>;

/** The timed block waits for nothing, but the background blocks wait for its stop, so every block must be resident. */
export template<link::Background background>
struct ReleaseKernel {
	static constexpr ptx::Image image = link_image;
	static constexpr std::string_view entry = background == link::Background::Idle         ? "ubench_release_idle"
	                                          : background == link::Background::PeerStores ? "ubench_release_peer_stores"
	                                                                                       : "ubench_release_host_writes";
	static constexpr driver::Dim grid = every_sm;
	static constexpr device::prims::Budget budget = link::release_budget;
	static constexpr driver::Residency residency = driver::Residency::CoResident;
	using Params = link::ReleaseParams;

	/** `peer` lives on the other card; everything else but `host` on the launching card. */
	[[nodiscard]] static auto params(Flag const& flag, ReleaseScratch const& scratch, BackgroundSlots const& peer, BackgroundHost const& host,
	                                 Flag const& stop, Cycles const& cycles) -> Params {
		return {
		    .flag = flag.address(),
		    .scratch = scratch.address(),
		    .peer = peer.address(),
		    .host = host.device_address(),
		    .stop = stop.address(),
		    .cycles = cycles.address(),
		};
	}
};

export using Table = driver::MappedArray<ub::Word4, link::table_chunks>;
export using FetchReading = driver::DeviceArray<link::FetchReading, 1>;

export template<int rows>
struct FetchKernel {
	using Shape = link::Fetch<rows>;
	static constexpr ptx::Image image = link_image;
	static constexpr EntryName name = entry_name("ubench_fetch_r", rows);
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = link::fetch_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = link::FetchParams;

	[[nodiscard]] static auto params(Table const& table, FetchReading const& reading) -> Params {
		return {.table = table.device_address(), .reading = reading.address()};
	}
};

export using FetchSweep = std::tuple<FetchKernel<1>, FetchKernel<8>, FetchKernel<24>, FetchKernel<32>>;

/** Where a resident mailbox's ring slots are, as the ring's owner reports it. */
export struct MailboxSlots {
	std::uint64_t address;
};

export struct MailboxKernel {
	static constexpr ptx::Image image = link_image;
	static constexpr std::string_view entry = "ubench_mailbox";
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = link::mailbox_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = link::MailboxParams;

	[[nodiscard]] static auto params(MailboxSlots const& slots) -> Params {
		return {.slots = slots.address};
	}
};

}
