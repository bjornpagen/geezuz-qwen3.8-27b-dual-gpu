// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.file;
import engine.hugepage;
import engine.testing;

namespace {

namespace device = engine::device;
namespace fake = engine::driver_double;

using device::Devices;
using device::EventState;
using device::Rank;
using fake::Fake;
using fake::Kind;
using fake::Machine;

static_assert(device::HostBytes<engine::file::Mapping>, "a baked image's mapping can be pinned for its upload");
static_assert(device::HostBytes<engine::hugepage::Region>);

constexpr auto words = std::size_t{4096};

using Words = device::DeviceArray<std::uint32_t, words>;
using Staging = device::MappedArray<std::uint32_t, words>;

template<class Destination, class Source>
[[nodiscard]] auto finish(device::Transfer<Destination, Source>& transfer) -> std::optional<device::Delivered<Destination, Source>> {
	if (transfer.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds{5}) != EventState::Complete) {
		return std::nullopt;
	}
	return transfer.try_finish();
}

/** Host to card and back through mapped staging returns exactly the words sent, on either card. */
[[nodiscard]] auto upload_then_download_round_trips(std::uint64_t seed) -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto random = engine::testing::generator(seed);
	auto holds = true;
	for (auto const rank : device::ranks) {
		auto const& context = opened->context(rank);
		auto staging = context.allocate_mapped<std::uint32_t, words>();
		auto back = context.allocate_mapped<std::uint32_t, words>();
		auto array = context.allocate<std::uint32_t, words>();
		auto const stream = context.create_stream();
		if (!staging || !back || !array || !stream) {
			return false;
		}
		std::ranges::generate(staging->host(), [&] {
			return static_cast<std::uint32_t>(random());
		});
		auto const sent = std::vector<std::uint32_t>(staging->host().begin(), staging->host().end());
		auto up = context.upload(*stream, std::move(*array), std::move(*staging), 0);
		auto uploaded = finish(up);
		if (!uploaded || !uploaded->copied) {
			return false;
		}
		auto down = context.download(*stream, std::move(*back), std::move(uploaded->destination));
		auto downloaded = finish(down);
		holds = holds && downloaded && downloaded->copied && std::ranges::equal(downloaded->destination.host(), sent);
	}
	return holds && driver.violations().empty();
}

/** A pinned owner uploads from any offset, and unpinning hands the same bytes back with no pin left. */
[[nodiscard]] auto a_pinned_owner_uploads_from_an_offset() -> bool {
	constexpr auto offset = std::size_t{37};
	constexpr auto bytes = std::size_t{1000};
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const& context = opened->context(Rank::One);
	auto region = engine::hugepage::Region::allocate(4096);
	if (!region) {
		return false;
	}
	auto value = std::uint8_t{0};
	for (auto& byte : region->bytes()) {
		byte = std::byte{value};
		value = static_cast<std::uint8_t>(value * 5 + 3);
	}
	auto const base = std::bit_cast<std::uintptr_t>(region->bytes().data());
	auto source = context.pin(std::move(*region));
	auto array = context.allocate<std::byte, bytes>();
	auto back = context.allocate_mapped<std::byte, bytes>();
	auto const stream = context.create_stream();
	if (!source || !array || !back || !stream || driver.live(Kind::HostMemory) != 2) {
		return false;
	}
	auto up = context.upload(*stream, std::move(*array), std::move(*source), offset);
	auto uploaded = finish(up);
	if (!uploaded || !uploaded->copied) {
		return false;
	}
	auto down = context.download(*stream, std::move(*back), std::move(uploaded->destination));
	auto downloaded = finish(down);
	auto const owner = std::move(uploaded->source).unpin();
	return downloaded && downloaded->copied && std::bit_cast<std::uintptr_t>(owner.bytes().data()) == base &&
	       std::ranges::equal(downloaded->destination.host(), owner.bytes().subspan(offset, bytes)) && driver.live(Kind::HostMemory) == 1 &&
	       driver.violations().empty();
}

/** A copy from one card's memory lands in the other's exactly. */
[[nodiscard]] auto peer_copies_cross_the_cards_exactly(std::uint64_t seed) -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const& zero = opened->context(Rank::Zero);
	auto const& one = opened->context(Rank::One);
	auto staging = zero.allocate_mapped<std::uint32_t, words>();
	auto back = one.allocate_mapped<std::uint32_t, words>();
	auto source = zero.allocate<std::uint32_t, words>();
	auto destination = one.allocate<std::uint32_t, words>();
	auto const zero_stream = zero.create_stream();
	auto const one_stream = one.create_stream();
	if (!staging || !back || !source || !destination || !zero_stream || !one_stream) {
		return false;
	}
	auto random = engine::testing::generator(seed);
	std::ranges::generate(staging->host(), [&] {
		return static_cast<std::uint32_t>(random());
	});
	auto const sent = std::vector<std::uint32_t>(staging->host().begin(), staging->host().end());
	auto up = zero.upload(*zero_stream, std::move(*source), std::move(*staging), 0);
	auto uploaded = finish(up);
	if (!uploaded || !uploaded->copied) {
		return false;
	}
	auto across = one.copy_from_peer(*one_stream, std::move(*destination), zero, std::move(uploaded->destination));
	auto crossed = finish(across);
	if (!crossed || !crossed->copied) {
		return false;
	}
	auto down = one.download(*one_stream, std::move(*back), std::move(crossed->destination));
	auto downloaded = finish(down);
	return downloaded && downloaded->copied && std::ranges::equal(downloaded->destination.host(), sent) && driver.violations().empty();
}

/** The buffers come back only once the copy's event fires, however long that takes. */
[[nodiscard]] auto buffers_return_only_after_completion() -> bool {
	constexpr auto queries = std::size_t{4};
	auto const driver = Fake{Machine{.pending_queries = queries}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const& context = opened->context(Rank::Zero);
	auto staging = context.allocate_mapped<std::uint32_t, words>();
	auto array = context.allocate<std::uint32_t, words>();
	auto const stream = context.create_stream();
	if (!staging || !array || !stream) {
		return false;
	}
	auto transfer = context.upload(*stream, std::move(*array), std::move(*staging), 0);
	auto holds = true;
	for (auto query = std::size_t{0}; query < queries; ++query) {
		holds = holds && !transfer.try_finish().has_value();
	}
	auto const delivered = transfer.try_finish();
	return holds && delivered && delivered->copied && !transfer.try_finish().has_value();
}

/** A copy the driver refuses still hands both buffers back, with the refusal; they work for the next copy. */
[[nodiscard]] auto a_refused_copy_returns_its_buffers() -> bool {
	auto const attempt = [](std::optional<std::size_t> failing) -> std::pair<std::size_t, bool> {
		auto const driver = Fake{Machine{.failing_call = failing}};
		auto const opened = Devices::open(driver.driver());
		if (!opened) {
			return {0, false};
		}
		auto const& context = opened->context(Rank::Zero);
		auto staging = context.allocate_mapped<std::uint32_t, words>();
		auto array = context.allocate<std::uint32_t, words>();
		auto const stream = context.create_stream();
		if (!staging || !array || !stream) {
			return {0, false};
		}
		// The upload makes its event (current context, then create), then copies: the copy is the third call.
		auto const copy_call = driver.calls() + 2;
		auto first = context.upload(*stream, std::move(*array), std::move(*staging), 0);
		auto delivered = first.try_finish();
		if (!delivered) {
			return {copy_call, false};
		}
		auto const refused = !delivered->copied && delivered->copied.error() == device::DeviceError{.reason = device::Failure{
		                                                                                                .symbol = "cuMemcpyHtoDAsync_v2",
		                                                                                                .status = device::Status::Unknown,
		                                                                                            }};
		auto again = context.upload(*stream, std::move(delivered->destination), std::move(delivered->source), 0);
		auto const retried = again.try_finish();
		auto const reused = retried && retried->copied;
		return {copy_call, (failing ? refused : static_cast<bool>(delivered->copied)) && reused && driver.violations().empty()};
	};
	auto const [copy_call, clean] = attempt(std::nullopt);
	return clean && attempt(copy_call).second;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "words uploaded through mapped staging download back exactly, on both cards (seed 41)",
	        .holds = upload_then_download_round_trips(41),
	    },
	    engine::testing::Case{
	        .name = "a pinned owner uploads from an offset, and unpinning returns the same bytes with no pin left",
	        .holds = a_pinned_owner_uploads_from_an_offset(),
	    },
	    engine::testing::Case{
	        .name = "a peer copy lands one card's words in the other's memory exactly (seed 42)",
	        .holds = peer_copies_cross_the_cards_exactly(42),
	    },
	    engine::testing::Case{
	        .name = "a transfer hands its buffers back only once its completion event fires",
	        .holds = buffers_return_only_after_completion(),
	    },
	    engine::testing::Case{
	        .name = "a copy the driver refuses hands both buffers back with the refusal, ready for the next copy",
	        .holds = a_refused_copy_returns_its_buffers(),
	    },
	};
	return engine::testing::report(cases);
}
