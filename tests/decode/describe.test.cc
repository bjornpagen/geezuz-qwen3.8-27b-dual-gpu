// SPDX-License-Identifier: AGPL-3.0-only
//
// The one description of a decode error: specific enough that two errors with the same text are the same error.
import std;
import engine.core;
import engine.decode;
import engine.device;
import engine.file;
import engine.image;
import engine.model;
import engine.testing;

namespace {

namespace decode = engine::decode;
namespace device = engine::device;
namespace file = engine::file;
namespace image = engine::image;

using decode::DecodeError;

template<class T, std::size_t N>
[[nodiscard]] auto pick(std::mt19937_64& random, std::array<T, N> const& choices) -> T {
	return choices[std::uniform_int_distribution<std::size_t>{0, N - 1}(random)];
}

/** Every value of an enumeration whose values run from zero to `last`. */
template<class E, E last>
[[nodiscard]] auto any_of(std::mt19937_64& random) -> E {
	return static_cast<E>(std::uniform_int_distribution<int>{0, std::to_underlying(last)}(random));
}

/** A decode error of any kind, with facts drawn small so that equal draws occur and distinct ones are near. */
[[nodiscard]] auto any_error(std::mt19937_64& random) -> DecodeError {
	auto const small = [&] {
		return std::uniform_int_distribution<std::uint32_t>{0, 3}(random);
	};
	auto const maybe = [&] -> std::optional<std::uint64_t> {
		auto const ticket = small();
		return ticket == 0 ? std::nullopt : std::optional<std::uint64_t>{ticket};
	};
	auto const card = [&] {
		return pick(random, std::array{engine::model::Card::Zero, engine::model::Card::One});
	};
	switch (std::uniform_int_distribution{0, 8}(random)) {
	case 0:
		return {.reason = device::DeviceError{.reason = device::Failure{.symbol = pick(random, std::array<std::string_view, 2>{"cuMemAlloc_v2", "cuLaunchKernel"}),
		                                                                 .status = pick(random, std::array{device::Status::OutOfMemory, device::Status::LaunchFailed})}}};
	case 1:
		return {.reason = device::DeviceError{.reason = device::CardCount{.found = static_cast<int>(small())}}};
	case 2:
		return {.reason = decode::Unloadable{.image = any_of<decode::ImageOf, decode::ImageOf::CardOne>(random),
		                                     .error = {.cause = file::FileError{.failure = any_of<file::Failure, file::Failure::Unreadable>(random)}}}};
	case 3:
		return {.reason = decode::Unloadable{
		            .image = any_of<decode::ImageOf, decode::ImageOf::CardOne>(random),
		            .error = {.cause = image::Rejection{.failure = any_of<image::Failure, image::Failure::PayloadMismatch>(random)}}}};
	case 4:
		return {.reason = decode::Refused{.reason = any_of<decode::Refusal, decode::Refusal::WrongTicket>(random), .ticket = maybe()}};
	case 5:
		return {.reason = decode::TimedOut{.card = card(), .waiting = any_of<decode::Waiting, decode::Waiting::Rotations>(random)}};
	case 6:
		return {.reason = decode::Abandoned{.ticket = maybe(), .kernels = any_of<decode::Kernels, decode::Kernels::Finished>(random)}};
	case 7:
		return {.reason = decode::LoopGone{.ticket = small()}};
	default:
		return {.reason = decode::CardsDisagree{.ticket = small(),
		                                        .slot = small(),
		                                        .what = any_of<decode::Disagreement, decode::Disagreement::Drafted>(random),
		                                        .at = small(),
		                                        .zero = small(),
		                                        .one = small()}};
	}
}

/**
 * Two errors with the same description are the same error, so the text names every fact that tells failures apart.
 * Searched over many errors of every kind with near-identical facts.
 */
[[nodiscard]] auto equal_descriptions_mean_equal_errors(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto seen = std::map<std::string, DecodeError>{};
	auto kinds = std::set<std::size_t>{};
	auto holds = true;
	for (auto draw = 0; draw < 20'000; ++draw) {
		auto const error = any_error(random);
		auto const text = decode::describe(error);
		kinds.insert(error.reason.index());
		auto const [known, fresh] = seen.try_emplace(text, error);
		holds = holds && !text.empty() && (fresh || known->second == error);
	}
	return holds && kinds.size() == std::variant_size_v<decltype(DecodeError::reason)>;
}

/** A device error is described by device's own description, inside decode's. */
[[nodiscard]] auto device_errors_read_as_device_describes_them() -> bool {
	auto const error = device::DeviceError{.reason = device::Failure{.symbol = "cuMemAlloc_v2", .status = device::Status::OutOfMemory}};
	return decode::describe({.reason = error}).contains(device::describe(error));
}

/** An image error is described by image's own description, after the name of the image it is about. */
[[nodiscard]] auto image_errors_read_as_image_describes_them() -> bool {
	auto const error = image::ImageError{.cause = image::Rejection{.failure = image::Failure::ScheduleMismatch}};
	return decode::describe({.reason = decode::Unloadable{.image = decode::ImageOf::CardOne, .error = error}})
	    .contains(image::describe(error));
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "random decode errors of every kind: equal descriptions only for equal errors (seed 71)",
	        .holds = equal_descriptions_mean_equal_errors(71),
	    },
	    engine::testing::Case{
	        .name = "a device error reads as device describes it",
	        .holds = device_errors_read_as_device_describes_them(),
	    },
	    engine::testing::Case{
	        .name = "an image error reads as image describes it",
	        .holds = image_errors_read_as_image_describes_them(),
	    },
	};
	return engine::testing::report(cases);
}
