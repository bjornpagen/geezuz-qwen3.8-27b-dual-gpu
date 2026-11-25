// SPDX-License-Identifier: AGPL-3.0-only
//
// The bring-up every program that serves the model runs, in the order that refuses earliest: the machine (the driver,
// two bridged 3090s), then the memory clock under load against the one a soak accepted, and only then the images,
// opened and verified against the checkpoint and copied to the cards, prefill's programs loaded, and the loops started.
export module engine.cards:start;

import std;
import engine.bake;
import engine.core;
import engine.decode;
import engine.device;
import engine.image;
import engine.platform;
import :adapter;

namespace engine::cards {

/** Why the cards could not be brought up: the machine, its memory clock, or the images and the loops. */
export struct [[nodiscard]] StartError {
	std::variant<device::DeviceError, platform::ClockError, decode::DecodeError> reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(StartError const&) const -> bool = default;
};

/** Each reason in its owner's words, under the step that failed. */
struct Describe {
	[[nodiscard]] auto operator()(device::DeviceError const& reason) const -> std::string {
		return std::format("the cards: {}", device::describe(reason));
	}

	[[nodiscard]] auto operator()(platform::ClockError const& reason) const -> std::string {
		return std::format("the memory clock: {}", platform::describe(reason));
	}

	[[nodiscard]] auto operator()(decode::DecodeError const& reason) const -> std::string {
		return std::format("the images and the loops: {}", decode::describe(reason));
	}
};

/** The failure as one line. */
export [[nodiscard]] auto describe(StartError const& error) -> std::string {
	return std::visit(Describe{}, error.reason);
}

/** The machine's time for the clock check: real waits on the steady clock. */
struct MachineClock {
	[[nodiscard]] auto now() const -> platform::Moment {
		return {.wall = std::chrono::system_clock::now(), .steady = std::chrono::steady_clock::now()};
	}

	auto sleep_until(std::chrono::steady_clock::time_point until) const -> void {
		std::this_thread::sleep_until(until);
	}
};

static_assert(platform::Clock<MachineClock>);

/** How long the images may take to reach the cards: about 14 GB each over pcie, seconds when all is well. */
inline constexpr auto upload_patience = std::chrono::minutes{10};

/** Where the baker wrote the three images in `directory`. */
export [[nodiscard]] auto image_paths(std::filesystem::path const& directory) -> decode::Paths {
	return {
	    .card_zero = directory / bake::image_name(image::Target::CardZero),
	    .card_one = directory / bake::image_name(image::Target::CardOne),
	    .host = directory / bake::image_name(image::Target::Host),
	};
}

/** Brings both cards up from the images in `directory`, serving: the machine, its clock, the images, the loops. */
export [[nodiscard]] auto start(std::filesystem::path const& directory) -> core::Result<Cards, StartError> {
	auto devices = device::start();
	if (!devices) {
		return std::unexpected(StartError{.reason = devices.error()});
	}
	auto clock = MachineClock{};
	if (auto const clocks = platform::check_memory_clocks(*devices, clock); !clocks) {
		return std::unexpected(StartError{.reason = clocks.error()});
	}
	auto opened = decode::Decode::open(std::move(*devices), image_paths(directory), bake::checkpoint,
	                                   std::chrono::steady_clock::now() + upload_patience);
	if (!opened) {
		return std::unexpected(StartError{.reason = opened.error()});
	}
	return Cards::assemble(std::move(*opened)).transform_error([](decode::DecodeError failed) {
		return StartError{.reason = std::move(failed)};
	});
}

}
