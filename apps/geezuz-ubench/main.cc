// SPDX-License-Identifier: AGPL-3.0-only
//
// geezuz-ubench [benchmark...]: runs the device microbenchmarks at bring-up, every one or the ones named, and prints
// each measurement beside the review's expectation and the decision it settles. Exit status 0 when every selected
// benchmark ran and none failed its gate; 1 otherwise, including on a machine without the driver (macOS), where the
// engine's start refuses with "driver unavailable"; 2 for a name that is not a benchmark.
import std;
import engine.core;
import engine.device;
import engine.ubench;

namespace ubench = engine::ubench;

auto main(int argc, char** argv) -> int {
	auto const words = std::span{argv, static_cast<std::size_t>(argc)};
	auto names = std::vector<std::string_view>{};
	for (auto const* word : words.subspan(1)) {
		names.emplace_back(word);
	}
	auto const selection = ubench::select(names);
	if (!selection) {
		std::println("geezuz-ubench: {} is not a benchmark; the benchmarks are:", selection.error().name);
		for (auto const& listed : ubench::catalogue) {
			std::println("  {}", listed.name);
		}
		return 2;
	}
	auto const devices = engine::device::start();
	if (!devices) {
		std::println("geezuz-ubench: {}", ubench::describe(ubench::UbenchError{.reason = devices.error()}));
		return 1;
	}
	auto const ran = ubench::run_all(*devices, *selection);
	if (!ran) {
		std::println("geezuz-ubench: {}", ubench::describe(ran.error()));
		return 1;
	}
	return *ran ? 0 : 1;
}
