// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.file;
import engine.image;
import engine.testing;

namespace {

namespace file = engine::file;
namespace image = engine::image;

/** Every image error there is: each mapping failure and each rejection. */
[[nodiscard]] auto every_error() -> std::vector<image::ImageError> {
	auto errors = std::vector<image::ImageError>{};
	for (auto code = 0; std::cmp_less_equal(code, std::to_underlying(file::Failure::Unreadable)); ++code) {
		errors.push_back({.cause = file::FileError{.failure = static_cast<file::Failure>(code)}});
	}
	for (auto code = 0; std::cmp_less_equal(code, std::to_underlying(image::Failure::PayloadMismatch)); ++code) {
		errors.push_back({.cause = image::Rejection{.failure = static_cast<image::Failure>(code)}});
	}
	return errors;
}

/** Every image error has a description, and no two share one: the text tells the operator which one happened. */
[[nodiscard]] auto every_error_reads_differently() -> bool {
	auto const errors = every_error();
	auto texts = std::set<std::string>{};
	for (auto const& error : errors) {
		auto const text = image::describe(error);
		if (text.empty() || !texts.insert(text).second) {
			return false;
		}
	}
	return texts.size() == errors.size();
}

/** A mapping failure reads as engine.file describes it, inside the image's description. */
[[nodiscard]] auto mapping_failures_read_as_file_describes_them() -> bool {
	return std::ranges::all_of(every_error(), [](image::ImageError const& error) {
		auto const mapping = engine::core::alternative<file::FileError>(error.cause);
		return !mapping || image::describe(error).contains(file::describe(*mapping));
	});
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "every image error has its own description", .holds = every_error_reads_differently()},
	    engine::testing::Case{.name = "a mapping failure reads as engine.file describes it", .holds = mapping_failures_read_as_file_describes_them()},
	};
	return engine::testing::report(cases);
}
