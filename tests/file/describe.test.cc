// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.file;
import engine.testing;

namespace {

namespace file = engine::file;

constexpr auto failures = std::array{
    file::Failure::NotFound, file::Failure::Permission, file::Failure::NotRegular, file::Failure::Empty,
    file::Failure::Exhausted, file::Failure::MapFailed, file::Failure::Unreadable,
};

/** Every failure has a description, and no two share one: the text tells the operator which one happened. */
[[nodiscard]] auto every_failure_reads_differently() -> bool {
	auto texts = std::set<std::string>{};
	for (auto const failure : failures) {
		auto const text = file::describe(file::FileError{.failure = failure});
		if (text.empty() || !texts.insert(text).second) {
			return false;
		}
	}
	return texts.size() == failures.size() && std::to_underlying(failures.back()) + std::size_t{1} == failures.size();
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "every file failure has its own description", .holds = every_failure_reads_differently()},
	};
	return engine::testing::report(cases);
}
