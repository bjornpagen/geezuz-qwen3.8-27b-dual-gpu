// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.file;
import engine.testing;

namespace {

using engine::file::Failure;
using engine::file::FileError;
using engine::file::Mapping;

// ctest runs the test in its build directory, where the test's CMakeLists.txt wrote these inputs.
[[nodiscard]] auto input(std::string_view name) -> std::filesystem::path {
	return std::filesystem::current_path() / name;
}

[[nodiscard]] auto text(std::span<std::byte const> bytes) -> std::string {
	auto out = std::string{};
	out.reserve(bytes.size());
	for (auto const byte : bytes) {
		out.push_back(static_cast<char>(byte));
	}
	return out;
}

[[nodiscard]] auto fails_with(std::filesystem::path const& path, Failure failure) -> bool {
	auto const mapped = Mapping::open(path);
	return !mapped.has_value() && mapped.error() == FileError{.failure = failure};
}

[[nodiscard]] auto maps_exact_bytes() -> bool {
	auto const mapped = Mapping::open(input("known"));
	return mapped.has_value() && text(mapped->bytes()) == "mapped bytes\n\tare exactly the file";
}

[[nodiscard]] auto move_transfers_the_bytes() -> bool {
	auto mapped = Mapping::open(input("known"));
	if (!mapped) {
		return false;
	}
	auto const before = text(mapped->bytes());
	auto moved = std::move(*mapped);
	auto assigned = Mapping::open(input("known"));
	if (!assigned) {
		return false;
	}
	*assigned = std::move(moved);
	return text(assigned->bytes()) == before;
}

[[nodiscard]] auto classification_is_pinned() -> bool {
	auto holds = true;
	for (auto const failure :
	     {Failure::NotFound, Failure::Permission, Failure::NotRegular, Failure::Empty, Failure::MapFailed, Failure::Unreadable}) {
		holds = holds && FileError{.failure = failure}.retry() == engine::core::Retry::Permanent;
	}
	return holds && FileError{.failure = Failure::Exhausted}.retry() == engine::core::Retry::Transient;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "a mapping holds exactly the file's bytes", .holds = maps_exact_bytes()},
	    engine::testing::Case{.name = "a mapping moved over another keeps its bytes", .holds = move_transfers_the_bytes()},
	    engine::testing::Case{.name = "a missing path is not found", .holds = fails_with(input("absent"), Failure::NotFound)},
	    engine::testing::Case{.name = "a path through a file is not found",
		                      .holds = fails_with(input("known") / "below", Failure::NotFound)},
	    engine::testing::Case{.name = "a directory is not a regular file",
		                      .holds = fails_with(std::filesystem::current_path(), Failure::NotRegular)},
	    engine::testing::Case{.name = "an empty file is refused as empty", .holds = fails_with(input("empty"), Failure::Empty)},
	    engine::testing::Case{.name = "an unreadable file is a permission failure",
		                      .holds = fails_with(input("unreadable"), Failure::Permission)},
	    engine::testing::Case{.name = "only exhaustion is transient", .holds = classification_is_pinned()},
	};
	return engine::testing::report(cases);
}
