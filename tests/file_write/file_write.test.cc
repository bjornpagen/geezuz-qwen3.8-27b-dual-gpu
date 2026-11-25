// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.file;
import engine.file_write;
import engine.testing;

namespace {

using engine::file_write::Failure;
using engine::file_write::Output;
using engine::file_write::WriteError;

/** A fresh, empty directory of the test's own under the directory ctest runs it in. */
[[nodiscard]] auto scratch(std::string_view name) -> std::filesystem::path {
	auto const directory = std::filesystem::current_path() / "file_write.scratch" / name;
	std::filesystem::remove_all(directory);
	std::filesystem::create_directories(directory);
	return directory;
}

[[nodiscard]] auto entries(std::filesystem::path const& directory) -> std::size_t {
	return static_cast<std::size_t>(std::distance(std::filesystem::directory_iterator{directory}, std::filesystem::directory_iterator{}));
}

[[nodiscard]] auto contents(std::filesystem::path const& path) -> std::optional<std::vector<std::byte>> {
	auto const mapped = engine::file::Mapping::open(path);
	if (!mapped) {
		return std::nullopt;
	}
	return std::vector<std::byte>{mapped->bytes().begin(), mapped->bytes().end()};
}

/** Writes random pieces at random offsets and checks the committed file against a model: written bytes, zeros elsewhere. */
[[nodiscard]] auto committed_file_is_the_writes_over_zeros(std::uint64_t seed) -> bool {
	auto const directory = scratch(std::format("model-{}", seed));
	auto random = engine::testing::generator(seed);
	auto const size = std::uniform_int_distribution<std::size_t>{1, std::size_t{1} << 20U}(random);
	auto model = std::vector<std::byte>(size, std::byte{0});
	auto const destination = directory / "image";
	auto output = Output::create(destination, size);
	if (!output) {
		return false;
	}
	for (auto write = 0; write < 64; ++write) {
		auto const offset = std::uniform_int_distribution<std::size_t>{0, size - 1}(random);
		auto const length = std::uniform_int_distribution<std::size_t>{0, std::min<std::size_t>(size - offset, std::size_t{1} << 16U)}(random);
		auto piece = std::vector<std::byte>(length);
		std::ranges::generate(piece, [&random] { return std::byte{static_cast<std::uint8_t>(random() & 0xffU)}; });
		std::ranges::copy(piece, std::next(model.begin(), static_cast<std::ptrdiff_t>(offset)));
		if (!output->write(offset, piece)) {
			return false;
		}
	}
	auto const before = std::filesystem::exists(destination);
	if (!std::move(*output).commit()) {
		return false;
	}
	return !before && contents(destination) == model && entries(directory) == 1;
}

[[nodiscard]] auto dropped_output_leaves_nothing() -> bool {
	auto const directory = scratch("dropped");
	{
		auto output = Output::create(directory / "image", 4096);
		if (!output || !output->write(0, std::vector<std::byte>(100, std::byte{7}))) {
			return false;
		}
	}
	return entries(directory) == 0;
}

[[nodiscard]] auto commit_replaces_an_existing_file_whole() -> bool {
	auto const directory = scratch("replace");
	auto const destination = directory / "image";
	for (auto const fill : {std::byte{1}, std::byte{2}}) {
		auto output = Output::create(destination, 3000);
		if (!output || !output->write(0, std::vector<std::byte>(3000, fill)) || !std::move(*output).commit()) {
			return false;
		}
	}
	return contents(destination) == std::vector<std::byte>(3000, std::byte{2}) && entries(directory) == 1;
}

[[nodiscard]] auto missing_directory_is_not_found() -> bool {
	auto const output = Output::create(scratch("missing") / "absent" / "image", 10);
	return !output && output.error() == WriteError{.failure = Failure::NotFound};
}

[[nodiscard]] auto directory_destination_is_refused_and_cleaned() -> bool {
	auto const directory = scratch("directory");
	std::filesystem::create_directory(directory / "image");
	auto output = Output::create(directory / "image", 10);
	if (!output) {
		return false;
	}
	auto const committed = std::move(*output).commit();
	return !committed && committed.error() == WriteError{.failure = Failure::IsDirectory} && entries(directory) == 1 &&
	       std::filesystem::is_directory(directory / "image");
}

[[nodiscard]] auto classification_is_pinned() -> bool {
	auto holds = true;
	for (auto const failure : {Failure::NotFound, Failure::Permission, Failure::IsDirectory, Failure::Io}) {
		holds = holds && WriteError{.failure = failure}.retry() == engine::core::Retry::Permanent;
	}
	for (auto const failure : {Failure::NoSpace, Failure::Exhausted}) {
		holds = holds && WriteError{.failure = failure}.retry() == engine::core::Retry::Transient;
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "a committed file is every write over zeros (seed 1)", .holds = committed_file_is_the_writes_over_zeros(1)},
	    engine::testing::Case{.name = "a committed file is every write over zeros (seed 2)", .holds = committed_file_is_the_writes_over_zeros(2)},
	    engine::testing::Case{.name = "a committed file is every write over zeros (seed 3)", .holds = committed_file_is_the_writes_over_zeros(3)},
	    engine::testing::Case{.name = "an output dropped before commit leaves no file at all", .holds = dropped_output_leaves_nothing()},
	    engine::testing::Case{.name = "commit replaces an existing destination with the whole new file",
	                          .holds = commit_replaces_an_existing_file_whole()},
	    engine::testing::Case{.name = "a destination in a missing directory is NotFound", .holds = missing_directory_is_not_found()},
	    engine::testing::Case{.name = "a directory at the destination is refused at commit and the temporary is removed",
	                          .holds = directory_destination_is_refused_and_cleaned()},
	    engine::testing::Case{.name = "space and descriptor exhaustion are transient, the rest permanent", .holds = classification_is_pinned()},
	};
	return engine::testing::report(cases);
}
