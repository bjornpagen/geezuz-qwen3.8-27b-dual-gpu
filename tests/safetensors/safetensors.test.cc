// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.safetensors;
import engine.testing;

namespace {

namespace safetensors = engine::safetensors;

using safetensors::Dtype;
using safetensors::Failure;

/** A tensor the generator lays out, with its data offsets relative to the data section. */
struct Planned {
	std::string name;
	std::string dtype;
	std::vector<std::size_t> shape;
	std::size_t begin;
	std::size_t end;
};

[[nodiscard]] auto dtype_name(Dtype dtype) -> std::string_view {
	switch (dtype) {
	case Dtype::Bf16:
		return "BF16";
	case Dtype::F8E4M3:
		return "F8_E4M3";
	}
	std::unreachable();
}

/** A small random tensor list, laid out back to back in a random order, as a writer may. */
[[nodiscard]] auto plan(std::mt19937_64& random) -> std::vector<Planned> {
	auto const count = std::uniform_int_distribution<std::size_t>{1, 12}(random);
	auto tensors = std::vector<Planned>{};
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const dtype = std::bernoulli_distribution{0.5}(random) ? Dtype::Bf16 : Dtype::F8E4M3;
		auto const rank = std::uniform_int_distribution<std::size_t>{0, 3}(random);
		auto shape = std::vector<std::size_t>{};
		auto bytes = safetensors::element_bytes(dtype);
		for (auto axis = std::size_t{0}; axis < rank; ++axis) {
			shape.push_back(std::uniform_int_distribution<std::size_t>{0, 5}(random));
			bytes *= shape.back();
		}
		tensors.push_back(
		    {.name = std::format("t{}.w", index), .dtype = std::string{dtype_name(dtype)}, .shape = shape, .begin = 0, .end = bytes});
	}
	std::ranges::shuffle(tensors, random);
	auto offset = std::size_t{0};
	for (auto& tensor : tensors) {
		auto const bytes = tensor.end;
		tensor.begin = offset;
		tensor.end = offset + bytes;
		offset = tensor.end;
	}
	std::ranges::shuffle(tensors, random);
	return tensors;
}

[[nodiscard]] auto header_text(std::span<Planned const> tensors, std::string_view metadata) -> std::string {
	auto text = std::string{"{"};
	if (!metadata.empty()) {
		text += std::format("\"__metadata__\":{},", metadata);
	}
	for (auto const& tensor : tensors) {
		auto shape = std::string{};
		for (auto const extent : tensor.shape) {
			shape += std::format("{}{}", shape.empty() ? "" : ",", extent);
		}
		text += std::format(R"("{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}},)", tensor.name, tensor.dtype, shape, tensor.begin,
		                    tensor.end);
	}
	text.back() = '}';
	return text;
}

/** A whole file: the 8-byte little-endian header length, the header padded with spaces, then the data bytes. */
[[nodiscard]] auto encode(std::string_view header, std::size_t data_bytes, std::size_t padding) -> std::vector<std::byte> {
	auto const length = header.size() + padding;
	auto file = std::vector<std::byte>{};
	for (auto index = std::size_t{0}; index < 8; ++index) {
		file.push_back(static_cast<std::byte>((length >> (std::size_t{8} * index)) & std::size_t{0xff}));
	}
	for (auto const character : header) {
		file.push_back(static_cast<std::byte>(character));
	}
	file.insert(file.end(), padding, std::byte{' '});
	file.insert(file.end(), data_bytes, std::byte{0x5a});
	return file;
}

[[nodiscard]] auto data_size(std::span<Planned const> tensors) -> std::size_t {
	auto total = std::size_t{0};
	for (auto const& tensor : tensors) {
		total = std::max(total, tensor.end);
	}
	return total;
}

[[nodiscard]] auto failure_of(std::span<std::byte const> file) -> std::optional<Failure> {
	auto const decoded = safetensors::decode(file);
	if (decoded) {
		return std::nullopt;
	}
	return decoded.error().failure;
}

constexpr auto seeds = std::array<std::uint64_t, 4>{1, 2, 3, 0x5afe'7e45};
constexpr auto rounds = 200;

/** Every generated file decodes to its tensors, in data order, at absolute file offsets, with its metadata. */
[[nodiscard]] auto decode_recovers_every_tensor(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto round = 0; round < rounds; ++round) {
		auto const tensors = plan(random);
		auto const header = header_text(tensors, R"({"format":"pt"})");
		auto const padding = std::uniform_int_distribution<std::size_t>{0, 7}(random);
		auto const file = encode(header, data_size(tensors), padding);
		auto const decoded = safetensors::decode(file);
		if (!decoded) {
			return false;
		}
		auto expected = tensors;
		std::ranges::sort(expected, std::ranges::less{}, [](Planned const& tensor) {
			return std::pair{tensor.begin, tensor.end};
		});
		auto const data_start = 8 + header.size() + padding;
		auto const matches =
		    std::ranges::equal(decoded->tensors, expected, [data_start](safetensors::Tensor const& got, Planned const& want) {
			    return got.name == want.name && dtype_name(got.dtype) == want.dtype && got.shape == want.shape &&
				       got.bytes == safetensors::ByteRange{.begin = data_start + want.begin, .end = data_start + want.end};
		    });
		auto const metadata = decoded->metadata == std::vector{safetensors::Metadata{.key = "format", .value = "pt"}};
		if (!matches || !metadata) {
			return false;
		}
	}
	return true;
}

/** A mutation of one generated layout, and the failure the format law requires for it. */
struct Breakage {
	std::string_view what;
	Failure failure;
};

[[nodiscard]] auto breaks(std::uint64_t seed, Breakage const& breakage) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto round = 0; round < rounds; ++round) {
		auto tensors = plan(random);
		auto const victim = std::uniform_int_distribution<std::size_t>{0, tensors.size() - 1}(random);
		auto extra = std::size_t{0};
		auto metadata = std::string{};
		if (breakage.what == "gap") {
			auto const opening = tensors[victim].begin;
			for (auto& tensor : tensors) {
				auto const shift = tensor.begin >= opening ? std::size_t{1} : std::size_t{0};
				tensor.begin += shift;
				tensor.end += shift;
			}
		} else if (breakage.what == "size") {
			tensors[victim].end += 1;
		} else if (breakage.what == "dtype") {
			tensors[victim].dtype = "F32";
		} else if (breakage.what == "duplicate") {
			tensors.push_back(tensors[victim]);
			auto const bytes = tensors.back().end - tensors.back().begin;
			tensors.back().begin = data_size(tensors);
			tensors.back().end = tensors.back().begin + bytes;
		} else if (breakage.what == "leftover") {
			extra = 1;
		} else if (breakage.what == "metadata") {
			metadata = R"({"format":1})";
		}
		auto const header = header_text(tensors, metadata);
		auto const file = encode(header, data_size(tensors) + extra, 0);
		if (failure_of(file) != breakage.failure) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto refuses_a_file_shorter_than_its_header() -> bool {
	auto const file = encode(R"({"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})", 2, 0);
	auto const cut = std::span{file}.first(file.size() - 2 - 1);
	return failure_of(std::span{file}.first(7)) == Failure::Truncated && failure_of(cut) == Failure::Truncated;
}

[[nodiscard]] auto refuses_an_oversized_header_length() -> bool {
	auto file = encode("{}", 0, 0);
	auto const length = safetensors::max_header_bytes + 1;
	for (auto index = std::size_t{0}; index < 8; ++index) {
		file[index] = static_cast<std::byte>((length >> (std::size_t{8} * index)) & std::size_t{0xff});
	}
	file.resize(8 + length, std::byte{' '});
	return failure_of(file) == Failure::HeaderTooLarge;
}

[[nodiscard]] auto refuses_headers_that_are_not_tensor_tables() -> bool {
	auto const refused = std::array<std::pair<std::string_view, Failure>, 6>{{
	    {R"({"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]})", Failure::HeaderNotJson},
	    {R"([1,2])", Failure::MalformedEntry},
	    {R"({"a":{"shape":[1],"data_offsets":[0,2]}})", Failure::MalformedEntry},
	    {R"({"a":{"dtype":"BF16","shape":[-1],"data_offsets":[0,2]}})", Failure::MalformedEntry},
	    {R"({"a":{"dtype":"BF16","shape":[1],"data_offsets":[0]}})", Failure::MalformedEntry},
	    {R"({"a":{"dtype":"BF16","shape":[4611686018427387904,4],"data_offsets":[0,2]}})", Failure::ShapeOverflow},
	}};
	return std::ranges::all_of(refused, [](auto const& item) {
		return failure_of(encode(item.first, 2, 0)) == item.second;
	});
}

}

auto main() -> int {
	auto cases = std::vector<engine::testing::Case>{};
	for (auto const seed : seeds) {
		cases.push_back({.name = "a generated file decodes to its tensors in data order at absolute offsets (seeds 1, 2, 3, 0x5afe7e45)",
		                 .holds = decode_recovers_every_tensor(seed)});
		cases.push_back({.name = "a gap in the data section is refused as untiled",
		                 .holds = breaks(seed, {.what = "gap", .failure = Failure::Untiled})});
		cases.push_back({.name = "bytes past the last tensor are refused as untiled",
		                 .holds = breaks(seed, {.what = "leftover", .failure = Failure::Untiled})});
		cases.push_back({.name = "a range not holding its shape's bytes is refused",
		                 .holds = breaks(seed, {.what = "size", .failure = Failure::SizeMismatch})});
		cases.push_back({.name = "a dtype the checkpoint does not hold is refused",
		                 .holds = breaks(seed, {.what = "dtype", .failure = Failure::UnknownDtype})});
		cases.push_back({.name = "two entries with one name are refused",
		                 .holds = breaks(seed, {.what = "duplicate", .failure = Failure::DuplicateName})});
		cases.push_back({.name = "metadata that is not all strings is refused",
		                 .holds = breaks(seed, {.what = "metadata", .failure = Failure::MalformedMetadata})});
	}
	cases.push_back(
	    {.name = "a file shorter than its length field or its header is refused", .holds = refuses_a_file_shorter_than_its_header()});
	cases.push_back(
	    {.name = "a header length beyond the bound is refused before reading it", .holds = refuses_an_oversized_header_length()});
	cases.push_back({.name = "headers that are not json, or not tables of well-formed entries, are refused by kind",
	                 .holds = refuses_headers_that_are_not_tensor_tables()});
	return engine::testing::report(cases);
}
