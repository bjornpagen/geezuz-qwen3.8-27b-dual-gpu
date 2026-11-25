// SPDX-License-Identifier: AGPL-3.0-only
export module engine.safetensors:layout;

import std;
import engine.core;
import :error;

namespace engine::safetensors {

/** The element types the checkpoint holds; any other dtype is refused at decode. */
export enum class Dtype : std::uint8_t {
	Bf16,
	F8E4M3,
};

export [[nodiscard]] constexpr auto element_bytes(Dtype dtype) -> std::size_t {
	switch (dtype) {
	case Dtype::Bf16:
		return 2;
	case Dtype::F8E4M3:
		return 1;
	}
	std::unreachable();
}

[[nodiscard]] auto dtype_named(std::string_view name) -> std::optional<Dtype> {
	if (name == "BF16") {
		return Dtype::Bf16;
	}
	if (name == "F8_E4M3") {
		return Dtype::F8E4M3;
	}
	return std::nullopt;
}

/** A half-open byte range of the whole file, so a reader slices the mapped file with it directly. */
export struct ByteRange {
	std::size_t begin;
	std::size_t end;

	[[nodiscard]] constexpr auto operator==(ByteRange const&) const -> bool = default;
};

/** One tensor: row-major, little-endian, its bytes exactly `bytes`. */
export struct Tensor {
	std::string name;
	Dtype dtype;
	std::vector<std::size_t> shape;
	ByteRange bytes;

	[[nodiscard]] auto operator==(Tensor const&) const -> bool = default;
};

export struct Metadata {
	std::string key;
	std::string value;

	[[nodiscard]] auto operator==(Metadata const&) const -> bool = default;
};

/** A decoded file: its tensors in data order, tiling the data section, and its free-form metadata. */
export struct File {
	std::vector<Tensor> tensors;
	std::vector<Metadata> metadata;
};

/**
 * The longest header the engine reads. The format allows far larger; the checkpoint's largest is under 40 KB, and a
 * header is parsed whole into memory.
 */
export inline constexpr std::size_t max_header_bytes = std::size_t{1024} * 1024;

inline constexpr std::size_t length_bytes = 8;

/** A tensor entry as the header states it: data offsets relative to the data section, not yet checked. */
struct Entry {
	std::string name;
	Dtype dtype;
	std::vector<std::size_t> shape;
	std::size_t begin;
	std::size_t end;
};

/** The header's byte length, read from the file's first eight bytes (little-endian), checked to fit the file. */
[[nodiscard]] auto header_length(std::span<std::byte const> file) -> core::Result<std::size_t, Error> {
	if (file.size() < length_bytes) {
		return std::unexpected(Error{.failure = Failure::Truncated});
	}
	auto length = std::uint64_t{0};
	for (auto index = std::size_t{0}; index < length_bytes; ++index) {
		length |= std::to_integer<std::uint64_t>(file[index]) << (std::size_t{8} * index);
	}
	if (length > max_header_bytes) {
		return std::unexpected(Error{.failure = Failure::HeaderTooLarge});
	}
	auto const header = core::narrow<std::size_t>(length);
	if (file.size() - length_bytes < header) {
		return std::unexpected(Error{.failure = Failure::Truncated});
	}
	return header;
}

[[nodiscard]] auto shape_bytes(std::span<std::size_t const> shape, Dtype dtype) -> std::optional<std::size_t> {
	auto total = element_bytes(dtype);
	for (auto const extent : shape) {
		if (extent != 0 && total > std::numeric_limits<std::size_t>::max() / extent) {
			return std::nullopt;
		}
		total *= extent;
	}
	return total;
}

/**
 * Checks the entries against the format's layout law and places them in the file: each range holds exactly its
 * shape's bytes, and the ranges, in offset order, tile the data section from its first byte to the file's last. The
 * header's json reading has already refused repeated names.
 */
[[nodiscard]] auto arrange(std::vector<Entry> entries, std::size_t data_start, std::size_t file_size)
    -> core::Result<std::vector<Tensor>, Error> {
	for (auto const& entry : entries) {
		auto const bytes = shape_bytes(entry.shape, entry.dtype);
		if (!bytes) {
			return std::unexpected(Error{.failure = Failure::ShapeOverflow});
		}
		if (entry.end < entry.begin || entry.end - entry.begin != *bytes) {
			return std::unexpected(Error{.failure = Failure::SizeMismatch});
		}
	}
	std::ranges::sort(entries, std::ranges::less{}, [](Entry const& entry) {
		return std::pair{entry.begin, entry.end};
	});
	auto const data_size = file_size - data_start;
	auto next = std::size_t{0};
	auto tensors = std::vector<Tensor>{};
	tensors.reserve(entries.size());
	for (auto& entry : entries) {
		if (entry.begin != next) {
			return std::unexpected(Error{.failure = Failure::Untiled});
		}
		next = entry.end;
		if (next > data_size) {
			return std::unexpected(Error{.failure = Failure::Untiled});
		}
		tensors.push_back(Tensor{
		    .name = std::move(entry.name),
		    .dtype = entry.dtype,
		    .shape = std::move(entry.shape),
		    .bytes = {.begin = data_start + entry.begin, .end = data_start + entry.end},
		});
	}
	if (next != data_size) {
		return std::unexpected(Error{.failure = Failure::Untiled});
	}
	return tensors;
}

}
