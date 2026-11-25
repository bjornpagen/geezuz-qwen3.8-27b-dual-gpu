// SPDX-License-Identifier: AGPL-3.0-only
export module engine.safetensors:decode;

import std;
import engine.core;
import engine.json;
import :error;
import :layout;

namespace engine::safetensors {

/**
 * A header is one object of tensor entries, each an object holding a string and two arrays of integers, plus a
 * metadata object of strings. Offsets and extents are at most 20 digits; the checkpoint's largest file holds 336
 * tensors.
 */
inline constexpr json::Bounds header_bounds{
    .document_bytes = max_header_bytes,
    .depth = 3,
    .string_bytes = 4096,
    .members = 65'536,
    .number_chars = 20,
};

static_assert(json::Enforceable<header_bounds>);

using Reader = json::Reader<header_bounds>;
using Value = json::Value<header_bounds>;

[[nodiscard]] auto refuse(Failure failure) -> std::unexpected<Error> {
	return std::unexpected(Error{.failure = failure});
}

/** Why the strict whole-header reading failed: a repeated name is the format's own failure, the rest is not json. */
[[nodiscard]] auto unreadable(json::ReadError const& error) -> Failure {
	auto const* const parse = std::get_if<json::ParseError>(&error.cause);
	return parse != nullptr && parse->fault == json::ParseFault::DuplicateKey ? Failure::DuplicateName : Failure::HeaderNotJson;
}

/** An array of non-negative integers that fit a size. */
[[nodiscard]] auto sizes(Value value) -> core::Result<std::vector<std::size_t>, Error> {
	auto array = value.as_array();
	if (!array) {
		return refuse(Failure::MalformedEntry);
	}
	auto result = std::vector<std::size_t>{};
	while (true) {
		auto element = array->next();
		if (!element) {
			return refuse(Failure::MalformedEntry);
		}
		if (!*element) {
			return result;
		}
		auto const integer = (*element)->as_integer();
		if (!integer) {
			return refuse(Failure::MalformedEntry);
		}
		auto const size = core::try_narrow<std::size_t>(*integer);
		if (!size) {
			return refuse(Failure::MalformedEntry);
		}
		result.push_back(*size);
	}
}

[[nodiscard]] auto read_entry(std::string_view name, Value value) -> core::Result<Entry, Error> {
	auto object = value.as_object();
	if (!object) {
		return refuse(Failure::MalformedEntry);
	}
	auto dtype = std::optional<Dtype>{};
	auto shape = std::optional<std::vector<std::size_t>>{};
	auto offsets = std::optional<std::vector<std::size_t>>{};
	while (true) {
		auto member = object->next();
		if (!member) {
			return refuse(Failure::MalformedEntry);
		}
		if (!*member) {
			break;
		}
		auto& [field, field_value] = **member;
		if (field == "dtype") {
			auto const text = field_value.as_string();
			if (!text) {
				return refuse(Failure::MalformedEntry);
			}
			dtype = dtype_named(*text);
			if (!dtype) {
				return refuse(Failure::UnknownDtype);
			}
		} else if (field == "shape") {
			auto read = sizes(std::move(field_value));
			if (!read) {
				return std::unexpected(read.error());
			}
			shape = std::move(*read);
		} else if (field == "data_offsets") {
			auto read = sizes(std::move(field_value));
			if (!read) {
				return std::unexpected(read.error());
			}
			offsets = std::move(*read);
		} else {
			return refuse(Failure::MalformedEntry);
		}
	}
	if (!dtype || !shape || !offsets || offsets->size() != 2) {
		return refuse(Failure::MalformedEntry);
	}
	return Entry{.name = std::string{name}, .dtype = *dtype, .shape = std::move(*shape), .begin = (*offsets)[0], .end = (*offsets)[1]};
}

[[nodiscard]] auto read_metadata(Value value) -> core::Result<std::vector<Metadata>, Error> {
	auto object = value.as_object();
	if (!object) {
		return refuse(Failure::MalformedMetadata);
	}
	auto metadata = std::vector<Metadata>{};
	while (true) {
		auto member = object->next();
		if (!member) {
			return refuse(Failure::MalformedMetadata);
		}
		if (!*member) {
			return metadata;
		}
		auto const text = (*member)->value.as_string();
		if (!text) {
			return refuse(Failure::MalformedMetadata);
		}
		metadata.push_back({.key = std::string{(*member)->name}, .value = std::string{*text}});
	}
}

/**
 * Decodes a whole safetensors file: its tensors in data order, each with its byte range in the file, and its
 * metadata. The header must be strict json within the header bounds; its entries must tile the data section.
 */
export [[nodiscard]] auto decode(std::span<std::byte const> file) -> core::Result<File, Error> {
	auto const length = header_length(file);
	if (!length) {
		return std::unexpected(length.error());
	}
	auto header = std::string(*length, '\0');
	std::ranges::transform(file.subspan(length_bytes, *length), header.begin(), [](std::byte byte) {
		return static_cast<char>(byte);
	});
	auto reader = Reader::make();
	if (!reader) {
		return refuse(Failure::HeaderNotJson);
	}
	if (auto const strict = json::check_text(*reader, header); !strict) {
		return refuse(unreadable(strict.error()));
	}
	auto root = reader->read_copy(header);
	if (!root) {
		return refuse(Failure::HeaderNotJson);
	}
	auto object = root->as_object();
	if (!object) {
		return refuse(Failure::MalformedEntry);
	}
	auto entries = std::vector<Entry>{};
	auto metadata = std::vector<Metadata>{};
	while (true) {
		auto member = object->next();
		if (!member) {
			return refuse(Failure::MalformedEntry);
		}
		if (!*member) {
			break;
		}
		auto& [name, value] = **member;
		// The one member that is not a tensor: the format reserves its name.
		if (name == "__metadata__") {
			auto read = read_metadata(std::move(value));
			if (!read) {
				return std::unexpected(read.error());
			}
			metadata = std::move(*read);
			continue;
		}
		auto entry = read_entry(name, std::move(value));
		if (!entry) {
			return std::unexpected(entry.error());
		}
		entries.push_back(std::move(*entry));
	}
	auto tensors = arrange(std::move(entries), length_bytes + *length, file.size());
	if (!tensors) {
		return std::unexpected(tensors.error());
	}
	return File{.tensors = std::move(*tensors), .metadata = std::move(metadata)};
}

}
