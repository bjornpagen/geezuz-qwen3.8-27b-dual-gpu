// SPDX-License-Identifier: AGPL-3.0-only
//
// The fp8 checkpoint as the reference reads it: every safetensors file mapped, every tensor the model description
// names found, its dtype and shape checked, and every e4m3 code checked finite, once, at open. Past `open` a tensor is
// a byte range of a mapping the checkpoint owns, so nothing is copied and the 108 GB fp32 model never exists.
export module engine.ref:checkpoint;

import std;
import engine.core;
import engine.file;
import engine.model;
import engine.safetensors;
import engine.schedule;
import engine.thread_pool;
import :numeric;

namespace engine::ref {

namespace model = engine::model;
namespace schedule = engine::schedule;
namespace safetensors = engine::safetensors;

using model::Count;

/** Why a directory is not the checkpoint the reference computes. */
export enum class CheckpointFailure : std::uint8_t {
	/** A safetensors file of the directory cannot be mapped. */
	Unreadable,
	/** A file is not a safetensors file the engine reads. */
	Malformed,
	/** A tensor the model description names is in no file. */
	Missing,
	/** A tensor is stored twice. */
	Duplicate,
	/** A tensor's dtype or shape is not the model description's. */
	Mismatched,
	/** A tensor the model description does not name, outside the dropped vision tower. */
	Unexpected,
	/** An e4m3 weight holds a nan code, which no weight of the model can be. */
	NanCode,
};

export struct [[nodiscard]] CheckpointError {
	CheckpointFailure failure;
	/** The file or tensor the failure is about. */
	std::string subject;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case CheckpointFailure::Unreadable:
		case CheckpointFailure::Malformed:
		case CheckpointFailure::Missing:
		case CheckpointFailure::Duplicate:
		case CheckpointFailure::Mismatched:
		case CheckpointFailure::Unexpected:
		case CheckpointFailure::NanCode:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] auto operator==(CheckpointError const&) const -> bool = default;
};

/** Bytes of one mapped file. */
struct Located {
	std::size_t file;
	safetensors::ByteRange bytes;
};

/** A stored tensor: its elements, and an e4m3 tensor's block scales. */
struct Stored {
	Located data;
	std::optional<Located> scales;
};

/** Which copy of a layer tensor: a trunk layer's, or (one past the trunk) the mtp head's. */
[[nodiscard]] constexpr auto copy_index(schedule::Instance at) -> Count {
	return at.stream == model::Stream::Mtp ? model::trunk_layers + at.layer : at.layer;
}

inline constexpr Count copies = model::trunk_layers + model::mtp_layers;

/** A tensor's name in the checkpoint, at `copy` (a trunk layer, or past the trunk an mtp layer). */
[[nodiscard]] auto checkpoint_name(model::Tensor const& tensor, Count copy) -> std::string {
	switch (tensor.scope) {
	case model::Scope::Model:
		return std::string{tensor.name};
	case model::Scope::Mtp:
		return std::format("{}{}", model::mtp_prefix, tensor.name);
	case model::Scope::DecoderLayer:
	case model::Scope::LinearAttentionLayer:
	case model::Scope::FullAttentionLayer:
		break;
	}
	if (copy < model::trunk_layers) {
		return std::format("{}{}.{}", model::trunk_layer_prefix, copy, tensor.name);
	}
	return std::format("{}{}.{}", model::mtp_layer_prefix, copy - model::trunk_layers, tensor.name);
}

/** Whether `copy` of the tensor exists: every layer holds its decoder tensors and its mixer's. */
[[nodiscard]] constexpr auto exists(model::Tensor const& tensor, Count copy) -> bool {
	switch (tensor.scope) {
	case model::Scope::Model:
	case model::Scope::Mtp:
		return copy == 0;
	case model::Scope::DecoderLayer:
	case model::Scope::LinearAttentionLayer:
	case model::Scope::FullAttentionLayer:
		break;
	}
	auto const mixer = copy < model::trunk_layers ? model::trunk_mixers[copy] : model::mtp_mixer;
	return model::in_layer(tensor.scope, mixer);
}

[[nodiscard]] auto extents(model::Shape const& shape) -> std::vector<std::size_t> {
	auto const all = std::span{shape.extents};
	return {all.begin(), all.begin() + static_cast<std::ptrdiff_t>(shape.rank)};
}

[[nodiscard]] constexpr auto stored_as(model::Format format) -> safetensors::Dtype {
	switch (format) {
	case model::Format::E4m3:
		return safetensors::Dtype::F8E4M3;
	case model::Format::Bf16:
		return safetensors::Dtype::Bf16;
	}
	std::unreachable();
}

struct Found {
	std::size_t file;
	safetensors::Tensor tensor;
};

using Catalog = std::map<std::string, Found, std::less<>>;

[[nodiscard]] auto take(Catalog& catalog, std::string const& name, safetensors::Dtype dtype, std::vector<std::size_t> const& shape)
    -> core::Result<Located, CheckpointError> {
	auto const found = catalog.find(name);
	if (found == catalog.end()) {
		return std::unexpected(CheckpointError{.failure = CheckpointFailure::Missing, .subject = name});
	}
	if (found->second.tensor.dtype != dtype || found->second.tensor.shape != shape) {
		return std::unexpected(CheckpointError{.failure = CheckpointFailure::Mismatched, .subject = name});
	}
	auto const located = Located{.file = found->second.file, .bytes = found->second.tensor.bytes};
	catalog.erase(found);
	return located;
}

/** The checkpoint's safetensors files, mapped in name order, and every tensor outside the vision tower they hold. */
struct Opened {
	std::vector<file::Mapping> files;
	Catalog catalog;
};

[[nodiscard]] auto safetensors_paths(std::filesystem::path const& directory) -> core::Result<std::vector<std::filesystem::path>, CheckpointError> {
	auto paths = std::vector<std::filesystem::path>{};
	auto listing = std::error_code{};
	for (auto const& item : std::filesystem::directory_iterator{directory, listing}) {
		if (item.path().extension() == ".safetensors") {
			paths.push_back(item.path());
		}
	}
	if (listing || paths.empty()) {
		return std::unexpected(CheckpointError{.failure = CheckpointFailure::Unreadable, .subject = directory.string()});
	}
	std::ranges::sort(paths);
	return paths;
}

[[nodiscard]] auto catalog_files(std::span<std::filesystem::path const> paths) -> core::Result<Opened, CheckpointError> {
	auto opened = Opened{};
	for (auto const& path : paths) {
		auto mapping = file::Mapping::open(path);
		if (!mapping) {
			return std::unexpected(CheckpointError{.failure = CheckpointFailure::Unreadable, .subject = path.string()});
		}
		auto decoded = safetensors::decode(mapping->bytes());
		if (!decoded) {
			return std::unexpected(CheckpointError{.failure = CheckpointFailure::Malformed, .subject = path.string()});
		}
		for (auto& tensor : decoded->tensors) {
			if (tensor.name.starts_with(model::excluded_prefix)) {
				continue;
			}
			auto name = tensor.name;
			auto const [place, fresh] = opened.catalog.try_emplace(std::move(name), Found{.file = opened.files.size(), .tensor = std::move(tensor)});
			if (!fresh) {
				return std::unexpected(CheckpointError{.failure = CheckpointFailure::Duplicate, .subject = place->first});
			}
		}
		opened.files.push_back(std::move(*mapping));
	}
	return opened;
}

/** One stored copy of `tensor`, taken out of the catalog with its scales. */
[[nodiscard]] auto take_stored(Catalog& catalog, model::Tensor const& tensor, Count copy) -> core::Result<Stored, CheckpointError> {
	auto const name = checkpoint_name(tensor, copy);
	auto data = take(catalog, name, stored_as(tensor.format), extents(tensor.shape));
	if (!data) {
		return std::unexpected(data.error());
	}
	if (tensor.format != model::Format::E4m3) {
		return Stored{.data = *data, .scales = std::nullopt};
	}
	return take(catalog, name + model::scale_suffix, safetensors::Dtype::Bf16, extents(model::scale_shape(tensor.shape)))
	    .transform([&](Located const& scales) {
		    return Stored{.data = *data, .scales = scales};
	    });
}

using Table = std::array<std::vector<std::optional<Stored>>, schedule::tensor_count>;

/** Every tensor the inventory names, each copy, out of the catalog; whatever is left is not the model's. */
[[nodiscard]] auto resolve(Catalog& catalog) -> core::Result<Table, CheckpointError> {
	auto table = Table{};
	for (auto entry = Count{0}; entry < schedule::tensor_count; ++entry) {
		auto const& tensor = model::inventory[entry];
		auto& slots = table.at(entry);
		slots.resize(copies);
		for (auto copy = Count{0}; copy < copies; ++copy) {
			if (!exists(tensor, copy)) {
				continue;
			}
			auto stored = take_stored(catalog, tensor, copy);
			if (!stored) {
				return std::unexpected(stored.error());
			}
			slots.at(copy) = *stored;
		}
	}
	if (!catalog.empty()) {
		return std::unexpected(CheckpointError{.failure = CheckpointFailure::Unexpected, .subject = catalog.begin()->first});
	}
	return table;
}

export class Checkpoint {
public:
	/** Maps and checks the checkpoint in `directory`; `pool` scans the e4m3 codes. */
	[[nodiscard]] static auto open(std::filesystem::path const& directory, thread_pool::Pool& pool) -> core::Result<Checkpoint, CheckpointError> {
		auto opened = safetensors_paths(directory).and_then([](std::vector<std::filesystem::path> const& paths) {
			return catalog_files(paths);
		});
		if (!opened) {
			return std::unexpected(opened.error());
		}
		auto table = resolve(opened->catalog);
		if (!table) {
			return std::unexpected(table.error());
		}
		auto result = Checkpoint{std::move(opened->files), std::move(*table)};
		if (auto nan = result.first_nan(pool)) {
			return std::unexpected(CheckpointError{.failure = CheckpointFailure::NanCode, .subject = std::move(*nan)});
		}
		return result;
	}

	/** The elements of inventory entry `tensor` at `at`: e4m3 codes or little-endian bf16. */
	[[nodiscard]] auto data(Count tensor, schedule::Instance at) const -> std::span<std::byte const> {
		return bytes(stored(tensor, at).data);
	}

	/** An e4m3 tensor's block scales, little-endian bf16, row-major over the 128 x 128 blocks. */
	[[nodiscard]] auto scales(Count tensor, schedule::Instance at) const -> std::span<std::byte const> {
		auto const& held = stored(tensor, at);
		core::invariant(held.scales.has_value());
		return bytes(held.scales.value());
	}

private:
	Checkpoint(std::vector<file::Mapping> files, Table tensors) : files_{std::move(files)}, tensors_{std::move(tensors)} {}

	[[nodiscard]] auto stored(Count tensor, schedule::Instance at) const -> Stored const& {
		auto const& held = tensors_.at(tensor).at(model::inventory[tensor].scope == model::Scope::Model ||
		                                                  model::inventory[tensor].scope == model::Scope::Mtp
		                                              ? 0
		                                              : copy_index(at));
		core::invariant(held.has_value());
		return held.value();
	}

	[[nodiscard]] auto bytes(Located const& where) const -> std::span<std::byte const> {
		return files_.at(where.file).bytes().subspan(where.bytes.begin, where.bytes.end - where.bytes.begin);
	}

	/** The first e4m3 nan code, named by tensor and element, scanning every weight in parallel. */
	[[nodiscard]] auto first_nan(thread_pool::Pool& pool) const -> std::optional<std::string> {
		struct Piece {
			Count tensor;
			Count copy;
			Located data;
			std::size_t begin;
			std::size_t end;
		};
		constexpr auto piece_bytes = std::size_t{64} << 20U;
		auto pieces = std::vector<Piece>{};
		for (auto entry = Count{0}; entry < schedule::tensor_count; ++entry) {
			if (model::inventory[entry].format != model::Format::E4m3) {
				continue;
			}
			for (auto copy = Count{0}; copy < copies; ++copy) {
				auto const& held = tensors_.at(entry).at(copy);
				if (!held.has_value()) {
					continue;
				}
				auto const data = held.value().data;
				auto const size = data.bytes.end - data.bytes.begin;
				for (auto begin = std::size_t{0}; begin < size; begin += piece_bytes) {
					pieces.push_back({.tensor = entry, .copy = copy, .data = data, .begin = begin, .end = std::min(size, begin + piece_bytes)});
				}
			}
		}
		auto found = std::vector<std::size_t>(pieces.size(), piece_bytes);
		pool.share(pieces.size(), [&](std::size_t index) {
			auto const& piece = pieces[index];
			auto const codes = bytes(piece.data).subspan(piece.begin, piece.end - piece.begin);
			found[index] = static_cast<std::size_t>(std::ranges::find_if(codes, e4m3_nan) - codes.begin());
		});
		for (auto index = std::size_t{0}; index < pieces.size(); ++index) {
			auto const& piece = pieces[index];
			if (found[index] < piece.end - piece.begin) {
				return std::format("{} element {}", checkpoint_name(model::inventory[piece.tensor], piece.copy), piece.begin + found[index]);
			}
		}
		return std::nullopt;
	}

	std::vector<file::Mapping> files_;
	Table tensors_;
};

}
