// SPDX-License-Identifier: AGPL-3.0-only
export module engine.bake:source;

import std;
import engine.core;
import engine.model;
import engine.safetensors;
import engine.schedule;
import :error;
import :plan;
import :verify;

namespace engine::bake {

/** What the plan needs of one checkpoint tensor: its name, dtype and shape. */
struct Needed {
	std::string name;
	safetensors::Dtype dtype;
	std::vector<std::size_t> shape;
};

[[nodiscard]] constexpr auto stored_as(model::Format format) -> safetensors::Dtype {
	switch (format) {
	case model::Format::E4m3:
		return safetensors::Dtype::F8E4M3;
	case model::Format::Bf16:
		return safetensors::Dtype::Bf16;
	}
	std::unreachable();
}

[[nodiscard]] auto extents(model::Shape const& shape) -> std::vector<std::size_t> {
	auto const all = std::span{shape.extents}.first(shape.rank);
	return {all.begin(), all.end()};
}

/** The weight of `tensor`'s copy `at`, and for an e4m3 weight its block scales. */
auto need(std::vector<Needed>& needed, model::Tensor const& tensor, schedule::Instance const& at) -> void {
	auto const name = checkpoint_name(tensor, at);
	if (tensor.format == model::Format::E4m3) {
		needed.push_back({.name = scale_name(name), .dtype = safetensors::Dtype::Bf16, .shape = extents(model::scale_shape(tensor.shape))});
	}
	needed.push_back({.name = name, .dtype = stored_as(tensor.format), .shape = extents(tensor.shape)});
}

/** Every checkpoint tensor plan `P` reads, once: the tensors of every card placement, and the host tensors. */
template<Plan P>
[[nodiscard]] auto needed_by() -> std::vector<Needed> {
	auto needed = std::vector<Needed>{};
	for (auto const card : cards) {
		for (auto const& placement : P::placements(card)) {
			for (auto const tensor : std::span{placement.read.tensor}.first(placement.read.tensor_count)) {
				need(needed, std::span{P::inventory}[tensor], placement.instance);
			}
		}
	}
	for (auto const tensor : std::span<std::size_t const>{P::host_tensors}) {
		need(needed, std::span{P::inventory}[tensor], schedule::Instance{.stream = model::Stream::Head, .layer = 0});
	}
	std::ranges::sort(needed, std::less{}, &Needed::name);
	auto const repeated = std::ranges::unique(needed, std::equal_to{}, &Needed::name);
	needed.erase(repeated.begin(), repeated.end());
	return needed;
}

/** One stored tensor: the file holding it and its bytes' range in that file. */
struct Location {
	std::size_t file;
	safetensors::ByteRange bytes;
};

/**
 * The tensors of a verified checkpoint that a plan reads, each found once with the dtype and shape the model
 * description gives it. The checkpoint holds nothing else but the vision tower, which is dropped here. Its spans
 * borrow the verified files' mappings: valid while the source lives.
 */
export class Source {
public:
	template<Plan P>
	[[nodiscard]] static auto open(Verified verified) -> core::Result<Source, BakeError> {
		auto const needed = needed_by<P>();
		auto found = std::map<std::string, Location, std::less<>>{};
		for (auto file = std::size_t{0}; file < verified.files.size(); ++file) {
			auto const& each = verified.files[file];
			auto decoded = safetensors::decode(each.mapping.bytes());
			if (!decoded) {
				return refuse(NotSafetensors{.file = each.name, .error = decoded.error()});
			}
			for (auto& tensor : decoded->tensors) {
				auto const wanted = std::ranges::lower_bound(needed, tensor.name, std::less{}, &Needed::name);
				if (wanted == needed.end() || wanted->name != tensor.name) {
					if (!tensor.name.starts_with(model::excluded_prefix)) {
						return refuse(UnexpectedTensor{.tensor = tensor.name});
					}
					continue;
				}
				if (tensor.dtype != wanted->dtype || tensor.shape != wanted->shape) {
					return refuse(WrongTensor{.tensor = tensor.name});
				}
				if (!found.try_emplace(tensor.name, Location{.file = file, .bytes = tensor.bytes}).second) {
					return refuse(DuplicateTensor{.tensor = tensor.name});
				}
			}
		}
		for (auto const& each : needed) {
			if (!found.contains(each.name)) {
				return refuse(MissingTensor{.tensor = each.name});
			}
		}
		return Source{std::move(verified), std::move(found)};
	}

	/** The bytes of a tensor the plan reads; any other name is a bake bug. */
	[[nodiscard]] auto bytes(std::string_view name) const -> std::span<std::byte const> {
		auto const found = tensors_.find(name);
		core::invariant(found != tensors_.end());
		auto const& [file, range] = found->second;
		return verified_.files.at(file).mapping.bytes().subspan(range.begin, range.end - range.begin);
	}

	[[nodiscard]] auto verified() const -> Verified const& {
		return verified_;
	}

private:
	Source(Verified verified, std::map<std::string, Location, std::less<>> tensors) : verified_{std::move(verified)}, tensors_{std::move(tensors)} {}

	Verified verified_;
	std::map<std::string, Location, std::less<>> tensors_;
};

}
