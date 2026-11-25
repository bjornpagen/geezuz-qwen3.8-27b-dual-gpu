// SPDX-License-Identifier: AGPL-3.0-only
//
// The model's own plan: the checkpoint's safetensors files with their published digests, the model description's
// tensors, the schedule's placements for each card, the host image's tensors, and the vocabulary permutation. All of
// it is compile-time data.
export module engine.bake:qwen;

import std;
import engine.bake_data;
import engine.core;
import engine.image;
import engine.model;
import engine.schedule;
import engine.sha256;
import engine.vocabulary;
import :plan;

namespace engine::bake {

inline constexpr std::string_view digest_list = bake_data::checkpoint_digests;

[[nodiscard]] consteval auto line_count(std::string_view text) -> std::size_t {
	return static_cast<std::size_t>(std::ranges::count(text, '\n'));
}

inline constexpr std::size_t hex_digits = 2 * sha256::digest_bytes;
inline constexpr std::string_view separator = "  ";

/** The pinned list as the build reads it: `<64 hex digits><two spaces><file>` lines; a malformed one does not compile. */
template<std::size_t lines>
[[nodiscard]] consteval auto parse_pinned(std::string_view text) -> std::array<Pinned, lines> {
	auto pinned = std::array<Pinned, lines>{};
	for (auto& entry : pinned) {
		auto const end = text.find('\n');
		auto const line = text.substr(0, end);
		core::invariant(line.size() > hex_digits + separator.size() && line.substr(hex_digits, separator.size()) == separator);
		entry = Pinned{.name = line.substr(hex_digits + separator.size()), .digest = sha256::pinned(line.substr(0, hex_digits))};
		text.remove_prefix(end + 1);
	}
	core::invariant(text.empty());
	return pinned;
}

inline constexpr auto pinned_files = parse_pinned<line_count(digest_list)>(digest_list);

[[nodiscard]] constexpr auto is_safetensors(Pinned const& file) -> bool {
	return file.name.ends_with(".safetensors");
}

inline constexpr std::size_t safetensors_count = static_cast<std::size_t>(std::ranges::count_if(pinned_files, is_safetensors));

/** The files the bake reads: every safetensors file of the checkpoint. Its other files hold no tensor. */
inline constexpr std::array<Pinned, safetensors_count> safetensors_files = [] {
	auto files = std::array<Pinned, safetensors_count>{};
	std::ranges::copy_if(pinned_files, files.begin(), is_safetensors);
	return files;
}();

inline constexpr std::array<schedule::CardImage, model::cards> card_images = {schedule::image(0), schedule::image(1)};

[[nodiscard]] constexpr auto on_host(model::Tensor const& tensor) -> bool {
	return tensor.split == model::Split::Host;
}

inline constexpr std::array host_inventory = [] {
	auto indices = std::array<std::size_t, static_cast<std::size_t>(std::ranges::count_if(model::inventory, on_host))>{};
	auto count = std::size_t{0};
	for (auto index = std::size_t{0}; index < std::size(model::inventory); ++index) {
		if (on_host(std::span{model::inventory}[index])) {
			indices.at(count) = index;
			++count;
		}
	}
	return indices;
}();

/** Qwen3.8-27B-FP8 at the pinned commit, split for the two cards by the decode schedule. */
export struct Qwen {
	static constexpr auto const& files = safetensors_files;
	static constexpr image::Commit commit = image::checkpoint_commit;
	static constexpr auto const& inventory = model::inventory;

	using CardZero = image::CardCatalog<model::Card::Zero>;
	using CardOne = image::CardCatalog<model::Card::One>;
	using Host = image::HostCatalog;

	static constexpr auto const& host_tensors = host_inventory;

	[[nodiscard]] static constexpr auto placements(model::Card card) -> std::span<schedule::Placement const> {
		return std::span{card_images.at(std::to_underlying(card)).placements};
	}

	[[nodiscard]] static constexpr auto blocks(schedule::Select select, std::size_t tensor, model::Card card) -> schedule::Blocks {
		return schedule::blocks(select, tensor, std::to_underlying(card));
	}

	[[nodiscard]] static constexpr auto token(model::Card card, std::size_t row) -> std::size_t {
		return vocabulary::token(card, row);
	}

	static constexpr std::size_t v_proj = schedule::tensors.v;
	static constexpr std::size_t input_norm = schedule::tensors.input_norm;
};

static_assert(Consistent<Qwen> && Published<Qwen>);

inline constexpr auto file_digests = [] {
	auto digests = std::array<image::FileDigest, safetensors_count>{};
	std::ranges::transform(safetensors_files, digests.begin(), [](Pinned const& file) {
		return image::FileDigest{.name = file.name, .digest = file.digest};
	});
	return digests;
}();

inline constexpr std::array manifest_chars = [] {
	auto chars = std::array<char, image::manifest(file_digests)->size()>{};
	std::ranges::copy(*image::manifest(file_digests), chars.begin());
	return chars;
}();

/**
 * The identity every image baked from the pinned checkpoint carries: the commit and the digest of the manifest of the
 * files the bake reads, each with its published SHA-256. The engine opens images against it without hashing the
 * checkpoint, because the bake refuses any file whose bytes are not the published ones.
 */
export inline constexpr image::Checkpoint checkpoint = {.commit = image::checkpoint_commit, .manifest = sha256::constant_digest<manifest_chars>};

}
