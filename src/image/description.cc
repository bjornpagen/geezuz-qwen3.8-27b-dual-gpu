// SPDX-License-Identifier: AGPL-3.0-only
export module engine.image:description;

import std;
import engine.core;
import engine.model;
import engine.prims;
import engine.sha256;

namespace engine::image {

[[nodiscard]] constexpr auto decimal(std::uint64_t value) -> std::string {
	auto digits = std::string{};
	do {
		digits.push_back(static_cast<char>('0' + (value % 10)));
		value /= 10;
	} while (value != 0);
	std::ranges::reverse(digits);
	return digits;
}

/** Appends one `name value...` line; numbers are decimal, words and enum cases (by their underlying value) as given. */
struct Lines {
	std::string text;

	constexpr auto line(std::string_view name, std::initializer_list<std::uint64_t> values) -> void {
		text += name;
		for (auto const value : values) {
			text += ' ';
			text += decimal(value);
		}
		text += '\n';
	}

	constexpr auto word(std::string_view name, std::string_view value) -> void {
		text += name;
		text += ' ';
		text += value;
		text += '\n';
	}
};

template<class E>
[[nodiscard]] constexpr auto code(E value) -> std::uint64_t {
	return std::to_underlying(value);
}

[[nodiscard]] constexpr auto bits(float value) -> std::uint64_t {
	return std::bit_cast<std::uint32_t>(value);
}

/**
 * The model description as canonical text: every dimension, layer kind, numeric convention and name pattern of
 * `engine.model`, every inventory tensor with its shape, scope, format and split, the ranges each card holds of it, and
 * every field of the weight tile encoding the device decodes. The ranges come from the split functions, so a change to
 * how a tensor divides changes the text even when no constant does; the tile encoding is in it so an image baked under
 * another encoding is refused rather than decoded wrongly. The text exists to be hashed; a human can read it to see
 * what an image's description digest covers.
 */
[[nodiscard]] constexpr auto describe() -> std::string {
	namespace m = model;
	auto lines = Lines{};
	lines.word("description", "qwen3.8-27b-fp8 tensor-parallel");
	lines.line("cards", {m::cards});
	lines.line("hidden", {m::hidden});
	lines.line("vocab", {m::vocab});
	lines.line("ffn", {m::ffn});
	lines.line("max_positions", {m::max_positions});
	lines.line("rms_norm_eps", {bits(m::rms_norm_eps)});
	lines.line("norm_scale", {code(m::rms_norm_scale), code(m::deltanet_norm_scale)});
	lines.line("mlp_gate", {code(m::mlp_gate)});
	auto mixers = std::string{};
	for (auto const mixer : m::trunk_mixers) {
		mixers += decimal(code(mixer));
	}
	lines.word("trunk_mixers", mixers);
	lines.line("mtp", {m::mtp_layers, code(m::mtp_mixer)});
	lines.line("attention", {m::query_heads, m::kv_heads, m::head_dim, m::q_proj_rows_per_head, code(m::attention_output_gate)});
	lines.line("rotary", {m::partial_rotary.numerator, m::partial_rotary.denominator, m::rotary_dim, bits(m::rope_theta)});
	lines.line("mrope_sections", {m::mrope_sections[0], m::mrope_sections[1], m::mrope_sections[2]});
	lines.line("deltanet", {m::key_heads, m::value_heads, m::key_dim, m::value_dim, m::conv_kernel, code(m::deltanet_output_gate)});
	lines.line("draft_vocab", {m::draft_vocab});
	lines.line("scale_block", {m::scale_block});
	lines.word("scale_suffix", m::scale_suffix);
	lines.word("excluded_prefix", m::excluded_prefix);
	lines.word("mtp_prefix", m::mtp_prefix);
	lines.word("trunk_layer_prefix", m::trunk_layer_prefix);
	lines.word("mtp_layer_prefix", m::mtp_layer_prefix);
	for (auto const& tensor : m::inventory) {
		lines.word("tensor", tensor.name);
		lines.line("shape", {tensor.shape.rank, tensor.shape.extents[0], tensor.shape.extents[1], tensor.shape.extents[2]});
		lines.line("kind", {code(tensor.scope), code(tensor.format), code(tensor.split)});
		for (auto card = m::Count{0}; card < m::cards; ++card) {
			auto const part = m::shard(tensor, card);
			lines.line("shard", {card, part.axis, part.count});
			for (auto const& range : std::span{part.ranges}.first(part.count)) {
				lines.line("range", {range.begin, range.length});
			}
		}
	}
	auto const& tile = engine::device::prims::tile::encoding;
	auto const count = [](int value) { return core::narrow<std::uint64_t>(value); };
	lines.line("tile_encoding", {tile.version, count(tile.bytes), count(tile.unit_bytes), count(tile.e4m3_rows), count(tile.e4m3_columns),
	                             count(tile.bf16_rows), count(tile.bf16_columns), tile.placed_mask, tile.left_mask, tile.right_mask,
	                             tile.select_lut, tile.two_to_120, tile.negative_zero});
	return std::move(lines.text);
}

inline constexpr std::array description_chars = [] {
	auto chars = std::array<char, describe().size()>{};
	std::ranges::copy(describe(), chars.begin());
	return chars;
}();

/** The model description's canonical text, as the build computed it. */
export inline constexpr std::string_view description_text = {description_chars.data(), description_chars.size()};

/** The digest of the model description this engine is built from; an image baked from another is refused. */
export inline constexpr sha256::Digest description_digest = sha256::constant_digest<description_chars>;

}
