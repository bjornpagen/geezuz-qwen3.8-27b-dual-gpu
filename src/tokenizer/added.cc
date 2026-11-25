// SPDX-License-Identifier: AGPL-3.0-only
export module engine.tokenizer:added;

import std;
import engine.model;

namespace engine::tokenizer {

/** A token id: a row of the model's embedding and lm_head. */
export using Id = std::uint32_t;

/** The byte-pair model's vocabulary: ids [0, bpe_vocabulary) are byte sequences the merges can produce. */
export inline constexpr Id bpe_vocabulary = 248044;

/** The byte-pair model's merge rules, by rank. */
export inline constexpr std::size_t merge_count = 247587;

/** The added tokens, by name; each is one id after the byte-pair vocabulary, in this order. */
export enum class Added : Id {
	endoftext = bpe_vocabulary,
	im_start,
	im_end,
	object_ref_start,
	object_ref_end,
	box_start,
	box_end,
	quad_start,
	quad_end,
	vision_start,
	vision_end,
	vision_pad,
	image_pad,
	video_pad,
	tool_call,
	tool_call_end,
	fim_prefix,
	fim_middle,
	fim_suffix,
	fim_pad,
	repo_name,
	file_sep,
	tool_response,
	tool_response_end,
	think,
	think_end,
	audio_start,
	audio_end,
	tts_pad,
	tts_text_bos,
	tts_text_eod,
	tts_text_bos_single,
	audio_pad,
};

/** Every id the tokenizer can produce or decode: the byte-pair vocabulary, then the added tokens. */
export inline constexpr Id vocabulary = std::to_underlying(Added::audio_pad) + 1;

/**
 * Hugging Face's `special` flag. Special tokens are control tokens a client never sees as text; ordinary added
 * tokens (`<think>`, `<tool_call>`, ...) are markup the model writes into its visible output.
 */
export enum class Role : std::uint8_t {
	Special,
	Ordinary,
};

export struct AddedToken {
	Added token;
	std::string_view spelling;
	Role role;
};

/** The added tokens in id order, as the checkpoint's tokenizer.json declares them. */
export inline constexpr auto added_tokens = std::array{
    AddedToken{.token = Added::endoftext, .spelling = "<|endoftext|>", .role = Role::Special},
    AddedToken{.token = Added::im_start, .spelling = "<|im_start|>", .role = Role::Special},
    AddedToken{.token = Added::im_end, .spelling = "<|im_end|>", .role = Role::Special},
    AddedToken{.token = Added::object_ref_start, .spelling = "<|object_ref_start|>", .role = Role::Special},
    AddedToken{.token = Added::object_ref_end, .spelling = "<|object_ref_end|>", .role = Role::Special},
    AddedToken{.token = Added::box_start, .spelling = "<|box_start|>", .role = Role::Special},
    AddedToken{.token = Added::box_end, .spelling = "<|box_end|>", .role = Role::Special},
    AddedToken{.token = Added::quad_start, .spelling = "<|quad_start|>", .role = Role::Special},
    AddedToken{.token = Added::quad_end, .spelling = "<|quad_end|>", .role = Role::Special},
    AddedToken{.token = Added::vision_start, .spelling = "<|vision_start|>", .role = Role::Special},
    AddedToken{.token = Added::vision_end, .spelling = "<|vision_end|>", .role = Role::Special},
    AddedToken{.token = Added::vision_pad, .spelling = "<|vision_pad|>", .role = Role::Special},
    AddedToken{.token = Added::image_pad, .spelling = "<|image_pad|>", .role = Role::Special},
    AddedToken{.token = Added::video_pad, .spelling = "<|video_pad|>", .role = Role::Special},
    AddedToken{.token = Added::tool_call, .spelling = "<tool_call>", .role = Role::Ordinary},
    AddedToken{.token = Added::tool_call_end, .spelling = "</tool_call>", .role = Role::Ordinary},
    AddedToken{.token = Added::fim_prefix, .spelling = "<|fim_prefix|>", .role = Role::Ordinary},
    AddedToken{.token = Added::fim_middle, .spelling = "<|fim_middle|>", .role = Role::Ordinary},
    AddedToken{.token = Added::fim_suffix, .spelling = "<|fim_suffix|>", .role = Role::Ordinary},
    AddedToken{.token = Added::fim_pad, .spelling = "<|fim_pad|>", .role = Role::Ordinary},
    AddedToken{.token = Added::repo_name, .spelling = "<|repo_name|>", .role = Role::Ordinary},
    AddedToken{.token = Added::file_sep, .spelling = "<|file_sep|>", .role = Role::Ordinary},
    AddedToken{.token = Added::tool_response, .spelling = "<tool_response>", .role = Role::Ordinary},
    AddedToken{.token = Added::tool_response_end, .spelling = "</tool_response>", .role = Role::Ordinary},
    AddedToken{.token = Added::think, .spelling = "<think>", .role = Role::Ordinary},
    AddedToken{.token = Added::think_end, .spelling = "</think>", .role = Role::Ordinary},
    AddedToken{.token = Added::audio_start, .spelling = "<|audio_start|>", .role = Role::Special},
    AddedToken{.token = Added::audio_end, .spelling = "<|audio_end|>", .role = Role::Special},
    AddedToken{.token = Added::tts_pad, .spelling = "<tts_pad>", .role = Role::Special},
    AddedToken{.token = Added::tts_text_bos, .spelling = "<tts_text_bos>", .role = Role::Special},
    AddedToken{.token = Added::tts_text_eod, .spelling = "<tts_text_eod>", .role = Role::Special},
    AddedToken{.token = Added::tts_text_bos_single, .spelling = "<tts_text_bos_single>", .role = Role::Special},
    AddedToken{.token = Added::audio_pad, .spelling = "<|audio_pad|>", .role = Role::Special},
};

static_assert(added_tokens.size() == vocabulary - bpe_vocabulary);
// Every id the tokenizer produces is a row of the model's embedding and lm_head; the rows past it are padding.
static_assert(vocabulary <= model::vocab);
static_assert(std::ranges::all_of(std::views::iota(std::size_t{0}, added_tokens.size()), [](std::size_t index) {
	return std::to_underlying(added_tokens.at(index).token) == bpe_vocabulary + index;
}));

// Decoding passes an added token's spelling through the byte-level map, which sends printable ascii to itself, and
// matching scans only at '<'; both hold because every spelling is printable ascii starting with '<'.
static_assert(std::ranges::all_of(added_tokens, [](AddedToken const& added) {
	return added.spelling.starts_with('<') && std::ranges::all_of(added.spelling, [](char const letter) {
		       return letter >= '!' && letter <= '~';
	       });
}));

export [[nodiscard]] constexpr auto id(Added token) -> Id {
	return std::to_underlying(token);
}

export [[nodiscard]] constexpr auto entry(Added token) -> AddedToken const& {
	return added_tokens.at(id(token) - bpe_vocabulary);
}

/** The added token an id names, if it names one. */
export [[nodiscard]] constexpr auto added(Id token) -> std::optional<Added> {
	if (token < bpe_vocabulary || token >= vocabulary) {
		return std::nullopt;
	}
	return added_tokens.at(token - bpe_vocabulary).token;
}

// Hugging Face matches added tokens leftmost-longest. No spelling is a prefix of another, so at most one spelling
// starts at any position and the leftmost match is the whole rule.
static_assert(std::ranges::none_of(added_tokens, [](AddedToken const& shorter) {
	return std::ranges::any_of(added_tokens, [&](AddedToken const& longer) {
		return longer.token != shorter.token && longer.spelling.starts_with(shorter.spelling);
	});
}));

/** An added token's spelling found in text: bytes [begin, begin + spelling length). */
struct Spelled {
	std::size_t begin;
	Added token;
};

/** The leftmost spelling at or after `from`. */
[[nodiscard]] constexpr auto find_spelled(std::string_view text, std::size_t from) -> std::optional<Spelled> {
	for (auto at = text.find('<', from); at != std::string_view::npos; at = text.find('<', at + 1)) {
		auto const rest = text.substr(at);
		auto const found = std::ranges::find_if(added_tokens, [&](AddedToken const& candidate) {
			return rest.starts_with(candidate.spelling);
		});
		if (found != added_tokens.end()) {
			return Spelled{.begin = at, .token = found->token};
		}
	}
	return std::nullopt;
}

}
