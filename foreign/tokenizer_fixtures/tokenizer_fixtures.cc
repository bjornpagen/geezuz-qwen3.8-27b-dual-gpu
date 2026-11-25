// SPDX-License-Identifier: AGPL-3.0-only
export module engine.tokenizer_fixtures;

import std;

namespace engine::tokenizer_fixtures {

// Generated once from Hugging Face tokenizers over the checkpoint's tokenizer.json; data/tokenizer/README.md
// records how and describes the layout.
constexpr unsigned char encode_cases[] = {
#embed "encode.fix"
};

constexpr unsigned char decode_cases[] = {
#embed "decode.fix"
};

/** Texts with the ids the reference encodes them to, in both added-token readings. Program-lifetime storage. */
export [[nodiscard]] auto encode() -> std::span<std::byte const> {
	return std::as_bytes(std::span{encode_cases});
}

/** Id sequences with the text the reference decodes them to. Program-lifetime storage. */
export [[nodiscard]] auto decode() -> std::span<std::byte const> {
	return std::as_bytes(std::span{decode_cases});
}

}
