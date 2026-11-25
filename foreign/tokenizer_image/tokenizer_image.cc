// SPDX-License-Identifier: AGPL-3.0-only
export module engine.tokenizer_image;

import std;

namespace engine::tokenizer_image {

// The tokenizer is part of the one model the engine runs, so its image is built in rather than found at run time.
constexpr unsigned char image[] = {
#embed "qwen.tok"
};

/** The tokenizer image, data/tokenizer/qwen.tok, as built into the binary. Program-lifetime storage. */
export [[nodiscard]] auto bytes() -> std::span<std::byte const> {
	return std::as_bytes(std::span{image});
}

}
