// SPDX-License-Identifier: AGPL-3.0-only
export module engine.chat_fixtures;

import std;

namespace engine::chat_fixtures {

// Generated once from Hugging Face transformers and tokenizers over the checkpoint's chat template, response template
// and tokenizer.json; data/chat/README.md records how and describes each file.
constexpr char render_cases[] = {
#embed "render.json"
};

constexpr char parse_cases[] = {
#embed "parse.json"
};

constexpr char number_cases[] = {
#embed "numbers.json"
};

/** Chat requests with the template's text and the ids the engine and the reference make of it. Program-lifetime storage. */
export [[nodiscard]] auto render() -> std::string_view {
	return {render_cases, sizeof render_cases};
}

/** Model replies with the reference's parse. Program-lifetime storage. */
export [[nodiscard]] auto parse() -> std::string_view {
	return {parse_cases, sizeof parse_cases};
}

/** JSON number literals with the text the template's `tojson` writes for them. Program-lifetime storage. */
export [[nodiscard]] auto numbers() -> std::string_view {
	return {number_cases, sizeof number_cases};
}

}
