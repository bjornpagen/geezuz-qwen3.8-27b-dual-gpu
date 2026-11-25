// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.file;
import engine.sha256;
import engine.testing;
import engine_test_chat.fixtures;

namespace {

/** The lowercase hex SHA-256 of a checkpoint file, or nothing when it does not map. */
[[nodiscard]] auto file_sha256(std::filesystem::path const& path) -> std::optional<std::string> {
	auto const mapping = engine::file::Mapping::open(path);
	if (!mapping) {
		return std::nullopt;
	}
	auto hasher = engine::sha256::Hasher{};
	hasher.update(mapping->bytes());
	auto const digits = engine::sha256::hex(std::move(hasher).finish().bytes);
	return std::string{digits.begin(), digits.end()};
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	if (arguments.size() != 2) {
		std::println("FAIL: usage: <checkpoint directory>");
		return 1;
	}
	auto const directory = std::filesystem::path{arguments[1]};
	auto const chat_template = file_sha256(directory / "chat_template.jinja");
	auto const tokenizer_config = file_sha256(directory / "tokenizer_config.json");
	auto const render = engine_test_chat::render_fixture();
	auto const parse = engine_test_chat::parse_fixture();
	auto const numbers = engine_test_chat::number_fixture();
	if (!chat_template || !tokenizer_config || !render || !parse || !numbers) {
		std::println("FAIL: the checkpoint's chat_template.jinja and tokenizer_config.json map, and the fixtures read");
		return 1;
	}
	auto const made_from = [&](auto const& fixture) {
		return fixture->template_sha256 == *chat_template && fixture->tokenizer_config_sha256 == *tokenizer_config;
	};
	auto const cases = std::array{
	    engine::testing::Case{.name = "the render fixtures were made from this checkpoint's chat template and tokenizer config",
		                      .holds = made_from(render)},
	    engine::testing::Case{.name = "the parse fixtures were made from this checkpoint's chat template and tokenizer config",
		                      .holds = made_from(parse)},
	    engine::testing::Case{.name = "the number fixtures were made from this checkpoint's chat template and tokenizer config",
		                      .holds = made_from(numbers)},
	};
	return engine::testing::report(cases);
}
