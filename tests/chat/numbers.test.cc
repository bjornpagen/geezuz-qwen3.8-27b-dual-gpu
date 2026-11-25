// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.tokenizer;
import engine.chat;
import engine.testing;
import engine_test_chat.fixtures;

namespace {

namespace chat = engine::chat;
namespace json = engine::json;
namespace tok = engine::tokenizer;
using engine_test_chat::NumberCase;

/**
 * Renders every in-range literal as an argument of one past tool call and reads back what the prompt holds for each:
 * the text the template's `tojson` wrote. Nothing when a literal does not read or the chat does not render.
 */
[[nodiscard]] auto written_numbers(std::span<NumberCase const> cases) -> std::optional<std::vector<std::string>> {
	auto reader = json::Reader<json::fragment_bounds>::make();
	auto renderer = chat::Renderer::make();
	auto const tokenizer = tok::load();
	if (!reader || !renderer || !tokenizer) {
		return std::nullopt;
	}
	auto call = json::ToolCall{.id = "c", .name = "f", .arguments = {}};
	for (auto const [index, number] : std::views::enumerate(cases)) {
		auto text = json::JsonText::from(*reader, number.literal);
		if (!text) {
			std::println("  {} does not read", number.literal);
			return std::nullopt;
		}
		call.arguments.push_back(json::Argument{.name = std::format("x{}", index), .value = std::move(*text)});
	}
	auto request = json::ChatRequest{};
	request.messages.emplace_back(json::UserMessage{.content = "q"});
	request.messages.emplace_back(json::AssistantMessage{.content = {}, .reasoning_content = {}, .tool_calls = {std::move(call)}});
	auto const prompt = renderer->render(*tokenizer, request);
	if (!prompt) {
		return std::nullopt;
	}
	auto const text = tokenizer->decode(prompt->ids);
	if (!text) {
		return std::nullopt;
	}
	auto out = std::vector<std::string>{};
	for (auto index = std::size_t{0}; index < cases.size(); ++index) {
		auto const open = std::format("<parameter=x{}>\n", index);
		auto const begin = text->find(open);
		auto const end = begin == std::string::npos ? begin : text->find("\n</parameter>", begin);
		if (end == std::string::npos) {
			return std::nullopt;
		}
		out.push_back(text->substr(begin + open.size(), end - begin - open.size()));
	}
	return out;
}

[[nodiscard]] auto out_of_range_refused(std::span<NumberCase const> cases) -> bool {
	auto reader = json::Reader<json::fragment_bounds>::make();
	return reader && std::ranges::all_of(cases, [&](NumberCase const& number) {
		       return !json::JsonText::from(*reader, number.literal);
	       });
}

}

auto main() -> int {
	auto const fixture = engine_test_chat::number_fixture();
	if (!fixture || fixture->cases.empty()) {
		std::println("FAIL: the number fixtures read");
		return 1;
	}
	auto in_range = std::vector<NumberCase>{};
	auto out_of_range = std::vector<NumberCase>{};
	for (auto const& number : fixture->cases) {
		(number.dumps ? in_range : out_of_range).push_back(number);
	}
	auto const written = written_numbers(in_range);
	auto const matches = written && std::ranges::equal(*written, in_range, {}, {}, [](NumberCase const& number) {
		                     return number.dumps.value_or("");
	                     });
	if (written && !matches) {
		for (auto const [index, number] : std::views::enumerate(in_range)) {
			auto const& wrote = written->at(static_cast<std::size_t>(index));
			if (wrote != number.dumps.value_or("")) {
				std::println("  first difference: {} wrote {}, Python writes {}", number.literal, wrote, number.dumps.value_or(""));
				break;
			}
		}
	}
	auto const checks = std::array{
	    engine::testing::Case{.name = "every JSON number literal is written as the template's tojson writes it (Python's json.dumps)",
		                      .holds = matches},
	    engine::testing::Case{.name = "literals past the double range, which Python writes as Infinity, never reach the template",
		                      .holds = !out_of_range.empty() && out_of_range_refused(out_of_range)},
	};
	return engine::testing::report(checks);
}
