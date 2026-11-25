// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.json_model;
import engine.testing;

namespace {

namespace json = engine::json;
namespace model = engine::json_model;

using model::Model;
using Reader = json::Reader<json::request_bounds>;

[[nodiscard]] auto make_reader() -> std::unique_ptr<Reader> {
	auto reader = Reader::make();
	engine::core::invariant(reader.has_value());
	return std::make_unique<Reader>(std::move(*reader));
}

/** Members are a set: the fixtures compare objects by name, never by the order the writer chose. */
[[nodiscard]] auto canonical(Model value) -> Model {
	if (std::holds_alternative<model::Elements>(value.value)) {
		for (auto& element : std::get<model::Elements>(value.value)) {
			element = canonical(std::move(element));
		}
	} else if (std::holds_alternative<model::Members>(value.value)) {
		auto& members = std::get<model::Members>(value.value);
		for (auto& member : members) {
			member.second = canonical(std::move(member.second));
		}
		std::ranges::sort(members, {}, &std::pair<std::string, Model>::first);
	}
	return value;
}

[[nodiscard]] auto object(model::Members members) -> Model {
	return Model{.value = std::move(members)};
}

[[nodiscard]] auto array(model::Elements elements) -> Model {
	return Model{.value = std::move(elements)};
}

[[nodiscard]] auto text(std::string_view value) -> Model {
	return Model{.value = std::string{value}};
}

[[nodiscard]] auto integer(std::uint64_t value) -> Model {
	return Model{.value = static_cast<std::int64_t>(value)};
}

[[nodiscard]] auto agrees(Reader& reader, std::string_view json_text, Model const& fixture) -> bool {
	auto const read = model::read_model(reader, json_text);
	if (!read || !model::same(canonical(*read), canonical(fixture))) {
		std::println("not as documented: {:?}", json_text);
		return false;
	}
	return true;
}

/** OpenAI's documented `usage` object. */
[[nodiscard]] auto usage_fixture(json::Usage const& usage) -> Model {
	return object({
	    {"prompt_tokens", integer(usage.prompt_tokens)},
	    {"completion_tokens", integer(usage.completion_tokens)},
	    {"total_tokens", integer(usage.prompt_tokens + usage.completion_tokens)},
	    {"prompt_tokens_details", object({{"cached_tokens", integer(usage.cached_tokens)}})},
	});
}

[[nodiscard]] auto reason(json::FinishReason finish) -> std::string_view {
	switch (finish) {
	case json::FinishReason::Stop:
		return "stop";
	case json::FinishReason::Length:
		return "length";
	case json::FinishReason::ToolCalls:
		return "tool_calls";
	}
	std::unreachable();
}

[[nodiscard]] auto random_usage(std::mt19937_64& rng) -> json::Usage {
	auto const prompt = rng() % 300000;
	return json::Usage{.prompt_tokens = prompt, .completion_tokens = rng() % 300000, .cached_tokens = prompt == 0 ? 0 : rng() % (prompt + 1)};
}

/** A JSON object of generated arguments, as the engine would hold it after parsing a model's tool call. */
[[nodiscard]] auto random_arguments(std::mt19937_64& rng, Reader& reader) -> json::JsonText {
	auto members = model::Members{};
	for (auto count = rng() % 5; count > 0; --count) {
		auto name = model::random_text(rng, 4) + std::to_string(count);
		members.emplace_back(std::move(name), model::generate(rng, model::random_shape(rng)));
	}
	auto const written = model::serialize(object(std::move(members)), model::Style{.spelling = model::Spelling::Raw, .spacing = model::Spacing::Compact}, rng);
	auto arguments = json::JsonText::from(reader, written);
	engine::core::invariant(arguments.has_value());
	return *std::move(arguments);
}

[[nodiscard]] auto random_finish(std::mt19937_64& rng) -> json::FinishReason {
	auto constexpr reasons = std::array{json::FinishReason::Stop, json::FinishReason::Length, json::FinishReason::ToolCalls};
	return reasons[rng() % reasons.size()];
}

/** A whole completion is the documented `chat.completion` object, with `reasoning_content` and cached tokens. */
[[nodiscard]] auto completions_as_documented(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	auto scratch = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto completion = json::ChatCompletion{
		    .id = "chatcmpl-" + std::to_string(rng()),
		    .created = static_cast<std::int64_t>(rng() % 4000000000),
		    .model = model::random_text(rng, 6),
		    .content = model::random_text(rng, 40),
		    .reasoning_content = model::random_text(rng, 40),
		    .tool_calls = {},
		    .finish_reason = random_finish(rng),
		    .usage = random_usage(rng),
		};
		for (auto calls = rng() % 3; calls > 0; --calls) {
			completion.tool_calls.push_back(json::ToolCallOutput{.id = "call_" + std::to_string(rng()), .name = model::random_text(rng, 5), .arguments = random_arguments(rng, *scratch)});
		}
		auto message = model::Members{
		    {"role", text("assistant")},
		    {"content", text(completion.content)},
		    {"reasoning_content", text(completion.reasoning_content)},
		};
		if (!completion.tool_calls.empty()) {
			auto calls = model::Elements{};
			for (auto const& call : completion.tool_calls) {
				calls.push_back(object({
				    {"id", text(call.id)},
				    {"type", text("function")},
				    {"function", object({{"name", text(call.name)}, {"arguments", text(call.arguments.text())}})},
				}));
			}
			message.emplace_back("tool_calls", array(std::move(calls)));
		}
		auto const fixture = object({
		    {"id", text(completion.id)},
		    {"object", text("chat.completion")},
		    {"created", Model{.value = completion.created}},
		    {"model", text(completion.model)},
		    {"choices", array({object({{"index", integer(0)}, {"message", object(std::move(message))}, {"finish_reason", text(reason(completion.finish_reason))}})})},
		    {"usage", usage_fixture(completion.usage)},
		});
		if (!agrees(*reader, json::encode(completion), fixture)) {
			return false;
		}
	}
	return true;
}

/** One chunk's JSON, as appended for one server-sent event. */
[[nodiscard]] auto chunk_text(json::ChunkHeader const& header, json::ChunkEvent const& event) -> std::string {
	auto buffer = json::Buffer{std::size_t{1} << 20U};
	auto const appended = json::append(buffer, header, event);
	engine::core::invariant(appended.has_value());
	return std::string{buffer.text()};
}

/**
 * The chunk, when it is one line: an event's data must hold no line break, or the http layer would have to split
 * it into several `data:` lines. JSON escaping guarantees it; this checks the writer keeps to it.
 */
[[nodiscard]] auto chunk_of(std::string chunk) -> std::optional<std::string> {
	if (chunk.find_first_of("\r\n") != std::string::npos) {
		return std::nullopt;
	}
	return chunk;
}

[[nodiscard]] auto chunk_fixture(json::ChunkHeader const& header, model::Elements choices) -> model::Members {
	return model::Members{
	    {"id", text(header.id)},
	    {"object", text("chat.completion.chunk")},
	    {"created", Model{.value = header.created}},
	    {"model", text(header.model)},
	    {"choices", array(std::move(choices))},
	};
}

[[nodiscard]] auto choice(model::Members delta, Model finish) -> Model {
	return object({{"index", integer(0)}, {"delta", object(std::move(delta))}, {"finish_reason", std::move(finish)}});
}

/** Every frame shape of a stream is the documented `chat.completion.chunk`, framed as one server-sent event. */
[[nodiscard]] auto frames_as_documented(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	auto const none = Model{.value = nullptr};
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const id = "chatcmpl-" + std::to_string(rng());
		auto const name = model::random_text(rng, 6);
		auto const header = json::ChunkHeader{.id = id, .created = static_cast<std::int64_t>(rng() % 4000000000), .model = name};
		auto const piece = model::random_text(rng, 12);
		auto const call_id = "call_" + std::to_string(rng());
		auto const function = model::random_text(rng, 5);
		auto const position = static_cast<std::uint32_t>(rng() % 4);
		auto const finish = random_finish(rng);
		auto const usage = random_usage(rng);
		auto usage_frame = chunk_fixture(header, {});
		usage_frame.emplace_back("usage", usage_fixture(usage));
		auto const frames = std::array<std::pair<json::ChunkEvent, model::Members>, 7>{{
		    {json::Opening{}, chunk_fixture(header, {choice({{"role", text("assistant")}}, none)})},
		    {json::ContentDelta{.text = piece}, chunk_fixture(header, {choice({{"content", text(piece)}}, none)})},
		    {json::ReasoningDelta{.text = piece}, chunk_fixture(header, {choice({{"reasoning_content", text(piece)}}, none)})},
		    {json::ToolCallStart{.index = position, .id = call_id, .name = function},
		     chunk_fixture(header, {choice({{"tool_calls", array({object({
		                                                         {"index", integer(position)},
		                                                         {"id", text(call_id)},
		                                                         {"type", text("function")},
		                                                         {"function", object({{"name", text(function)}, {"arguments", text("")}})},
		                                                     })})}},
		                                   none)})},
		    {json::ArgumentsDelta{.index = position, .fragment = piece},
		     chunk_fixture(header, {choice({{"tool_calls", array({object({{"index", integer(position)}, {"function", object({{"arguments", text(piece)}})}})})}}, none)})},
		    {json::Finish{.reason = finish}, chunk_fixture(header, {choice({}, text(reason(finish)))})},
		    {json::UsageReport{.usage = usage}, usage_frame},
		}};
		for (auto const& [event, fixture] : frames) {
			auto const chunk = chunk_of(chunk_text(header, event));
			if (!chunk || !agrees(*reader, *chunk, object(fixture))) {
				return false;
			}
		}
	}
	return true;
}

/** Cuts text into pieces at code point boundaries, as a decoder streams it. */
[[nodiscard]] auto pieces_of(std::string_view whole, std::mt19937_64& rng) -> std::vector<std::string_view> {
	auto pieces = std::vector<std::string_view>{};
	auto at = std::size_t{0};
	while (at < whole.size()) {
		auto end = std::min<std::size_t>(whole.size(), at + 1 + rng() % 8);
		while (end < whole.size() && (std::bit_cast<std::uint8_t>(whole[end]) & 0xC0U) == 0x80U) {
			++end;
		}
		pieces.push_back(whole.substr(at, end - at));
		at = end;
	}
	return pieces;
}

using Step = std::variant<std::string_view, std::size_t>;

/** Reads a string at a path of member names and array indexes. */
[[nodiscard]] auto string_at(Model const& value, std::span<Step const> path) -> std::optional<std::string> {
	if (path.empty()) {
		if (!std::holds_alternative<std::string>(value.value)) {
			return std::nullopt;
		}
		return std::get<std::string>(value.value);
	}
	auto const& step = path.front();
	if (std::holds_alternative<std::size_t>(step)) {
		if (!std::holds_alternative<model::Elements>(value.value) || std::get<model::Elements>(value.value).size() <= std::get<std::size_t>(step)) {
			return std::nullopt;
		}
		return string_at(std::get<model::Elements>(value.value)[std::get<std::size_t>(step)], path.subspan(1));
	}
	if (!std::holds_alternative<model::Members>(value.value)) {
		return std::nullopt;
	}
	auto const& members = std::get<model::Members>(value.value);
	auto const found = std::ranges::find(members, std::string{std::get<std::string_view>(step)}, &std::pair<std::string, Model>::first);
	if (found == members.end()) {
		return std::nullopt;
	}
	return string_at(found->second, path.subspan(1));
}

/**
 * A tool call streamed in argument fragments reassembles exactly: the fragments of each call, concatenated in the
 * order sent, are the arguments string of the same call in the whole completion.
 */
[[nodiscard]] auto streamed_arguments_reassemble(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	auto scratch = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto calls = std::vector<json::ToolCallOutput>{};
		for (auto made = 1 + rng() % 3; made > 0; --made) {
			calls.push_back(json::ToolCallOutput{.id = "call_" + std::to_string(rng()), .name = model::random_text(rng, 5), .arguments = random_arguments(rng, *scratch)});
		}
		auto const whole = json::ChatCompletion{
		    .id = "chatcmpl-x",
		    .created = 0,
		    .model = "m",
		    .content = "",
		    .reasoning_content = "",
		    .tool_calls = calls,
		    .finish_reason = json::FinishReason::ToolCalls,
		    .usage = {.prompt_tokens = 0, .completion_tokens = 0, .cached_tokens = 0},
		};
		auto const read_whole = model::read_model(*reader, json::encode(whole));
		auto const header = json::ChunkHeader{.id = "chatcmpl-x", .created = 0, .model = "m"};
		auto reassembled = std::vector<std::string>(calls.size());
		for (auto call = std::size_t{0}; call < calls.size(); ++call) {
			auto const position = static_cast<std::uint32_t>(call);
			auto const start = chunk_of(chunk_text(header, json::ToolCallStart{.index = position, .id = calls[call].id, .name = calls[call].name}));
			if (!start) {
				return false;
			}
			auto const started = model::read_model(*reader, *start);
			auto const head = std::array<Step, 4>{Step{"choices"}, Step{std::size_t{0}}, Step{"delta"}, Step{"tool_calls"}};
			auto const at = [&head](std::initializer_list<Step> tail) {
				auto path = std::vector<Step>{head.begin(), head.end()};
				path.emplace_back(std::size_t{0});
				path.insert(path.end(), tail);
				return path;
			};
			if (!started || string_at(*started, at({Step{"id"}})) != calls[call].id || string_at(*started, at({Step{"function"}, Step{"name"}})) != calls[call].name) {
				return false;
			}
			for (auto const piece : pieces_of(calls[call].arguments.text(), rng)) {
				auto const chunk = chunk_of(chunk_text(header, json::ArgumentsDelta{.index = position, .fragment = piece}));
				if (!chunk) {
					return false;
				}
				auto const read = model::read_model(*reader, *chunk);
				auto const fragment = read ? string_at(*read, at({Step{"function"}, Step{"arguments"}})) : std::nullopt;
				if (!fragment) {
					return false;
				}
				reassembled[call] += *fragment;
			}
			auto const whole_path = std::array<Step, 7>{Step{"choices"}, Step{std::size_t{0}}, Step{"message"}, Step{"tool_calls"}, Step{call}, Step{"function"}, Step{"arguments"}};
			if (!read_whole || string_at(*read_whole, whole_path) != reassembled[call]) {
				return false;
			}
		}
	}
	return true;
}

/** A chunk that does not fit its buffer leaves the buffer as it was. */
[[nodiscard]] auto chunks_append_whole_or_not_at_all(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const piece = model::random_text(rng, 30);
		auto const header = json::ChunkHeader{.id = "chatcmpl-y", .created = 1, .model = "m"};
		auto const event = json::ChunkEvent{json::ContentDelta{.text = piece}};
		auto const whole = chunk_text(header, event);
		auto buffer = json::Buffer{rng() % (2 * whole.size())};
		auto const appended = json::append(buffer, header, event);
		if (appended.has_value() != (whole.size() <= buffer.text().size() + buffer.available()) || buffer.text() != (appended ? whole : std::string{})) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "a whole completion is the documented chat.completion (seed 1)", .holds = completions_as_documented(1, 2000)},
	    engine::testing::Case{.name = "every stream frame is the documented chunk, on one line (seed 2)", .holds = frames_as_documented(2, 2000)},
	    engine::testing::Case{.name = "streamed argument fragments reassemble the whole call (seed 3)", .holds = streamed_arguments_reassemble(3, 1000)},
	    engine::testing::Case{.name = "a chunk is appended whole or not at all (seed 4)", .holds = chunks_append_whole_or_not_at_all(4, 5000)},
	};
	return engine::testing::report(cases);
}
