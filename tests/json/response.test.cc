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

/** Members are a set: the oracle compares objects by name, never by the order the writer chose. */
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

[[nodiscard]] auto signed_integer(std::int64_t value) -> Model {
	return Model{.value = value};
}

[[nodiscard]] auto real(double value) -> Model {
	return Model{.value = value};
}

[[nodiscard]] auto boolean(bool value) -> Model {
	return Model{.value = value};
}

[[nodiscard]] auto null() -> Model {
	return Model{.value = nullptr};
}

/** The JSON of a value written as text, read back strictly: a schema or a parameters object in a fixture. */
[[nodiscard]] auto parsed(Reader& reader, std::string_view written) -> Model {
	auto read = model::read_model(reader, written);
	engine::core::invariant(read.has_value());
	return *std::move(read);
}

[[nodiscard]] auto agrees(Reader& reader, std::string_view written, Model const& fixture) -> bool {
	auto const read = model::read_model(reader, written);
	if (!read || !model::same(canonical(*read), canonical(fixture))) {
		std::println("not as documented: {:?}", written);
		return false;
	}
	return true;
}

[[nodiscard]] auto coin(std::mt19937_64& rng) -> bool {
	return (rng() % 2) == 0;
}

[[nodiscard]] auto random_object(std::mt19937_64& rng, Reader& reader) -> json::JsonText {
	auto members = model::Members{};
	for (auto count = rng() % 4; count > 0; --count) {
		members.emplace_back(model::random_text(rng, 4) + std::to_string(count), model::generate(rng, model::random_shape(rng)));
	}
	auto const written = model::serialize(object(std::move(members)), model::Style{.spelling = model::Spelling::Raw, .spacing = model::Spacing::Compact}, rng);
	auto held = json::JsonText::from(reader, written);
	engine::core::invariant(held.has_value());
	return *std::move(held);
}

[[nodiscard]] auto random_status(std::mt19937_64& rng) -> json::ItemStatus {
	auto constexpr statuses = std::array{json::ItemStatus::InProgress, json::ItemStatus::Completed, json::ItemStatus::Incomplete};
	return statuses[rng() % statuses.size()];
}

[[nodiscard]] auto status_text(json::ItemStatus status) -> std::string_view {
	switch (status) {
	case json::ItemStatus::InProgress:
		return "in_progress";
	case json::ItemStatus::Completed:
		return "completed";
	case json::ItemStatus::Incomplete:
		return "incomplete";
	}
	std::unreachable();
}

[[nodiscard]] auto random_item(std::mt19937_64& rng) -> json::OutputItem {
	switch (rng() % 4) {
	case 0:
		return json::ReasoningOutput{.id = "rs_" + std::to_string(rng()), .text = model::random_text(rng, 30), .status = random_status(rng)};
	case 1:
		return json::MessageOutput{.id = "msg_" + std::to_string(rng()), .text = model::random_text(rng, 30), .status = random_status(rng)};
	case 2:
		return json::FunctionCallOutput{
		    .id = "fc_" + std::to_string(rng()),
		    .call_id = "call_" + std::to_string(rng()),
		    .name = model::random_text(rng, 8),
		    .arguments = model::random_text(rng, 30),
		    .status = random_status(rng),
		};
	default:
		return json::CustomCallOutput{
		    .id = "ctc_" + std::to_string(rng()),
		    .call_id = "call_" + std::to_string(rng()),
		    .name = model::random_text(rng, 8),
		    .input = model::random_text(rng, 30),
		};
	}
}

/** How much of an item the oracle expects: an `added` event shows it before any content, in progress. */
enum class Shown : std::uint8_t {
	Opening,
	Whole,
};

/** The sdk's documented output item, written from the sdk's types independently of the writer. */
[[nodiscard]] auto item_fixture(json::OutputItem const& item, Shown shown) -> Model {
	auto const whole = shown == Shown::Whole;
	auto const status = [whole](json::ItemStatus value) { return text(whole ? status_text(value) : "in_progress"); };
	if (auto const reasoning = engine::core::alternative<json::ReasoningOutput>(item)) {
		auto content = model::Elements{};
		if (whole) {
			content.push_back(object({{"type", text("reasoning_text")}, {"text", text(reasoning->text)}}));
		}
		return object({{"id", text(reasoning->id)}, {"type", text("reasoning")}, {"summary", array({})}, {"content", array(std::move(content))}, {"status", status(reasoning->status)}});
	}
	if (auto const message = engine::core::alternative<json::MessageOutput>(item)) {
		auto content = model::Elements{};
		if (whole) {
			content.push_back(object({{"type", text("output_text")}, {"text", text(message->text)}, {"annotations", array({})}, {"logprobs", array({})}}));
		}
		return object({{"id", text(message->id)}, {"type", text("message")}, {"role", text("assistant")}, {"status", status(message->status)}, {"content", array(std::move(content))}});
	}
	if (auto const call = engine::core::alternative<json::FunctionCallOutput>(item)) {
		return object({
		    {"id", text(call->id)},
		    {"type", text("function_call")},
		    {"call_id", text(call->call_id)},
		    {"name", text(call->name)},
		    {"arguments", text(whole ? std::string_view{call->arguments} : std::string_view{})},
		    {"status", status(call->status)},
		});
	}
	auto const& custom = std::get<json::CustomCallOutput>(item);
	return object({
	    {"id", text(custom.id)},
	    {"type", text("custom_tool_call")},
	    {"call_id", text(custom.call_id)},
	    {"name", text(custom.name)},
	    {"input", text(whole ? std::string_view{custom.input} : std::string_view{})},
	});
}

[[nodiscard]] auto random_tool(std::mt19937_64& rng, Reader& scratch) -> json::Tool {
	auto description = coin(rng) ? std::optional{model::random_text(rng, 10)} : std::nullopt;
	if (coin(rng)) {
		return json::Tool{
		    .name = "fn_" + std::to_string(rng() % 1000),
		    .description = std::move(description),
		    .input = json::FunctionTool{.parameters = coin(rng) ? std::optional{random_object(rng, scratch)} : std::nullopt},
		};
	}
	auto grammar = std::optional<json::Grammar>{};
	if (coin(rng)) {
		grammar = json::Grammar{.syntax = coin(rng) ? json::GrammarSyntax::Lark : json::GrammarSyntax::Regex, .definition = model::random_text(rng, 20)};
	}
	return json::Tool{.name = "custom_" + std::to_string(rng() % 1000), .description = std::move(description), .input = json::CustomTool{.grammar = std::move(grammar)}};
}

/** A tool as the sdk's `Response.tools` documents it; every function tool shows `strict: true`. */
[[nodiscard]] auto tool_fixture(json::Tool const& tool, Reader& reader) -> Model {
	if (auto const function = engine::core::alternative<json::FunctionTool>(tool.input)) {
		return object({
		    {"type", text("function")},
		    {"name", text(tool.name)},
		    {"description", tool.description ? text(*tool.description) : null()},
		    {"parameters", function->parameters ? parsed(reader, function->parameters->text()) : null()},
		    {"strict", boolean(true)},
		});
	}
	auto const& custom = std::get<json::CustomTool>(tool.input);
	auto members = model::Members{{"type", text("custom")}, {"name", text(tool.name)}};
	if (tool.description) {
		members.emplace_back("description", text(*tool.description));
	}
	if (custom.grammar) {
		members.emplace_back("format", object({{"type", text("grammar")},
		                                       {"syntax", text(custom.grammar->syntax == json::GrammarSyntax::Lark ? "lark" : "regex")},
		                                       {"definition", text(custom.grammar->definition)}}));
	} else {
		members.emplace_back("format", object({{"type", text("text")}}));
	}
	return object(std::move(members));
}

[[nodiscard]] auto random_choice(std::mt19937_64& rng) -> json::ToolChoice {
	switch (rng() % 5) {
	case 0:
		return json::ToolChoiceMode::None;
	case 1:
		return json::ToolChoiceMode::Auto;
	case 2:
		return json::ToolChoiceMode::Required;
	case 3:
		return json::ForcedFunction{.name = "fn_" + std::to_string(rng() % 1000)};
	default:
		return json::ForcedCustom{.name = "custom_" + std::to_string(rng() % 1000)};
	}
}

[[nodiscard]] auto choice_fixture(json::ToolChoice const& choice) -> Model {
	if (auto const mode = engine::core::alternative<json::ToolChoiceMode>(choice)) {
		auto constexpr modes = std::array<std::string_view, 3>{"none", "auto", "required"};
		return text(modes[std::to_underlying(*mode)]);
	}
	if (auto const function = engine::core::alternative<json::ForcedFunction>(choice)) {
		return object({{"type", text("function")}, {"name", text(function->name)}});
	}
	return object({{"type", text("custom")}, {"name", text(std::get<json::ForcedCustom>(choice).name)}});
}

[[nodiscard]] auto random_format(std::mt19937_64& rng, Reader& scratch) -> json::TextFormat {
	switch (rng() % 3) {
	case 0:
		return json::TextFormatMode::Text;
	case 1:
		return json::TextFormatMode::JsonObject;
	default:
		return json::JsonSchemaFormat{
		    .name = "schema_" + std::to_string(rng() % 100),
		    .description = coin(rng) ? std::optional{model::random_text(rng, 10)} : std::nullopt,
		    .schema = random_object(rng, scratch),
		};
	}
}

[[nodiscard]] auto format_fixture(json::TextFormat const& format, Reader& reader) -> Model {
	if (auto const mode = engine::core::alternative<json::TextFormatMode>(format)) {
		return object({{"type", text(*mode == json::TextFormatMode::Text ? "text" : "json_object")}});
	}
	auto const& schema = std::get<json::JsonSchemaFormat>(format);
	auto members = model::Members{{"type", text("json_schema")}, {"name", text(schema.name)}, {"schema", parsed(reader, schema.schema.text())}, {"strict", boolean(true)}};
	if (schema.description) {
		members.emplace_back("description", text(*schema.description));
	}
	return object(std::move(members));
}

[[nodiscard]] auto effort_text(json::Effort effort) -> std::string_view {
	switch (effort) {
	case json::Effort::None:
		return "none";
	case json::Effort::Low:
		return "low";
	case json::Effort::Medium:
		return "medium";
	case json::Effort::XHigh:
		return "xhigh";
	}
	std::unreachable();
}

[[nodiscard]] auto random_response(std::mt19937_64& rng, Reader& scratch) -> json::Response {
	auto constexpr statuses = std::array{json::ResponseStatus::InProgress, json::ResponseStatus::Completed, json::ResponseStatus::Incomplete};
	auto constexpr efforts = std::array{json::Effort::None, json::Effort::Low, json::Effort::Medium, json::Effort::XHigh};
	auto response = json::Response{
	    .id = "resp_" + std::to_string(rng()),
	    .created_at = static_cast<std::int64_t>(rng() % 4000000000),
	    .completed_at = coin(rng) ? std::optional{static_cast<std::int64_t>(rng() % 4000000000)} : std::nullopt,
	    .status = statuses[rng() % statuses.size()],
	    .instructions = coin(rng) ? std::optional{model::random_text(rng, 20)} : std::nullopt,
	    .metadata = {},
	    .model = model::random_text(rng, 8),
	    .output = {},
	    .tools = {},
	    .tool_choice = random_choice(rng),
	    .temperature = static_cast<double>(rng() % 2001) / 1000.0,
	    .top_p = static_cast<double>(rng() % 1001) / 1000.0,
	    .format = random_format(rng, scratch),
	    .effort = efforts[rng() % efforts.size()],
	    .max_output_tokens = coin(rng) ? std::optional{static_cast<std::int64_t>(rng() % 262145)} : std::nullopt,
	    .previous_response_id = coin(rng) ? std::optional{"resp_" + std::to_string(rng())} : std::nullopt,
	    .storage = coin(rng) ? json::Storage::Stored : json::Storage::Unstored,
	    .usage = std::nullopt,
	};
	for (auto count = rng() % 5; count > 0; --count) {
		response.metadata.push_back(json::MetadataPair{.key = std::format("k{}", count), .value = model::random_text(rng, 20)});
	}
	for (auto count = rng() % 6; count > 0; --count) {
		response.output.push_back(random_item(rng));
	}
	for (auto count = rng() % 4; count > 0; --count) {
		response.tools.push_back(random_tool(rng, scratch));
	}
	if (coin(rng)) {
		auto const input = rng() % 300000;
		auto const output = rng() % 300000;
		response.usage = json::Usage{
		    .input_tokens = input,
		    .cached_tokens = input == 0 ? 0 : rng() % (input + 1),
		    .output_tokens = output,
		    .reasoning_tokens = output == 0 ? 0 : rng() % (output + 1),
		};
	}
	return response;
}

/** The sdk's documented `Response` object, member for member. */
[[nodiscard]] auto response_fixture(json::Response const& response, Reader& reader) -> Model {
	auto const status = [&response] {
		switch (response.status) {
		case json::ResponseStatus::InProgress:
			return "in_progress";
		case json::ResponseStatus::Completed:
			return "completed";
		case json::ResponseStatus::Incomplete:
			return "incomplete";
		}
		std::unreachable();
	}();
	auto metadata = model::Members{};
	for (auto const& pair : response.metadata) {
		metadata.emplace_back(pair.key, text(pair.value));
	}
	auto output = model::Elements{};
	for (auto const& item : response.output) {
		output.push_back(item_fixture(item, Shown::Whole));
	}
	auto tools = model::Elements{};
	for (auto const& tool : response.tools) {
		tools.push_back(tool_fixture(tool, reader));
	}
	auto usage = null();
	if (response.usage) {
		auto const& counts = *response.usage;
		usage = object({
		    {"input_tokens", integer(counts.input_tokens)},
		    {"input_tokens_details", object({{"cached_tokens", integer(counts.cached_tokens)}, {"cache_write_tokens", integer(0)}})},
		    {"output_tokens", integer(counts.output_tokens)},
		    {"output_tokens_details", object({{"reasoning_tokens", integer(counts.reasoning_tokens)}})},
		    {"total_tokens", integer(counts.input_tokens + counts.output_tokens)},
		});
	}
	return object({
	    {"id", text(response.id)},
	    {"object", text("response")},
	    {"created_at", signed_integer(response.created_at)},
	    {"completed_at", response.completed_at ? signed_integer(*response.completed_at) : null()},
	    {"status", text(status)},
	    {"error", null()},
	    {"incomplete_details", response.status == json::ResponseStatus::Incomplete ? object({{"reason", text("max_output_tokens")}}) : null()},
	    {"instructions", response.instructions ? text(*response.instructions) : null()},
	    {"metadata", object(std::move(metadata))},
	    {"model", text(response.model)},
	    {"output", array(std::move(output))},
	    {"parallel_tool_calls", boolean(true)},
	    {"temperature", real(response.temperature)},
	    {"tool_choice", choice_fixture(response.tool_choice)},
	    {"tools", array(std::move(tools))},
	    {"top_p", real(response.top_p)},
	    {"background", boolean(false)},
	    {"max_output_tokens", response.max_output_tokens ? signed_integer(*response.max_output_tokens) : null()},
	    {"previous_response_id", response.previous_response_id ? text(*response.previous_response_id) : null()},
	    {"reasoning", object({{"effort", text(effort_text(response.effort))}, {"summary", null()}})},
	    {"service_tier", text("default")},
	    {"store", boolean(response.storage == json::Storage::Stored)},
	    {"text", object({{"format", format_fixture(response.format, reader)}, {"verbosity", text("medium")}})},
	    {"top_logprobs", integer(0)},
	    {"truncation", text("disabled")},
	    {"usage", std::move(usage)},
	});
}

/** A whole response, encoded or appended, is the documented `response` object. */
[[nodiscard]] auto responses_as_documented(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	auto scratch = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const response = random_response(rng, *scratch);
		auto const written = json::encode(response);
		auto buffer = json::Buffer{written.size()};
		auto const appended = json::append(buffer, response);
		if (!appended || buffer.text() != written || !agrees(*reader, written, response_fixture(response, *reader))) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto delta_type(json::Stream stream) -> std::string_view {
	switch (stream) {
	case json::Stream::OutputText:
		return "response.output_text.delta";
	case json::Stream::ReasoningText:
		return "response.reasoning_text.delta";
	case json::Stream::Arguments:
		return "response.function_call_arguments.delta";
	case json::Stream::CustomInput:
		return "response.custom_tool_call_input.delta";
	}
	std::unreachable();
}

[[nodiscard]] auto done_type(json::Stream stream) -> std::string_view {
	switch (stream) {
	case json::Stream::OutputText:
		return "response.output_text.done";
	case json::Stream::ReasoningText:
		return "response.reasoning_text.done";
	case json::Stream::Arguments:
		return "response.function_call_arguments.done";
	case json::Stream::CustomInput:
		return "response.custom_tool_call_input.done";
	}
	std::unreachable();
}

[[nodiscard]] auto done_member(json::Stream stream) -> std::string_view {
	switch (stream) {
	case json::Stream::OutputText:
	case json::Stream::ReasoningText:
		return "text";
	case json::Stream::Arguments:
		return "arguments";
	case json::Stream::CustomInput:
		return "input";
	}
	std::unreachable();
}

[[nodiscard]] auto part_fixture(json::PartKind kind, std::string_view written) -> Model {
	if (kind == json::PartKind::OutputText) {
		return object({{"type", text("output_text")}, {"text", text(written)}, {"annotations", array({})}, {"logprobs", array({})}});
	}
	return object({{"type", text("reasoning_text")}, {"text", text(written)}});
}

/** Text events of streams whose text lives in a content part carry its index; a call's do not; only output text has logprobs. */
[[nodiscard]] auto text_event_fixture(json::Stream stream, std::string_view type, std::string_view item_id, std::uint32_t output_index, std::string_view member,
                                      std::string_view written, std::uint64_t sequence) -> Model {
	auto members = model::Members{{"type", text(type)}, {"item_id", text(item_id)}, {"output_index", integer(output_index)}, {std::string{member}, text(written)},
	                              {"sequence_number", integer(sequence)}};
	if (stream == json::Stream::OutputText || stream == json::Stream::ReasoningText) {
		members.emplace_back("content_index", integer(0));
	}
	if (stream == json::Stream::OutputText) {
		members.emplace_back("logprobs", array({}));
	}
	return object(std::move(members));
}

/** Writes `event` with `append` and checks it against `fixture`, that `measure` agrees, and that a short buffer takes nothing. */
[[nodiscard]] auto event_agrees(Reader& reader, json::StreamEvent const& event, std::uint64_t sequence, Model const& fixture, std::mt19937_64& rng) -> bool {
	auto const size = json::measure(event, sequence);
	auto buffer = json::Buffer{size};
	if (!json::append(buffer, event, sequence) || buffer.text().size() != size || !agrees(reader, buffer.text(), fixture)) {
		return false;
	}
	auto short_buffer = json::Buffer{1 + static_cast<std::size_t>(rng() % size)};
	short_buffer.put("x");
	auto const refused = json::append(short_buffer, event, sequence);
	return !refused && short_buffer.text() == "x" && refused.error().needed == size;
}

/** Every stream event is the sdk's documented event object, numbered as asked, and measured exactly. */
[[nodiscard]] auto events_as_documented(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	auto scratch = make_reader();
	auto constexpr streams = std::array{json::Stream::OutputText, json::Stream::ReasoningText, json::Stream::Arguments, json::Stream::CustomInput};
	auto constexpr lifecycles = std::array{json::Lifecycle::Created, json::Lifecycle::InProgress, json::Lifecycle::Completed, json::Lifecycle::Incomplete};
	auto constexpr lifecycle_types = std::array<std::string_view, 4>{"response.created", "response.in_progress", "response.completed", "response.incomplete"};
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const sequence = rng() % 100000;
		auto const output_index = static_cast<std::uint32_t>(rng() % 100);
		auto const item_id = "item_" + std::to_string(rng());
		auto const piece = model::random_text(rng, 30);
		auto holds = true;
		switch (rng() % 8) {
		case 0: {
			auto const response = random_response(rng, *scratch);
			auto const which = rng() % lifecycles.size();
			holds = event_agrees(*reader, json::ResponseEvent{.lifecycle = lifecycles[which], .response = response}, sequence,
			                     object({{"type", text(lifecycle_types[which])}, {"response", response_fixture(response, *reader)}, {"sequence_number", integer(sequence)}}), rng);
			break;
		}
		case 1: {
			auto const item = random_item(rng);
			holds = event_agrees(*reader, json::ItemAdded{.output_index = output_index, .item = item}, sequence,
			                     object({{"type", text("response.output_item.added")}, {"output_index", integer(output_index)}, {"item", item_fixture(item, Shown::Opening)},
			                             {"sequence_number", integer(sequence)}}),
			                     rng);
			break;
		}
		case 2: {
			auto const item = random_item(rng);
			holds = event_agrees(*reader, json::ItemDone{.output_index = output_index, .item = item}, sequence,
			                     object({{"type", text("response.output_item.done")}, {"output_index", integer(output_index)}, {"item", item_fixture(item, Shown::Whole)},
			                             {"sequence_number", integer(sequence)}}),
			                     rng);
			break;
		}
		case 3:
		case 4: {
			auto const kind = coin(rng) ? json::PartKind::OutputText : json::PartKind::ReasoningText;
			auto const added = (rng() % 2) == 0;
			auto const event = added ? json::StreamEvent{json::PartAdded{.item_id = item_id, .output_index = output_index, .kind = kind}}
			                         : json::StreamEvent{json::PartDone{.item_id = item_id, .output_index = output_index, .kind = kind, .text = piece}};
			holds = event_agrees(*reader, event, sequence,
			                     object({{"type", text(added ? "response.content_part.added" : "response.content_part.done")},
			                             {"item_id", text(item_id)},
			                             {"output_index", integer(output_index)},
			                             {"content_index", integer(0)},
			                             {"part", part_fixture(kind, added ? std::string_view{} : std::string_view{piece})},
			                             {"sequence_number", integer(sequence)}}),
			                     rng);
			break;
		}
		case 5: {
			auto const stream = streams[rng() % streams.size()];
			holds = event_agrees(*reader, json::TextDelta{.stream = stream, .item_id = item_id, .output_index = output_index, .delta = piece}, sequence,
			                     text_event_fixture(stream, delta_type(stream), item_id, output_index, "delta", piece, sequence), rng);
			break;
		}
		case 6: {
			auto const stream = streams[rng() % streams.size()];
			holds = event_agrees(*reader, json::TextDone{.stream = stream, .item_id = item_id, .output_index = output_index, .text = piece}, sequence,
			                     text_event_fixture(stream, done_type(stream), item_id, output_index, done_member(stream), piece, sequence), rng);
			break;
		}
		default: {
			auto const code = model::random_text(rng, 10);
			auto const param = coin(rng) ? std::optional{model::random_text(rng, 10)} : std::nullopt;
			auto const event = json::ErrorEvent{.code = code, .message = piece, .param = param ? std::optional<std::string_view>{*param} : std::nullopt};
			holds = event_agrees(*reader, event, sequence,
			                     object({{"type", text("error")}, {"code", text(code)}, {"message", text(piece)}, {"param", param ? text(*param) : null()}, {"sequence_number", integer(sequence)}}),
			                     rng);
			break;
		}
		}
		if (!holds) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto random_input(std::mt19937_64& rng, Reader& scratch) -> json::InputItem {
	switch (rng() % 6) {
	case 0: {
		auto constexpr roles = std::array{json::Role::User, json::Role::Assistant, json::Role::System, json::Role::Developer};
		return json::MessageItem{.role = roles[rng() % roles.size()], .text = model::random_text(rng, 20)};
	}
	case 1:
		return json::FunctionCallItem{.call_id = "call_" + std::to_string(rng()), .name = model::random_text(rng, 8), .arguments = random_object(rng, scratch)};
	case 2:
		return json::FunctionOutputItem{.call_id = "call_" + std::to_string(rng()), .output = model::random_text(rng, 20)};
	case 3:
		return json::CustomCallItem{.call_id = "call_" + std::to_string(rng()), .name = model::random_text(rng, 8), .input = model::random_text(rng, 20)};
	case 4:
		return json::CustomOutputItem{.call_id = "call_" + std::to_string(rng()), .output = model::random_text(rng, 20)};
	default:
		return json::ReasoningItem{.text = model::random_text(rng, 20)};
	}
}

[[nodiscard]] auto role_text(json::Role role) -> std::string_view {
	switch (role) {
	case json::Role::User:
		return "user";
	case json::Role::Assistant:
		return "assistant";
	case json::Role::System:
		return "system";
	case json::Role::Developer:
		return "developer";
	}
	std::unreachable();
}

/** A stored input item as the sdk's `ResponseItem` documents it. */
[[nodiscard]] auto input_fixture(json::StoredInput const& stored) -> Model {
	auto const id = std::pair<std::string, Model>{"id", text(stored.id)};
	if (auto const message = engine::core::alternative<json::MessageItem>(stored.item)) {
		auto const assistant = message->role == json::Role::Assistant;
		auto part = assistant ? object({{"type", text("output_text")}, {"text", text(message->text)}, {"annotations", array({})}})
		                      : object({{"type", text("input_text")}, {"text", text(message->text)}});
		return object({id, {"type", text("message")}, {"role", text(role_text(message->role))}, {"status", text("completed")}, {"content", array({std::move(part)})}});
	}
	if (auto const call = engine::core::alternative<json::FunctionCallItem>(stored.item)) {
		return object({id, {"type", text("function_call")}, {"call_id", text(call->call_id)}, {"name", text(call->name)}, {"arguments", text(call->arguments.text())},
		               {"status", text("completed")}});
	}
	if (auto const output = engine::core::alternative<json::FunctionOutputItem>(stored.item)) {
		return object({id, {"type", text("function_call_output")}, {"call_id", text(output->call_id)}, {"output", text(output->output)}, {"status", text("completed")}});
	}
	if (auto const custom = engine::core::alternative<json::CustomCallItem>(stored.item)) {
		return object({id, {"type", text("custom_tool_call")}, {"call_id", text(custom->call_id)}, {"name", text(custom->name)}, {"input", text(custom->input)}});
	}
	if (auto const output = engine::core::alternative<json::CustomOutputItem>(stored.item)) {
		return object({id, {"type", text("custom_tool_call_output")}, {"call_id", text(output->call_id)}, {"output", text(output->output)}});
	}
	auto const& reasoning = std::get<json::ReasoningItem>(stored.item);
	return object({id, {"type", text("reasoning")}, {"summary", array({})}, {"content", array({object({{"type", text("reasoning_text")}, {"text", text(reasoning.text)}})})}});
}

/** A page of input items is the sdk's `ResponseItemList`; an empty page has no first or last id. */
[[nodiscard]] auto input_items_as_documented(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	auto scratch = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto page = std::vector<json::StoredInput>{};
		for (auto items = rng() % 6; items > 0; --items) {
			page.push_back(json::StoredInput{.id = "item_" + std::to_string(rng()), .item = random_input(rng, *scratch)});
		}
		auto const more = coin(rng);
		auto data = model::Elements{};
		for (auto const& stored : page) {
			data.push_back(input_fixture(stored));
		}
		auto const fixture = object({
		    {"object", text("list")},
		    {"data", array(std::move(data))},
		    {"first_id", page.empty() ? null() : text(page.front().id)},
		    {"last_id", page.empty() ? null() : text(page.back().id)},
		    {"has_more", boolean(more)},
		});
		if (!agrees(*reader, json::encode_input_items(page, more), fixture)) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto deletion_as_documented() -> bool {
	auto reader = make_reader();
	return agrees(*reader, json::encode_deleted("resp_0123"), object({{"id", text("resp_0123")}, {"object", text("response.deleted")}, {"deleted", boolean(true)}}));
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "a whole response is the documented response object (seed 1)", .holds = responses_as_documented(1, 2000)},
	    engine::testing::Case{.name = "every stream event is documented, measured exactly, and appended whole or not at all (seed 2)", .holds = events_as_documented(2, 4000)},
	    engine::testing::Case{.name = "a page of input items is the documented item list (seed 3)", .holds = input_items_as_documented(3, 2000)},
	    engine::testing::Case{.name = "a deletion is the documented deleted response", .holds = deletion_as_documented()},
	};
	return engine::testing::report(cases);
}
