// SPDX-License-Identifier: AGPL-3.0-only
export module engine.chat:reply;

import std;
import engine.core;
import engine.json;
import :text;
import :dumps;
import :cast;
import :conversation;

namespace engine::chat {

/** More of the reasoning. */
export struct ReasoningDelta {
	std::string text;

	[[nodiscard]] auto operator==(ReasoningDelta const&) const -> bool = default;
};

/** More of the content. */
export struct ContentDelta {
	std::string text;

	[[nodiscard]] auto operator==(ContentDelta const&) const -> bool = default;
};

/** A tool call opened: the reply's `index`-th, counting from zero, calling `name`. */
export struct ToolCallStart {
	std::size_t index;
	std::string name;

	[[nodiscard]] auto operator==(ToolCallStart const&) const -> bool = default;
};

/**
 * The tool call opened as `index` closed, with its whole `arguments`: a JSON object as Python's `json.dumps` writes
 * the reference's arguments. They come whole because the reference lets a later parameter of the same name replace
 * an earlier one in place, and drops a parameter whose end never comes, so no part of them is final before the call
 * ends.
 */
export struct ToolCallEnd {
	std::size_t index;
	std::string arguments;

	[[nodiscard]] auto operator==(ToolCallEnd const&) const -> bool = default;
};

/** What the reply parser reports: each piece of text is final when reported, and none is ever taken back. */
export using Event = std::variant<ReasoningDelta, ContentDelta, ToolCallStart, ToolCallEnd>;

/** A tool's name and the parameters whose arguments are cast. */
struct Signature {
	std::string name;
	std::vector<ParameterCasts> parameters;
};

/** The argument casts of a reply's tools, with the reader that reads object and array arguments. */
struct Casting {
	std::vector<Signature> signatures;
	json::Reader<json::fragment_bounds> reader;

	/** The casts of a tool's parameter: none for a parameter or tool without any. Of tools of one name, the reference keeps the last. */
	[[nodiscard]] auto casts(std::string_view tool, std::string_view parameter) const -> std::span<Cast const> {
		auto const named = std::ranges::find(signatures | std::views::reverse, tool, &Signature::name);
		if (named == std::ranges::end(signatures | std::views::reverse)) {
			return {};
		}
		auto const found = std::ranges::find(named->parameters, parameter, &ParameterCasts::name);
		return found == named->parameters.end() ? std::span<Cast const>{} : std::span<Cast const>{found->casts};
	}
};

/** Which field a run of text belongs to. */
enum class Field : std::uint8_t {
	Reasoning,
	Content,
};

/** Whether a field's current segment has written text yet; until it has, its leading whitespace is dropped. */
enum class Segment : std::uint8_t {
	Fresh,
	Started,
};

struct InText {
	Field field;
	Segment segment;
};

struct InCall {
	std::size_t index;
	std::string name;
	/** Everything between the opening `<function=...>` and the close. */
	std::string body;
};

using Region = std::variant<InText, InCall>;

/** Whether more output may follow. */
enum class Feed : std::uint8_t {
	Open,
	Ended,
};

/** How much of a marker the text at a position holds. */
enum class Prefix : std::uint8_t {
	Whole,
	Part,
	None,
};

[[nodiscard]] constexpr auto compare(std::string_view rest, std::string_view marker) -> Prefix {
	if (rest.starts_with(marker)) {
		return Prefix::Whole;
	}
	return marker.starts_with(rest) ? Prefix::Part : Prefix::None;
}

/** The boundaries the response template watches for. */
enum class Marker : std::uint8_t {
	/** `<think>` in content: a reasoning region opens. */
	ThinkOpen,
	/** `</think>`: the reasoning region closes. */
	ThinkClose,
	/** `<tool_call>`, whitespace, `<function=NAME>`: a tool call opens. */
	CallOpen,
	/** `</tool_call>`: the tool call closes. */
	CallClose,
	/** Before `<tool_call>`: the reasoning or tool call closes, and the next region starts at the tag. */
	CallNext,
	/** `<|im_end|>` or `<|endoftext|>` in content: the content segment closes. */
	MessageEnd,
};

struct Hit {
	std::size_t begin;
	std::size_t end;
	Marker marker;
	std::string_view name;
};

/** A marker may begin at `begin`; more input decides. */
struct Held {
	std::size_t begin;
};

/** No marker begins in the text. */
struct Clear {};

using Scan = std::variant<Hit, Held, Clear>;

/** Whether a marker's text is consumed, or only looked ahead at and left for the next region. */
enum class Reach : std::uint8_t {
	Consumed,
	Lookahead,
};

struct LiteralMarker {
	std::string_view text;
	Marker marker;
	Reach reach;
};

inline constexpr auto reasoning_markers = std::to_array<LiteralMarker>({
    {.text = "</think>", .marker = Marker::ThinkClose, .reach = Reach::Consumed},
    {.text = "<tool_call>", .marker = Marker::CallNext, .reach = Reach::Lookahead},
});

inline constexpr auto content_markers = std::to_array<LiteralMarker>({
    {.text = "<think>", .marker = Marker::ThinkOpen, .reach = Reach::Consumed},
    {.text = "<|im_end|>", .marker = Marker::MessageEnd, .reach = Reach::Consumed},
    {.text = "<|endoftext|>", .marker = Marker::MessageEnd, .reach = Reach::Consumed},
});

inline constexpr auto call_markers = std::to_array<LiteralMarker>({
    {.text = "</tool_call>", .marker = Marker::CallClose, .reach = Reach::Consumed},
    {.text = "<tool_call>", .marker = Marker::CallNext, .reach = Reach::Lookahead},
});

struct OpenMatch {
	Prefix prefix;
	std::size_t end;
	std::string_view name;
};

/** The tool-call opening at the front of `rest`: `<tool_call>\s*<function=(?P<name>[^>\n]+)>`. */
[[nodiscard]] auto match_open(std::string_view rest, Feed input) -> OpenMatch {
	auto const none = OpenMatch{.prefix = Prefix::None, .end = 0, .name = {}};
	auto const part = OpenMatch{.prefix = input == Feed::Open ? Prefix::Part : Prefix::None, .end = 0, .name = {}};
	auto constexpr tag = std::string_view{"<tool_call>"};
	auto constexpr function = std::string_view{"<function="};
	switch (compare(rest, tag)) {
	case Prefix::Whole:
		break;
	case Prefix::Part:
		return part;
	case Prefix::None:
		return none;
	}
	auto at = tag.size();
	while (at < rest.size()) {
		auto const c = front(rest.substr(at));
		if (!c) {
			// Output that ends inside a character ends with no whitespace there.
			if (input == Feed::Open) {
				return part;
			}
			break;
		}
		if (!regex_space(c->value)) {
			break;
		}
		at += c->length;
	}
	switch (compare(rest.substr(at), function)) {
	case Prefix::Whole:
		break;
	case Prefix::Part:
		return part;
	case Prefix::None:
		return none;
	}
	at += function.size();
	auto const stop = rest.find_first_of(">\n", at);
	if (stop == std::string_view::npos) {
		return part;
	}
	if (rest[stop] == '\n' || stop == at) {
		return none;
	}
	return OpenMatch{.prefix = Prefix::Whole, .end = stop + 1, .name = rest.substr(at, stop - at)};
}

/** One parameter as the model wrote it: its name and its text, both views into the call's body. */
struct Written {
	std::string_view name;
	std::string_view text;
};

/**
 * A call body's parameters as the response template reads them with
 * `<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?(?:</parameter>|(?=<parameter=)|(?=</function>))`: a value runs
 * to the first `</parameter>`, `<parameter=` or `</function>`, less one newline at each end; a parameter with no end
 * is dropped; text between parameters is ignored; a repeated name keeps its first place and takes the last value.
 */
[[nodiscard]] auto written_parameters(std::string_view body) -> std::vector<Written> {
	auto constexpr open = std::string_view{"<parameter="};
	auto constexpr close = std::string_view{"</parameter>"};
	auto out = std::vector<Written>{};
	auto at = body.find(open);
	while (at != std::string_view::npos) {
		auto const name_begin = at + open.size();
		auto const stop = body.find_first_of(">\n", name_begin);
		auto const value_begin = stop == std::string_view::npos ? stop : stop + 1 + (body.substr(stop + 1).starts_with('\n') ? 1 : 0);
		auto const end =
		    value_begin == std::string_view::npos
		        ? value_begin
		        : std::min({body.find(close, value_begin), body.find(open, value_begin), body.find("</function>", value_begin)});
		if (stop == std::string_view::npos || body[stop] != '>' || stop == name_begin || end == std::string_view::npos) {
			at = body.find(open, at + 1);
			continue;
		}
		auto const value_end = end > value_begin && body[end - 1] == '\n' ? end - 1 : end;
		auto const parameter =
		    Written{.name = body.substr(name_begin, stop - name_begin), .text = body.substr(value_begin, value_end - value_begin)};
		auto const same = std::ranges::find(out, parameter.name, &Written::name);
		if (same == out.end()) {
			out.push_back(parameter);
		} else {
			same->text = parameter.text;
		}
		at = body.find(open, body.substr(end).starts_with(close) ? end + close.size() : end);
	}
	return out;
}

/**
 * Parses a model's reply as the checkpoint's response template does (`tokenizer_config.json`), as it streams: text
 * pushed in any pieces, then `finish`, reports reasoning, content and tool calls. The template's regions: reasoning
 * until `</think>` or a `<tool_call>`; content (the implicit region) until a `<think>`, a tool call or a message end;
 * a tool call until `</tool_call>` or the next `<tool_call>`. Reasoning and content are each trimmed per region and
 * joined, and a tool call's arguments are cast by its tool's parameter schemas.
 *
 * Nothing reported is ever taken back: a piece of text is reported once no marker can still begin in it, and
 * trailing whitespace once text follows it. Where the reference would take text back, the parser differs: the
 * reference keeps only the last reasoning region, and a reasoning region after content would replace the reasoning
 * already streamed, so reasoning regions are joined as content regions are.
 */
export class Parser {
public:
	/**
	 * A parser for a reply: it starts where the generation prompt leaves the model, inside the thinking block or
	 * after it, and casts arguments by the function tools' parameter schemas. A custom tool's `input` is never cast.
	 */
	[[nodiscard]] static auto make(std::span<json::Tool const> tools, Thinking thinking) -> core::Result<Parser, json::ReadError>;

	/** Reads more of the reply. */
	[[nodiscard]] auto push(std::string_view text) -> std::vector<Event> {
		core::invariant(stream_ == Feed::Open);
		pending_.append(text);
		return drain(Feed::Open);
	}

	/** Ends the reply: what was held for more input is decided, and an open tool call closes. */
	[[nodiscard]] auto finish() -> std::vector<Event> {
		core::invariant(stream_ == Feed::Open);
		stream_ = Feed::Ended;
		auto events = drain(Feed::Ended);
		if (std::holds_alternative<InCall>(region_)) {
			close_call(events);
		}
		return events;
	}

private:
	Parser(Region region, std::optional<Casting> casting) : casting_{std::move(casting)}, region_{std::move(region)} {}

	[[nodiscard]] auto scan(Feed input) const -> Scan;
	[[nodiscard]] auto drain(Feed input) -> std::vector<Event>;
	/** Reports the region's text that is final; returns how many bytes of `text` it used. */
	[[nodiscard]] auto deliver(std::vector<Event>& events, std::string_view text, Feed input) -> std::size_t;
	auto enter(std::vector<Event>& events, Marker marker, std::string name) -> void;
	auto close_call(std::vector<Event>& events) -> void;
	[[nodiscard]] auto arguments(std::string_view name, std::string_view body) -> std::string;

	/** Present when the request has tools. */
	std::optional<Casting> casting_;
	Region region_;
	/** Output not yet reported: a possible marker, trailing whitespace, a partial character. */
	std::string pending_;
	std::size_t calls_ = 0;
	Feed stream_ = Feed::Open;
};

[[nodiscard]] constexpr auto opening_field(Thinking thinking) -> Field {
	switch (thinking) {
	case Thinking::Enabled:
		return Field::Reasoning;
	case Thinking::Disabled:
		return Field::Content;
	}
	std::unreachable();
}

auto Parser::make(std::span<json::Tool const> tools, Thinking thinking) -> core::Result<Parser, json::ReadError> {
	auto const region = InText{.field = opening_field(thinking), .segment = Segment::Fresh};
	if (tools.empty()) {
		return Parser{region, std::nullopt};
	}
	auto reader = json::Reader<json::fragment_bounds>::make();
	if (!reader) {
		return std::unexpected(reader.error());
	}
	auto signatures = std::vector<Signature>{};
	for (auto const& tool : tools) {
		auto signature = Signature{.name = tool.name, .parameters = {}};
		auto const function = core::alternative<json::FunctionTool>(tool.input);
		if (function && function->parameters) {
			auto casts = parameter_casts(*reader, function->parameters->text());
			if (!casts) {
				return std::unexpected(casts.error());
			}
			signature.parameters = *std::move(casts);
		}
		signatures.push_back(std::move(signature));
	}
	return Parser{region, Casting{.signatures = std::move(signatures), .reader = std::move(*reader)}};
}

auto Parser::scan(Feed input) const -> Scan {
	auto const markers = [&] -> std::span<LiteralMarker const> {
		if (std::holds_alternative<InCall>(region_)) {
			return call_markers;
		}
		return std::get<InText>(region_).field == Field::Reasoning ? std::span<LiteralMarker const>{reasoning_markers}
		                                                           : std::span<LiteralMarker const>{content_markers};
	}();
	auto const in_content = std::holds_alternative<InText>(region_) && std::get<InText>(region_).field == Field::Content;
	auto const view = std::string_view{pending_};
	for (auto at = view.find('<'); at != std::string_view::npos; at = view.find('<', at + 1)) {
		auto const rest = view.substr(at);
		auto held = false;
		for (auto const& marker : markers) {
			switch (compare(rest, marker.text)) {
			case Prefix::Whole:
				return Hit{.begin = at,
				           .end = marker.reach == Reach::Consumed ? at + marker.text.size() : at,
				           .marker = marker.marker,
				           .name = {}};
			case Prefix::Part:
				held = true;
				break;
			case Prefix::None:
				break;
			}
		}
		if (in_content) {
			auto const open = match_open(rest, input);
			if (open.prefix == Prefix::Whole) {
				return Hit{.begin = at, .end = at + open.end, .marker = Marker::CallOpen, .name = open.name};
			}
			held = held || open.prefix == Prefix::Part;
		}
		if (held && input == Feed::Open) {
			return Held{.begin = at};
		}
	}
	return Clear{};
}

auto Parser::drain(Feed input) -> std::vector<Event> {
	auto events = std::vector<Event>{};
	while (true) {
		auto const found = scan(input);
		if (std::holds_alternative<Hit>(found)) {
			auto const& hit = std::get<Hit>(found);
			// The text before a marker ends its segment.
			auto const used = deliver(events, std::string_view{pending_}.substr(0, hit.begin), Feed::Ended);
			core::invariant(used == hit.begin);
			auto name = std::string{hit.name};
			auto const marker = hit.marker;
			pending_.erase(0, hit.end);
			enter(events, marker, std::move(name));
			continue;
		}
		auto const limit = std::holds_alternative<Held>(found) ? std::get<Held>(found).begin : pending_.size();
		auto const used =
		    deliver(events, std::string_view{pending_}.substr(0, limit), std::holds_alternative<Held>(found) ? Feed::Open : input);
		pending_.erase(0, used);
		return events;
	}
}

auto Parser::deliver(std::vector<Event>& events, std::string_view text, Feed input) -> std::size_t {
	if (std::holds_alternative<InCall>(region_)) {
		std::get<InCall>(region_).body.append(text);
		return text.size();
	}
	auto& in = std::get<InText>(region_);
	auto begin = std::size_t{0};
	if (in.segment == Segment::Fresh) {
		while (begin < text.size()) {
			auto const c = front(text.substr(begin));
			if (!c) {
				if (input == Feed::Open) {
					return begin;
				}
				break;
			}
			if (!python_space(c->value)) {
				break;
			}
			begin += c->length;
		}
	}
	// The end of the last character that is not whitespace; what follows it waits for more text, or is trimmed.
	auto end = begin;
	auto at = begin;
	while (at < text.size()) {
		auto const c = front(text.substr(at));
		if (!c) {
			if (input == Feed::Open) {
				break;
			}
			end = text.size();
			break;
		}
		at += c->length;
		if (!python_space(c->value)) {
			end = at;
		}
	}
	if (end > begin) {
		auto delta = std::string{text.substr(begin, end - begin)};
		if (in.field == Field::Reasoning) {
			events.emplace_back(ReasoningDelta{.text = std::move(delta)});
		} else {
			events.emplace_back(ContentDelta{.text = std::move(delta)});
		}
		in.segment = Segment::Started;
	}
	return input == Feed::Open ? end : text.size();
}

auto Parser::enter(std::vector<Event>& events, Marker marker, std::string name) -> void {
	auto const fresh = [](Field field) {
		return InText{.field = field, .segment = Segment::Fresh};
	};
	switch (marker) {
	case Marker::ThinkOpen:
		region_ = fresh(Field::Reasoning);
		return;
	case Marker::ThinkClose:
	case Marker::MessageEnd:
		region_ = fresh(Field::Content);
		return;
	case Marker::CallNext:
	case Marker::CallClose:
		if (std::holds_alternative<InCall>(region_)) {
			close_call(events);
		}
		region_ = fresh(Field::Content);
		return;
	case Marker::CallOpen:
		events.emplace_back(ToolCallStart{.index = calls_, .name = name});
		region_ = InCall{.index = calls_, .name = std::move(name), .body = {}};
		++calls_;
		return;
	}
	std::unreachable();
}

auto Parser::close_call(std::vector<Event>& events) -> void {
	auto const& call = std::get<InCall>(region_);
	events.emplace_back(ToolCallEnd{.index = call.index, .arguments = arguments(call.name, call.body)});
	region_ = InText{.field = Field::Content, .segment = Segment::Fresh};
}

auto Parser::arguments(std::string_view name, std::string_view body) -> std::string {
	auto out = std::string{"{"};
	for (auto const [index, parameter] : std::views::enumerate(written_parameters(body))) {
		out.append(index == 0 ? "" : ", ");
		put_string(out, parameter.name);
		out.append(": ");
		auto const casts = casting_ ? casting_->casts(name, parameter.name) : std::span<Cast const>{};
		if (casting_ && !casts.empty()) {
			out.append(cast_argument(parameter.text, casts, casting_->reader));
		} else {
			put_string(out, parameter.text);
		}
	}
	out.push_back('}');
	return out;
}

/** The string `put_string` wrote at the front of `text`, read back, and the bytes it took. */
struct ReadString {
	std::string value;
	std::size_t length;
};

[[nodiscard]] auto read_string(std::string_view text) -> ReadString {
	core::invariant(text.starts_with('"'));
	auto out = ReadString{.value = {}, .length = 1};
	while (true) {
		core::invariant(out.length < text.size());
		auto const byte = text[out.length];
		if (byte == '"') {
			out.length += 1;
			return out;
		}
		if (byte != '\\') {
			out.value.push_back(byte);
			out.length += 1;
			continue;
		}
		core::invariant(out.length + 1 < text.size());
		auto const escape = text[out.length + 1];
		out.length += 2;
		switch (escape) {
		case '"':
		case '\\':
			out.value.push_back(escape);
			break;
		case 'n':
			out.value.push_back('\n');
			break;
		case 'r':
			out.value.push_back('\r');
			break;
		case 't':
			out.value.push_back('\t');
			break;
		case 'b':
			out.value.push_back('\b');
			break;
		case 'f':
			out.value.push_back('\f');
			break;
		default: {
			// `put_string` writes every other control as `\u00xx`, lowercase.
			core::invariant(escape == 'u' && out.length + 4 <= text.size() && text.substr(out.length, 2) == "00");
			auto code = 0U;
			auto const digits = text.substr(out.length + 2, 2);
			auto const read = std::from_chars(digits.data(), std::to_address(digits.end()), code, 16);
			core::invariant(read.ptr == std::to_address(digits.end()) && code < 0x20U);
			out.value.push_back(static_cast<char>(code));
			out.length += 4;
		}
		}
	}
}

/**
 * A custom tool call's input, read back from the arguments the parser reports for it: a custom tool's one parameter,
 * `input`, is never cast, so its arguments are an object of strings as `json.dumps` writes them, which this reads
 * exactly. Nothing when the model wrote no `input` parameter. Arguments of any other shape are a caller's bug.
 */
export [[nodiscard]] auto custom_input(std::string_view arguments) -> std::optional<std::string> {
	core::invariant(arguments.starts_with('{') && arguments.ends_with('}'));
	auto rest = arguments.substr(1, arguments.size() - 2);
	auto input = std::optional<std::string>{};
	while (!rest.empty()) {
		auto const name = read_string(rest);
		rest.remove_prefix(name.length);
		core::invariant(rest.starts_with(": "));
		rest.remove_prefix(2);
		auto const value = read_string(rest);
		rest.remove_prefix(value.length);
		if (name.value == "input") {
			input = value.value;
		}
		if (!rest.empty()) {
			core::invariant(rest.starts_with(", "));
			rest.remove_prefix(2);
		}
	}
	return input;
}

/** A call in a whole reply: its name and its arguments' JSON. */
export struct ParsedCall {
	std::string name;
	std::string arguments;

	[[nodiscard]] auto operator==(ParsedCall const&) const -> bool = default;
};

/** A whole reply: the events of a parse, gathered. */
export struct Reply {
	std::string reasoning;
	std::string content;
	std::vector<ParsedCall> calls;

	auto add(Event const& event) -> void {
		if (std::holds_alternative<ReasoningDelta>(event)) {
			reasoning.append(std::get<ReasoningDelta>(event).text);
		} else if (std::holds_alternative<ContentDelta>(event)) {
			content.append(std::get<ContentDelta>(event).text);
		} else if (std::holds_alternative<ToolCallStart>(event)) {
			auto const& start = std::get<ToolCallStart>(event);
			core::invariant(start.index == calls.size());
			calls.push_back(ParsedCall{.name = start.name, .arguments = {}});
		} else {
			auto const& end = std::get<ToolCallEnd>(event);
			core::invariant(end.index + 1 == calls.size());
			calls.back().arguments = end.arguments;
		}
	}

	[[nodiscard]] auto operator==(Reply const&) const -> bool = default;
};

/** Parses a whole reply at once. */
export [[nodiscard]] auto parse(std::string_view output, std::span<json::Tool const> tools, Thinking thinking) -> core::Result<Reply, json::ReadError> {
	auto parser = Parser::make(tools, thinking);
	if (!parser) {
		return std::unexpected(parser.error());
	}
	auto reply = Reply{};
	for (auto const& event : parser->push(output)) {
		reply.add(event);
	}
	for (auto const& event : parser->finish()) {
		reply.add(event);
	}
	return reply;
}

}
