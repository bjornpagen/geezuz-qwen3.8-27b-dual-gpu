// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:decode;

import std;
import engine.core;
import :bounds;
import :error;
import :text;
import :value;
import :containers;
import :reader;
import :decode_error;
import :request;
import :decode_support;
import :decode_neutral;
import :decode_format;
import :decode_tools;
import :decode_input;

namespace engine::json {

struct EffortName {
	std::string_view name;
	Effort effort;
};

/** The template's efforts by their sdk names: what a request may ask for, and what a refusal lists. */
inline constexpr auto served_efforts = std::to_array<EffortName>({
    {.name = "none", .effort = Effort::None},
    {.name = "low", .effort = Effort::Low},
    {.name = "medium", .effort = Effort::Medium},
    {.name = "xhigh", .effort = Effort::XHigh},
});

/** The sdk's efforts the template does not have. */
inline constexpr auto unserved_efforts = std::to_array<std::string_view>({"minimal", "high", "max"});

/** `reasoning.effort`: the template's four settings; the sdk's other efforts are refused by name, listing the four. */
[[nodiscard]] auto effort(Body& value) -> Decoded<Effort> {
	auto const read = text(value);
	if (!read) {
		return std::unexpected(read.error());
	}
	if (auto const served = std::ranges::find(served_efforts, *read, &EffortName::name); served != served_efforts.end()) {
		return served->effort;
	}
	if (std::ranges::contains(unserved_efforts, *read)) {
		return refuse(UnsupportedValue{
		    .supported = served_efforts | std::views::transform([](EffortName const& named) { return std::string{named.name}; }) |
		                 std::ranges::to<std::vector<std::string>>(),
		});
	}
	return refuse(Violation::UnknownValue);
}

/** `reasoning`: its effort, and members the engine accepts only at their neutral value. */
[[nodiscard]] auto reasoning(Body& value) -> Decoded<std::optional<Effort>> {
	auto read = std::optional<Effort>{};
	auto const visited = each_member(value, [&read](std::string_view name, Body& member) -> Decoded<void> {
		if (name == "effort") {
			return nullable<Effort>(member, [](Body& inner) { return effort(inner); }).transform([&read](std::optional<Effort> decoded) {
				read = decoded;
			});
		}
		if (auto const wanted = tolerated(reasoning_tolerated, name)) {
			return neutral(member, *wanted);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return read;
}

/** `stream_options`: only obfuscation, which this server never writes. */
[[nodiscard]] auto stream_options(Body& value) -> Decoded<bool> {
	auto const visited = each_member(value, [](std::string_view name, Body& member) -> Decoded<void> {
		if (name == "include_obfuscation") {
			return neutral(member, Neutral::False);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return true;
}

/** `metadata`: string pairs within the sdk's bounds. */
[[nodiscard]] auto metadata(Body& value) -> Decoded<std::vector<MetadataPair>> {
	auto pairs = std::vector<MetadataPair>{};
	auto const visited = each_member(value, [&pairs](std::string_view key, Body& member) -> Decoded<void> {
		if (pairs.size() == max_metadata_pairs) {
			return refuse(TooMany{.limit = max_metadata_pairs});
		}
		if (key.size() > max_metadata_key_bytes) {
			return refuse(TooLong{.limit = max_metadata_key_bytes});
		}
		auto read = text(member);
		if (!read) {
			return std::unexpected(read.error());
		}
		if (read->size() > max_metadata_value_bytes) {
			return refuse(TooLong{.limit = max_metadata_value_bytes});
		}
		pairs.push_back(MetadataPair{.key = std::string{key}, .value = *std::move(read)});
		return {};
	});
	if (!visited) {
		auto error = visited.error();
		// Too many is a fault of the whole object, not of its last member.
		if (std::holds_alternative<TooMany>(error.fault)) {
			error.path.clear();
		}
		return std::unexpected(std::move(error));
	}
	return pairs;
}

/** An optional member read with `read`; `null` leaves it absent. */
template<class T, class F>
[[nodiscard]] auto assign(std::optional<T>& slot, Body& value, F const& read) -> Decoded<void> {
	return nullable<T>(value, read).transform([&slot](std::optional<T> decoded) { slot = std::move(decoded); });
}

[[nodiscard]] auto create(Body& root, Fragments& fragments) -> Decoded<CreateRequest> {
	auto decoded = CreateRequest{
	    .model = std::nullopt,
	    .input = {},
	    .instructions = std::nullopt,
	    .tools = {},
	    .tool_choice = ToolChoiceMode::Auto,
	    .format = TextFormatMode::Text,
	    .effort = Effort::XHigh,
	    .max_output_tokens = std::nullopt,
	    .temperature = std::nullopt,
	    .top_p = std::nullopt,
	    .delivery = Delivery::Whole,
	    .storage = Storage::Stored,
	    .previous_response_id = std::nullopt,
	    .metadata = {},
	};
	auto input_items = std::optional<std::vector<InputItem>>{};
	auto stream = std::optional<bool>{};
	auto options = std::optional<bool>{};
	auto store = std::optional<bool>{};
	auto choice = std::optional<ToolChoice>{};
	auto format = std::optional<TextFormat>{};
	auto chosen_effort = std::optional<Effort>{};
	auto pairs = std::optional<std::vector<MetadataPair>>{};
	auto const integer = [](Body& value) { return lift(value.as_integer()); };
	auto const real = [](Body& value) { return lift(value.as_real()); };
	auto const boolean = [](Body& value) { return lift(value.as_boolean()); };
	auto const string = [](Body& value) { return text(value); };
	auto const visited = each_member(root, [&](std::string_view name, Body& member) -> Decoded<void> {
		if (name == "input") {
			return input(member, fragments).transform([&input_items](std::vector<InputItem> read) { input_items = std::move(read); });
		}
		if (name == "model") {
			return text(member).transform([&decoded](std::string&& read) { decoded.model = std::move(read); });
		}
		if (name == "instructions") {
			return assign(decoded.instructions, member, string);
		}
		if (name == "tools") {
			return nullable<bool>(member, [&](Body& list) -> Decoded<bool> {
				       return each_element(list, [&](std::size_t, Body& element) -> Decoded<void> {
					              return tool(element).transform([&decoded](Tool read) { decoded.tools.push_back(std::move(read)); });
				              })
				           .transform([] { return true; });
			       })
			    .transform([](std::optional<bool>) {});
		}
		if (name == "tool_choice") {
			return assign(choice, member, [](Body& value) { return tool_choice(value); });
		}
		if (name == "text") {
			return assign(format, member, [](Body& value) { return text_config(value); });
		}
		if (name == "reasoning") {
			return nullable<std::optional<Effort>>(member, [](Body& value) { return reasoning(value); })
			    .transform([&chosen_effort](std::optional<std::optional<Effort>> read) { chosen_effort = read.value_or(std::nullopt); });
		}
		if (name == "max_output_tokens") {
			return assign(decoded.max_output_tokens, member, integer);
		}
		if (name == "temperature") {
			return assign(decoded.temperature, member, real);
		}
		if (name == "top_p") {
			return assign(decoded.top_p, member, real);
		}
		if (name == "stream") {
			return assign(stream, member, boolean);
		}
		if (name == "stream_options") {
			return assign(options, member, [](Body& value) { return stream_options(value); });
		}
		if (name == "store") {
			return assign(store, member, boolean);
		}
		if (name == "previous_response_id") {
			return assign(decoded.previous_response_id, member, string);
		}
		if (name == "metadata") {
			return assign(pairs, member, [](Body& value) { return metadata(value); });
		}
		if (auto const wanted = tolerated(create_tolerated, name)) {
			return neutral(member, *wanted);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (!input_items) {
		return refuse_at("input", Violation::MissingField);
	}
	decoded.input = std::move(*input_items);
	decoded.tool_choice = std::move(choice).value_or(ToolChoice{ToolChoiceMode::Auto});
	decoded.format = std::move(format).value_or(TextFormat{TextFormatMode::Text});
	decoded.effort = chosen_effort.value_or(Effort::XHigh);
	decoded.storage = store.value_or(true) ? Storage::Stored : Storage::Unstored;
	decoded.metadata = std::move(pairs).value_or(std::vector<MetadataPair>{});
	if (!stream.value_or(false)) {
		if (options) {
			return refuse_at("stream_options", Conflict{.other = "stream"});
		}
		decoded.delivery = Delivery::Whole;
	} else {
		decoded.delivery = Delivery::Stream;
	}
	return decoded;
}

/**
 * Decodes Responses create requests into `CreateRequest`, one at a time, refusing every request it cannot serve
 * exactly with the member at fault. It owns its two readers (the body's, and one for the JSON a function call's
 * arguments hold in a string), each allocated once; decoding reads every part of the request, so nothing it accepts
 * goes unchecked.
 *
 * Unknown members are refused by name, at every level: the sdk's own types are the reference, and an ignored member
 * would serve the request differently from what it asked for. Members the sdk defines that the engine does not
 * implement are accepted only at their neutral value; deprecated ones are refused as deprecated.
 */
export class RequestDecoder {
public:
	[[nodiscard]] static auto make() -> core::Result<RequestDecoder, ReadError> {
		auto body = Reader<request_bounds>::make();
		if (!body) {
			return std::unexpected(body.error());
		}
		auto fragments = Reader<fragment_bounds>::make();
		if (!fragments) {
			return std::unexpected(fragments.error());
		}
		return RequestDecoder{std::make_unique<Reader<request_bounds>>(std::move(*body)), std::make_unique<Fragments>(std::move(*fragments))};
	}

	/** Decodes a body whose storage extends `padding` bytes past its end, in place. */
	[[nodiscard]] auto decode(PaddedView text) -> core::Result<CreateRequest, DecodeError> {
		auto root = lift(body_->read(text));
		if (!root) {
			return std::unexpected(root.error());
		}
		return finish(create(*root, *fragments_));
	}

	/** Copies a body into the decoder's padded buffer, then decodes it. */
	[[nodiscard]] auto decode_copy(std::string_view text) -> core::Result<CreateRequest, DecodeError> {
		auto root = lift(body_->read_copy(text));
		if (!root) {
			return std::unexpected(root.error());
		}
		return finish(create(*root, *fragments_));
	}

private:
	RequestDecoder(std::unique_ptr<Reader<request_bounds>> body, std::unique_ptr<Fragments> fragments)
	    : body_{std::move(body)}, fragments_{std::move(fragments)} {}

	[[nodiscard]] auto finish(Decoded<CreateRequest> decoded) -> core::Result<CreateRequest, DecodeError> {
		if (!decoded) {
			auto error = decoded.error();
			std::ranges::reverse(error.path);
			return std::unexpected(std::move(error));
		}
		if (auto const finished = lift(body_->finish()); !finished) {
			return std::unexpected(finished.error());
		}
		return decoded;
	}

	std::unique_ptr<Reader<request_bounds>> body_;
	std::unique_ptr<Fragments> fragments_;
};

}
