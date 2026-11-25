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
import :request;
import :decode_support;
import :decode_neutral;
import :decode_messages;
import :decode_tools;

namespace engine::json {

[[nodiscard]] auto stop(Body& value) -> Decoded<std::vector<std::string>> {
	auto const kind = kind_of(value);
	if (!kind) {
		return std::unexpected(kind.error());
	}
	switch (*kind) {
	case Kind::Null:
		return lift(value.as_null()).transform([] { return std::vector<std::string>{}; });
	case Kind::String:
		return text(value).transform([](std::string read) { return std::vector<std::string>{std::move(read)}; });
	case Kind::Array: {
		auto sequences = std::vector<std::string>{};
		auto const visited = each_element(value, [&sequences](std::size_t index, Body& element) -> Decoded<void> {
			if (index == max_stop_sequences) {
				return refuse(TooMany{.limit = max_stop_sequences});
			}
			return text(element).transform([&sequences](std::string read) { sequences.push_back(std::move(read)); });
		});
		if (!visited) {
			// Too many is a fault of the whole list, not of its last element.
			auto error = visited.error();
			if (std::holds_alternative<TooMany>(error.fault)) {
				error.path.clear();
			}
			return std::unexpected(std::move(error));
		}
		return sequences;
	}
	case Kind::Boolean:
	case Kind::Integer:
	case Kind::Real:
	case Kind::Object:
		return refuse(WrongShape{.expected = Expected::Stop, .found = *kind});
	}
	std::unreachable();
}

/** `stream_options.include_usage`, read from a present `stream_options`. */
[[nodiscard]] auto include_usage(Body& value) -> Decoded<bool> {
	auto include = std::optional<bool>{};
	auto const visited = each_member(value, [&include](std::string_view name, Body& member) -> Decoded<void> {
		if (name == "include_usage") {
			return nullable<bool>(member, [](Body& flag) { return lift(flag.as_boolean()); }).transform([&include](std::optional<bool> read) {
				include.emplace(read.value_or(false));
			});
		}
		if (name == "include_obfuscation") {
			return neutral(member, Neutral::False);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return include.value_or(false);
}

[[nodiscard]] auto template_options(Body& value) -> Decoded<TemplateOptions> {
	auto options = TemplateOptions{};
	auto const visited = each_member(value, [&options](std::string_view name, Body& member) -> Decoded<void> {
		auto const flag = [&member] { return nullable<bool>(member, [](Body& inner) { return lift(inner.as_boolean()); }); };
		if (name == "enable_thinking") {
			return flag().transform([&options](std::optional<bool> read) {
				options.thinking = read.transform([](bool on) { return on ? Thinking::Enabled : Thinking::Disabled; });
			});
		}
		if (name == "preserve_thinking") {
			return flag().transform([&options](std::optional<bool> read) {
				options.past_thinking = read.transform([](bool on) { return on ? PastThinking::Preserved : PastThinking::Dropped; });
			});
		}
		if (name == "reasoning_effort") {
			auto const effort = nullable<std::string>(member, [](Body& inner) { return text(inner); });
			if (!effort) {
				return std::unexpected(effort.error());
			}
			if (!*effort) {
				return {};
			}
			if (**effort == "low") {
				options.reasoning_effort = ReasoningEffort::Low;
			} else if (**effort == "medium") {
				options.reasoning_effort = ReasoningEffort::Medium;
			} else if (**effort == "xhigh") {
				options.reasoning_effort = ReasoningEffort::XHigh;
			} else {
				return refuse(Violation::UnknownValue);
			}
			return {};
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	return options;
}

/** An optional member read with `read`; `null` leaves it absent. */
template<class T, class F>
[[nodiscard]] auto assign(std::optional<T>& slot, Body& value, F const& read) -> Decoded<void> {
	return nullable<T>(value, read).transform([&slot](std::optional<T> decoded) { slot = std::move(decoded); });
}

[[nodiscard]] auto request(Body& root, Fragments& fragments) -> Decoded<ChatRequest> {
	auto decoded = ChatRequest{
	    .model = std::nullopt,
	    .messages = {},
	    .tools = {},
	    .tool_choice = std::nullopt,
	    .delivery = Delivery::Whole,
	    .max_tokens = std::nullopt,
	    .temperature = std::nullopt,
	    .top_p = std::nullopt,
	    .top_k = std::nullopt,
	    .min_p = std::nullopt,
	    .seed = std::nullopt,
	    .stop = {},
	    .template_options = {},
	};
	auto messages = std::optional<std::vector<Message>>{};
	auto stream = std::optional<bool>{};
	auto stream_usage = std::optional<bool>{};
	auto max_tokens = std::optional<std::int64_t>{};
	auto max_completion_tokens = std::optional<std::int64_t>{};
	auto const integer = [](Body& value) { return lift(value.as_integer()); };
	auto const real = [](Body& value) { return number(value); };
	auto const visited = each_member(root, [&](std::string_view name, Body& member) -> Decoded<void> {
		if (name == "messages") {
			auto& list = messages.emplace();
			return each_element(member, [&](std::size_t, Body& element) -> Decoded<void> {
				return message(element, fragments).transform([&list](Message read) { list.push_back(std::move(read)); });
			});
		}
		if (name == "tools") {
			return nullable<bool>(member, [&](Body& list) -> Decoded<bool> {
				       return each_element(list, [&](std::size_t, Body& element) -> Decoded<void> {
					              return tool(element, fragments).transform([&decoded](Tool read) { decoded.tools.push_back(std::move(read)); });
				              })
				           .transform([] { return true; });
			       })
			    .transform([](std::optional<bool>) {});
		}
		if (name == "model") {
			return assign(decoded.model, member, [](Body& value) { return text(value); });
		}
		if (name == "tool_choice") {
			return assign(decoded.tool_choice, member, [](Body& value) { return tool_choice(value); });
		}
		if (name == "stream") {
			return assign(stream, member, [](Body& value) { return lift(value.as_boolean()); });
		}
		if (name == "stream_options") {
			return assign(stream_usage, member, [](Body& value) { return include_usage(value); });
		}
		if (name == "max_tokens") {
			return assign(max_tokens, member, integer);
		}
		if (name == "max_completion_tokens") {
			return assign(max_completion_tokens, member, integer);
		}
		if (name == "temperature") {
			return assign(decoded.temperature, member, real);
		}
		if (name == "top_p") {
			return assign(decoded.top_p, member, real);
		}
		if (name == "top_k") {
			return assign(decoded.top_k, member, integer);
		}
		if (name == "min_p") {
			return assign(decoded.min_p, member, real);
		}
		if (name == "seed") {
			return assign(decoded.seed, member, integer);
		}
		if (name == "stop") {
			return stop(member).transform([&decoded](std::vector<std::string> read) { decoded.stop = std::move(read); });
		}
		if (name == "chat_template_kwargs") {
			return nullable<TemplateOptions>(member, [](Body& value) { return template_options(value); }).transform([&decoded](std::optional<TemplateOptions> read) {
				decoded.template_options = read.value_or(TemplateOptions{});
			});
		}
		if (auto const neutral_value = tolerated(request_tolerated, name)) {
			return neutral(member, *neutral_value);
		}
		return refuse(Violation::UnknownField);
	});
	if (!visited) {
		return std::unexpected(visited.error());
	}
	if (!messages) {
		return refuse_at("messages", Violation::MissingField);
	}
	decoded.messages = std::move(*messages);
	if (max_tokens && max_completion_tokens) {
		return refuse_at("max_completion_tokens", Conflict{.other = "max_tokens"});
	}
	decoded.max_tokens = max_completion_tokens ? max_completion_tokens : max_tokens;
	if (!stream.value_or(false)) {
		if (stream_usage) {
			return refuse_at("stream_options", Conflict{.other = "stream"});
		}
		decoded.delivery = Delivery::Whole;
	} else {
		decoded.delivery = stream_usage.value_or(false) ? Delivery::StreamWithUsage : Delivery::Stream;
	}
	return decoded;
}

/**
 * Decodes OpenAI chat requests into `ChatRequest`, one at a time, refusing every request it cannot serve exactly
 * with the member at fault. It owns its two readers (the body's, and one for JSON a request holds in a string), each
 * allocated once; decoding reads every part of the request, so nothing it accepts goes unchecked.
 *
 * Unknown members are refused by name, at every level. OpenAI's own endpoint refuses unrecognized arguments, so no
 * OpenAI client depends on extras being ignored; and an ignored sampling or output parameter would serve the request
 * differently from what it asked for. Parameters OpenAI defines that the engine does not implement are accepted
 * only with their neutral value (`n: 1`, `presence_penalty: 0`, `logprobs: false`, ...), which clients commonly send
 * as defaults; account identifiers (`user`, `prompt_cache_key`) are accepted and dropped.
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
	[[nodiscard]] auto decode(PaddedView text) -> core::Result<ChatRequest, DecodeError> {
		auto root = lift(body_->read(text));
		if (!root) {
			return std::unexpected(root.error());
		}
		return finish(request(*root, *fragments_));
	}

	/** Copies a body into the decoder's padded buffer, then decodes it. */
	[[nodiscard]] auto decode_copy(std::string_view text) -> core::Result<ChatRequest, DecodeError> {
		auto root = lift(body_->read_copy(text));
		if (!root) {
			return std::unexpected(root.error());
		}
		return finish(request(*root, *fragments_));
	}

private:
	RequestDecoder(std::unique_ptr<Reader<request_bounds>> body, std::unique_ptr<Fragments> fragments)
	    : body_{std::move(body)}, fragments_{std::move(fragments)} {}

	[[nodiscard]] auto finish(Decoded<ChatRequest> decoded) -> core::Result<ChatRequest, DecodeError> {
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
