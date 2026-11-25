// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:response;

import std;
import engine.core;
import :text;
import :write;

/**
 * The OpenAI chat completion as the engine writes it: one whole `chat.completion`, or the `chat.completion.chunk`
 * objects of a stream, each the payload of one server-sent event. Each is appended to a caller's bounded buffer
 * whole or not at all. The event framing and the stream's closing `[DONE]` are not JSON: they are the http layer's.
 */
namespace engine::json {

export enum class FinishReason : std::uint8_t {
	Stop,
	Length,
	ToolCalls,
};

[[nodiscard]] constexpr auto spelling(FinishReason reason) -> std::string_view {
	switch (reason) {
	case FinishReason::Stop:
		return "stop";
	case FinishReason::Length:
		return "length";
	case FinishReason::ToolCalls:
		return "tool_calls";
	}
	std::unreachable();
}

/** Token counts. The total is their sum and is written, never held; cached tokens are part of the prompt's. */
export struct Usage {
	std::uint64_t prompt_tokens;
	std::uint64_t completion_tokens;
	std::uint64_t cached_tokens;

	[[nodiscard]] auto operator==(Usage const&) const -> bool = default;
};

/** A tool call the model made. Its arguments are one JSON object, written as the string OpenAI's schema carries. */
export struct ToolCallOutput {
	std::string id;
	std::string name;
	JsonText arguments;

	[[nodiscard]] auto operator==(ToolCallOutput const&) const -> bool = default;
};

/** A whole, non-streamed completion: one choice. */
export struct ChatCompletion {
	std::string id;
	/** Unix seconds. */
	std::int64_t created;
	std::string model;
	std::string content;
	std::string reasoning_content;
	std::vector<ToolCallOutput> tool_calls;
	FinishReason finish_reason;
	Usage usage;

	[[nodiscard]] auto operator==(ChatCompletion const&) const -> bool = default;
};

template<Sink O>
auto put_usage(Members<O>& members, Usage const& usage) -> void {
	core::invariant(usage.cached_tokens <= usage.prompt_tokens);
	core::invariant(usage.completion_tokens <= std::numeric_limits<std::uint64_t>::max() - usage.prompt_tokens);
	members.object("usage", [&usage](Members<O>& inner) {
		inner.integer("prompt_tokens", usage.prompt_tokens);
		inner.integer("completion_tokens", usage.completion_tokens);
		inner.integer("total_tokens", usage.prompt_tokens + usage.completion_tokens);
		inner.object("prompt_tokens_details", [&usage](Members<O>& details) { details.integer("cached_tokens", usage.cached_tokens); });
	});
}

template<Sink O>
auto write_completion(O& out, ChatCompletion const& completion) -> void {
	write_object(out, [&completion](Members<O>& members) {
		members.string("id", completion.id);
		members.string("object", "chat.completion");
		members.integer("created", completion.created);
		members.string("model", completion.model);
		members.array("choices", [&completion](Elements<O>& choices) {
			choices.object([&completion](Members<O>& choice) {
				choice.integer("index", 0);
				choice.object("message", [&completion](Members<O>& message) {
					message.string("role", "assistant");
					message.string("content", completion.content);
					message.string("reasoning_content", completion.reasoning_content);
					// Clients take any `tool_calls` member, even an empty one, as a turn that called tools.
					if (!completion.tool_calls.empty()) {
						message.array("tool_calls", [&completion](Elements<O>& calls) {
							for (auto const& call : completion.tool_calls) {
								calls.object([&call](Members<O>& written) {
									written.string("id", call.id);
									written.string("type", "function");
									written.object("function", [&call](Members<O>& function) {
										function.string("name", call.name);
										function.string("arguments", call.arguments.text());
									});
								});
							}
						});
					}
				});
				choice.string("finish_reason", spelling(completion.finish_reason));
			});
		});
		put_usage(members, completion.usage);
	});
}

/** Appends a whole completion's JSON body. */
export [[nodiscard]] auto append(Buffer& buffer, ChatCompletion const& completion) -> core::Result<void, Overflow> {
	return buffer.append([&completion](auto& out) { write_completion(out, completion); });
}

/** A whole completion's JSON body, in a string of exactly its size. */
export [[nodiscard]] auto encode(ChatCompletion const& completion) -> std::string {
	return encode([&completion](auto& out) { write_completion(out, completion); });
}

/**
 * What every chunk of one stream repeats. The events below are borrowed views, alive for the one call that writes
 * them: a stream's text passes through without being copied.
 */
export struct ChunkHeader {
	std::string_view id;
	/** Unix seconds, the same for every chunk of the stream. */
	std::int64_t created;
	std::string_view model;
};

/** The first chunk: the assistant role, before any text. */
export struct Opening {};

export struct ContentDelta {
	std::string_view text;
};

export struct ReasoningDelta {
	std::string_view text;
};

/** A tool call begins: its position among the turn's calls, its id and its function; arguments follow in pieces. */
export struct ToolCallStart {
	std::uint32_t index;
	std::string_view id;
	std::string_view name;
};

/** The next piece of a tool call's arguments text: the pieces of one call, concatenated, are its whole arguments. */
export struct ArgumentsDelta {
	std::uint32_t index;
	std::string_view fragment;
};

export struct Finish {
	FinishReason reason;
};

/** The last chunk of a stream that asked for usage: no choices, the usage. */
export struct UsageReport {
	Usage usage;
};

export using ChunkEvent = std::variant<Opening, ContentDelta, ReasoningDelta, ToolCallStart, ArgumentsDelta, Finish, UsageReport>;

/** Writes the one choice of a chunk: its delta, and its finish reason. */
template<Sink O>
struct Delta {
	Elements<O>& choices;

	auto choice(std::invocable<Members<O>&> auto const& fill, std::optional<FinishReason> finish) const -> void {
		choices.object([&](Members<O>& choice) {
			choice.integer("index", 0);
			choice.object("delta", fill);
			if (finish) {
				choice.string("finish_reason", spelling(*finish));
			} else {
				choice.null("finish_reason");
			}
		});
	}

	auto operator()(Opening const& /*opening*/) const -> void {
		choice([](Members<O>& delta) { delta.string("role", "assistant"); }, std::nullopt);
	}

	auto operator()(ContentDelta const& delta) const -> void {
		choice([&delta](Members<O>& written) { written.string("content", delta.text); }, std::nullopt);
	}

	auto operator()(ReasoningDelta const& delta) const -> void {
		choice([&delta](Members<O>& written) { written.string("reasoning_content", delta.text); }, std::nullopt);
	}

	auto operator()(ToolCallStart const& start) const -> void {
		choice(
		    [&start](Members<O>& written) {
			    written.array("tool_calls", [&start](Elements<O>& calls) {
				    calls.object([&start](Members<O>& call) {
					    call.integer("index", start.index);
					    call.string("id", start.id);
					    call.string("type", "function");
					    call.object("function", [&start](Members<O>& function) {
						    function.string("name", start.name);
						    function.string("arguments", "");
					    });
				    });
			    });
		    },
		    std::nullopt);
	}

	auto operator()(ArgumentsDelta const& delta) const -> void {
		choice(
		    [&delta](Members<O>& written) {
			    written.array("tool_calls", [&delta](Elements<O>& calls) {
				    calls.object([&delta](Members<O>& call) {
					    call.integer("index", delta.index);
					    call.object("function", [&delta](Members<O>& function) { function.string("arguments", delta.fragment); });
				    });
			    });
		    },
		    std::nullopt);
	}

	auto operator()(Finish const& finish) const -> void {
		choice([](Members<O>& /*delta*/) {}, finish.reason);
	}

	auto operator()(UsageReport const& /*report*/) const -> void {}
};

template<Sink O>
auto write_chunk(O& out, ChunkHeader const& header, ChunkEvent const& event) -> void {
	write_object(out, [&](Members<O>& members) {
		members.string("id", header.id);
		members.string("object", "chat.completion.chunk");
		members.integer("created", header.created);
		members.string("model", header.model);
		members.array("choices", [&event](Elements<O>& choices) { std::visit(Delta<O>{.choices = choices}, event); });
		if (std::holds_alternative<UsageReport>(event)) {
			put_usage(members, std::get<UsageReport>(event).usage);
		}
	});
}

/** Appends one chunk's JSON: the payload of one server-sent event, which the http layer frames. */
export [[nodiscard]] auto append(Buffer& buffer, ChunkHeader const& header, ChunkEvent const& event) -> core::Result<void, Overflow> {
	return buffer.append([&](auto& out) { write_chunk(out, header, event); });
}

}
