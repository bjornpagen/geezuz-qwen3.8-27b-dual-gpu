// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:gather;

import std;
import engine.core;
import engine.http;
import engine.json;
import :engine;
import :render;
import :stream;

namespace engine::openai {

/** A call that has begun and not ended. */
struct OpenCall {
	std::uint32_t index;
	std::string name;
};

/** A whole (non-streamed) completion as it accumulates. */
struct Gathering {
	std::string content;
	std::string reasoning;
	std::vector<json::ToolCallOutput> calls;
	std::optional<OpenCall> open;
	/** Text held in all, against the bound of one response. */
	std::size_t bytes;
};

struct Gathered {};

/** The completion is over; the response is this. */
struct Complete {
	json::ChatCompletion completion;
};

/** The completion outgrew one response; the request is cancelled and refused. */
struct Outgrown {};

using GatherStep = std::variant<Gathered, Complete, Outgrown>;

/** One output of an accepted, non-streamed request. */
struct Gather {
	Gathering& gathering;
	json::Reader<json::fragment_bounds>& reader;
	Frame frame;

	[[nodiscard]] auto grow(std::size_t added) const -> GatherStep {
		gathering.bytes += added;
		return gathering.bytes > http::max_response_body_bytes ? GatherStep{Outgrown{}} : GatherStep{Gathered{}};
	}

	auto operator()(ReasoningText const& text) const -> GatherStep {
		gathering.reasoning += text.text;
		return grow(text.text.size());
	}

	auto operator()(ContentText const& text) const -> GatherStep {
		gathering.content += text.text;
		return grow(text.text.size());
	}

	auto operator()(CallBegin& begin) const -> GatherStep {
		core::invariant(!gathering.open && begin.index == gathering.calls.size() && !begin.name.empty() &&
		                begin.name.size() <= max_tool_name_bytes);
		auto const size = begin.name.size();
		gathering.open = OpenCall{.index = begin.index, .name = std::move(begin.name)};
		return grow(size);
	}

	// An end comes only inside a call: `value()` aborts on an engine that sends one outside.
	auto operator()(CallEnd const& end) const -> GatherStep {
		auto& open = gathering.open.value();
		core::invariant(end.index == open.index);
		if (end.arguments.size() > json::fragment_bounds.document_bytes) {
			return Outgrown{};
		}
		// The engine writes every call's arguments as one JSON object; anything else is its bug.
		auto arguments = json::JsonText::from(reader, end.arguments);
		core::invariant(arguments.has_value());
		gathering.calls.push_back(
		    json::ToolCallOutput{.id = call_id(frame.serial, end.index), .name = std::move(open.name), .arguments = *std::move(arguments)});
		gathering.open.reset();
		return grow(end.arguments.size());
	}

	auto operator()(Finished const& finished) const -> GatherStep {
		core::invariant(finished.ending == Ending::Length || !gathering.open);
		auto const reason = finish_reason(finished.ending, gathering.calls.size());
		return Complete{.completion = json::ChatCompletion{
		                    .id = completion_id(frame.serial),
		                    .created = frame.created,
		                    .model = std::string{model_id},
		                    .content = std::move(gathering.content),
		                    .reasoning_content = std::move(gathering.reasoning),
		                    .tool_calls = std::move(gathering.calls),
		                    .finish_reason = reason,
		                    .usage = finished.usage,
		                }};
	}
};

}
