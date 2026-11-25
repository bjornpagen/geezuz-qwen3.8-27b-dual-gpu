// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:assembly;

import std;
import engine.core;
import engine.json;
import engine.sched;
import :boundary;

/**
 * A response as the engine's outputs build it, the same for a whole response and a streamed one: its output items in
 * order, the one still growing, and the steps a stream announces them by. A reply's reasoning, its answer and each
 * call are items; a run of one kind of text is one item, and text of another kind, or a call, closes it.
 */
namespace engine::openai {

/** `response.output_item.added` for the item at `index`. */
export struct Added {
	std::uint32_t index;
};

/** `response.content_part.added` for the text item at `index`. */
export struct PartOpened {
	std::uint32_t index;
};

/** More text of the item at `index`: its text, its arguments or its input. */
export struct Delta {
	std::uint32_t index;
	std::string text;
};

/** The `.done` event of the item's text. */
export struct TextClosed {
	std::uint32_t index;
};

/** `response.content_part.done` for the text item at `index`. */
export struct PartClosed {
	std::uint32_t index;
};

/** `response.output_item.done` for the item at `index`. */
export struct Done {
	std::uint32_t index;
};

/** What a stream says, in order, as a response grows. */
export using Step = std::variant<Added, PartOpened, Delta, TextClosed, PartClosed, Done>;

/** The two kinds of text a reply writes outside its calls. */
export enum class Text : std::uint8_t {
	Reasoning,
	Answer,
};

/** An item's text: the one string the item grows by. */
struct Grown {
	template<class Item>
	[[nodiscard]] auto operator()(Item& item) const -> auto& {
		if constexpr (requires { item.arguments; }) {
			return item.arguments;
		} else if constexpr (requires { item.input; }) {
			return item.input;
		} else {
			return item.text;
		}
	}
};

struct Facts {
	[[nodiscard]] auto operator()(json::ReasoningOutput const& item) const -> std::pair<json::Stream, std::string_view> {
		return {json::Stream::ReasoningText, item.id};
	}

	[[nodiscard]] auto operator()(json::MessageOutput const& item) const -> std::pair<json::Stream, std::string_view> {
		return {json::Stream::OutputText, item.id};
	}

	[[nodiscard]] auto operator()(json::FunctionCallOutput const& item) const -> std::pair<json::Stream, std::string_view> {
		return {json::Stream::Arguments, item.id};
	}

	[[nodiscard]] auto operator()(json::CustomCallOutput const& item) const -> std::pair<json::Stream, std::string_view> {
		return {json::Stream::CustomInput, item.id};
	}
};

/** How an item ends: whole, or cut off by the budget. The sdk's custom call has no status. */
struct Settle {
	json::ItemStatus status;

	auto operator()(json::ReasoningOutput& item) const -> void {
		item.status = status;
	}

	auto operator()(json::MessageOutput& item) const -> void {
		item.status = status;
	}

	auto operator()(json::FunctionCallOutput& item) const -> void {
		item.status = status;
	}

	auto operator()(json::CustomCallOutput& /*item*/) const -> void {}
};

/** One response under construction. */
export class Assembly {
public:
	explicit Assembly(json::Response response) : response_{std::move(response)} {}

	/**
	 * One output of an accepted reply (its texts, its calls, its finish), and the steps a stream announces it by.
	 * `sources` names new items and stamps the finish. The verdicts and the other terminal answers are the caller's.
	 */
	template<Sources S>
	[[nodiscard]] auto take(Output const& output, S& sources) -> std::vector<Step> {
		if (auto const reasoning = core::alternative<Reasoning>(output)) {
			return text(Text::Reasoning, reasoning->text, sources);
		}
		if (auto const content = core::alternative<Content>(output)) {
			return text(Text::Answer, content->text, sources);
		}
		if (auto const opened = core::alternative<CallBegin>(output)) {
			return begin(*opened, sources);
		}
		if (auto const closed = core::alternative<CallEnd>(output)) {
			return end(*closed);
		}
		// After its verdict, an accepted request's reply says only its texts and calls, then its finish.
		core::invariant(std::holds_alternative<Finished>(output));
		return finish(std::get<Finished>(output), sources.unix_seconds());
	}

	/** More text of `kind`: it extends the open item of its kind, or closes the open item and opens its own. */
	template<Sources S>
	[[nodiscard]] auto text(Text kind, std::string text, S& sources) -> std::vector<Step> {
		auto steps = std::vector<Step>{};
		text_bytes_ += text.size();
		if (!open_ || open_->kind != kind) {
			close(steps, json::ItemStatus::Completed);
			auto const index = next_index();
			if (kind == Text::Reasoning) {
				response_.output.emplace_back(
				    json::ReasoningOutput{.id = fresh_id(sources, "rs"), .text = {}, .status = json::ItemStatus::InProgress});
			} else {
				response_.output.emplace_back(
				    json::MessageOutput{.id = fresh_id(sources, "msg"), .text = {}, .status = json::ItemStatus::InProgress});
			}
			open_ = Open{.index = index, .kind = kind};
			steps.emplace_back(Added{.index = index});
			steps.emplace_back(PartOpened{.index = index});
		}
		std::visit(Grown{}, response_.output.back()).append(text);
		steps.emplace_back(Delta{.index = open_->index, .text = std::move(text)});
		return steps;
	}

	/** A call opens, closing the open text item: announced now, its text comes whole when it ends. */
	template<Sources S>
	[[nodiscard]] auto begin(CallBegin const& call, S& sources) -> std::vector<Step> {
		// Calls open in order, one at a time.
		core::invariant(!calling_ && call.index == calls_);
		auto steps = std::vector<Step>{};
		close(steps, json::ItemStatus::Completed);
		auto const index = next_index();
		auto id = fresh_id(sources, call.callee == Callee::Function ? "fc" : "ctc");
		auto call_id = fresh_id(sources, "call");
		text_bytes_ += call.name.size();
		switch (call.callee) {
		case Callee::Function:
			response_.output.emplace_back(json::FunctionCallOutput{
			    .id = std::move(id),
			    .call_id = std::move(call_id),
			    .name = call.name,
			    .arguments = {},
			    .status = json::ItemStatus::InProgress,
			});
			break;
		case Callee::Custom:
			response_.output.emplace_back(
			    json::CustomCallOutput{.id = std::move(id), .call_id = std::move(call_id), .name = call.name, .input = {}});
			break;
		}
		calling_ = index;
		calls_ += 1;
		steps.emplace_back(Added{.index = index});
		return steps;
	}

	/** The open call ends with its whole text. */
	[[nodiscard]] auto end(CallEnd call) -> std::vector<Step> {
		core::invariant(calling_.has_value() && call.index + 1 == calls_);
		auto const index = calling_.value();
		calling_.reset();
		text_bytes_ += call.text.size();
		auto& item = response_.output[index];
		std::visit(Grown{}, item) = call.text;
		std::visit(Settle{.status = call.closing == Closing::Closed ? json::ItemStatus::Completed : json::ItemStatus::Incomplete}, item);
		auto steps = std::vector<Step>{};
		if (!call.text.empty()) {
			steps.emplace_back(Delta{.index = index, .text = std::move(call.text)});
		}
		steps.emplace_back(TextClosed{.index = index});
		steps.emplace_back(Done{.index = index});
		return steps;
	}

	/**
	 * The reply ends: an open item ends with it, cut off when the budget ended the reply, and the response takes its
	 * status, usage and completion time.
	 */
	[[nodiscard]] auto finish(Finished const& finished, std::int64_t now) -> std::vector<Step> {
		// The parser closes every call before the reply's end.
		core::invariant(!calling_);
		auto const stopped = std::holds_alternative<sched::Stopped>(finished.ending);
		auto steps = std::vector<Step>{};
		close(steps, stopped ? json::ItemStatus::Completed : json::ItemStatus::Incomplete);
		response_.status = stopped ? json::ResponseStatus::Completed : json::ResponseStatus::Incomplete;
		response_.completed_at = now;
		response_.usage = finished.usage;
		return steps;
	}

	[[nodiscard]] auto response() const -> json::Response const& {
		return response_;
	}

	[[nodiscard]] auto item(std::uint32_t index) const -> json::OutputItem const& {
		return response_.output.at(index);
	}

	/** The stream an item's text grows by, and the item's id. */
	[[nodiscard]] auto facts(std::uint32_t index) const -> std::pair<json::Stream, std::string_view> {
		return std::visit(Facts{}, item(index));
	}

	/** The item's whole text: its text, its arguments or its input. */
	[[nodiscard]] auto text_of(std::uint32_t index) const -> std::string_view {
		return std::visit(Grown{}, item(index));
	}

	/** The bytes of text the response holds: every item's text, and every call's name. */
	[[nodiscard]] auto text_bytes() const -> std::size_t {
		return text_bytes_;
	}

private:
	/** The text item still growing, and its kind. */
	struct Open {
		std::uint32_t index;
		Text kind;
	};

	[[nodiscard]] auto next_index() const -> std::uint32_t {
		return core::narrow<std::uint32_t>(response_.output.size());
	}

	auto close(std::vector<Step>& steps, json::ItemStatus status) -> void {
		if (!open_) {
			return;
		}
		auto const index = open_->index;
		std::visit(Settle{.status = status}, response_.output[index]);
		steps.emplace_back(TextClosed{.index = index});
		steps.emplace_back(PartClosed{.index = index});
		steps.emplace_back(Done{.index = index});
		open_.reset();
	}

	json::Response response_;
	std::optional<Open> open_;
	std::optional<std::uint32_t> calling_;
	std::uint32_t calls_ = 0;
	std::size_t text_bytes_ = 0;
};

}
