// SPDX-License-Identifier: AGPL-3.0-only
export module engine.store;

import std;
import engine.core;
import engine.json;
import engine.tokenizer;

/**
 * Stored responses: what `previous_response_id`, `GET` and `DELETE /v1/responses/{id}` and the input items list read.
 * One owner (the api's service, on the server's thread) holds the store; it is bounded in entries and bytes and
 * evicts the least recently used.
 */
namespace engine::store {

/** A response's id: 128 bits from the server's randomness, spelled `resp_` and 32 lowercase hex digits. */
export struct ResponseId {
	std::uint64_t high;
	std::uint64_t low;

	/** The id a client names, or nothing when the text is no id this server could have issued. */
	[[nodiscard]] static auto parse(std::string_view text) -> std::optional<ResponseId> {
		constexpr std::string_view prefix = "resp_";
		if (text.size() != prefix.size() + 32 || !text.starts_with(prefix)) {
			return std::nullopt;
		}
		auto const half = [](std::string_view digits) -> std::optional<std::uint64_t> {
			auto value = std::uint64_t{0};
			for (auto const digit : digits) {
				auto const at = std::string_view{"0123456789abcdef"}.find(digit);
				if (at == std::string_view::npos) {
					return std::nullopt;
				}
				value = (value << 4U) | at;
			}
			return value;
		};
		auto const high = half(text.substr(prefix.size(), 16));
		auto const low = half(text.substr(prefix.size() + 16, 16));
		if (!high || !low) {
			return std::nullopt;
		}
		return ResponseId{.high = *high, .low = *low};
	}

	[[nodiscard]] auto text() const -> std::string {
		return std::format("resp_{:016x}{:016x}", high, low);
	}

	[[nodiscard]] constexpr auto operator<=>(ResponseId const&) const = default;
};

/** A stored response: the response as it was answered, its input items, and the exact token sequence it leaves. */
export struct Stored {
	json::Response response;
	std::vector<json::StoredInput> input;
	/** The prompt as it ran, then every generated token: what a continuation submits before its new input. */
	std::vector<tokenizer::Id> sequence;

	[[nodiscard]] auto operator==(Stored const&) const -> bool = default;
};

/** A store's bounds: the most responses it holds at once, and the bytes they may take in all, by `footprint`. */
export struct Bounds {
	std::size_t responses;
	std::size_t bytes;
};

/**
 * The server's store. A stored response takes at most the context's tokens and one response body's text, about 17 mb,
 * so the byte bound holds over a hundred of the largest.
 */
export inline constexpr Bounds server_bounds{.responses = 4096, .bytes = std::size_t{2} << 30U};

/** What each item and each response costs beyond its text, so that many small ones count against the bound too. */
constexpr std::size_t entry_cost = 256;

struct TextBytes {
	[[nodiscard]] auto operator()(json::MessageItem const& item) const -> std::size_t {
		return item.text.size();
	}

	[[nodiscard]] auto operator()(json::FunctionCallItem const& item) const -> std::size_t {
		return item.call_id.size() + item.name.size() + item.arguments.text().size();
	}

	[[nodiscard]] auto operator()(json::FunctionOutputItem const& item) const -> std::size_t {
		return item.call_id.size() + item.output.size();
	}

	[[nodiscard]] auto operator()(json::CustomCallItem const& item) const -> std::size_t {
		return item.call_id.size() + item.name.size() + item.input.size();
	}

	[[nodiscard]] auto operator()(json::CustomOutputItem const& item) const -> std::size_t {
		return item.call_id.size() + item.output.size();
	}

	[[nodiscard]] auto operator()(json::ReasoningItem const& item) const -> std::size_t {
		return item.text.size();
	}

	[[nodiscard]] auto operator()(json::ReasoningOutput const& item) const -> std::size_t {
		return item.id.size() + item.text.size();
	}

	[[nodiscard]] auto operator()(json::MessageOutput const& item) const -> std::size_t {
		return item.id.size() + item.text.size();
	}

	[[nodiscard]] auto operator()(json::FunctionCallOutput const& item) const -> std::size_t {
		return item.id.size() + item.call_id.size() + item.name.size() + item.arguments.size();
	}

	[[nodiscard]] auto operator()(json::CustomCallOutput const& item) const -> std::size_t {
		return item.id.size() + item.call_id.size() + item.name.size() + item.input.size();
	}
};

/** What a stored response takes against a store's byte bound: its tokens, its text, and a fixed cost per part. */
export [[nodiscard]] auto footprint(Stored const& stored) -> std::size_t {
	auto bytes = entry_cost + (stored.sequence.size() * sizeof(tokenizer::Id));
	for (auto const& input : stored.input) {
		bytes += entry_cost + input.id.size() + std::visit(TextBytes{}, input.item);
	}
	for (auto const& output : stored.response.output) {
		bytes += entry_cost + std::visit(TextBytes{}, output);
	}
	auto const& response = stored.response;
	bytes += response.instructions.value_or(std::string{}).size();
	for (auto const& pair : response.metadata) {
		bytes += pair.key.size() + pair.value.size();
	}
	for (auto const& tool : response.tools) {
		bytes += entry_cost + tool.name.size() + tool.description.value_or(std::string{}).size();
		if (auto const function = core::alternative<json::FunctionTool>(tool.input); function && function->parameters) {
			bytes += function->parameters->text().size();
		} else if (auto const custom = core::alternative<json::CustomTool>(tool.input); custom && custom->grammar) {
			bytes += custom->grammar->definition.size();
		}
	}
	return bytes;
}

export enum class Removal : std::uint8_t {
	Removed,
	/** No such response is stored: it never was, it was deleted, or it was evicted. */
	Absent,
};

struct Hash {
	[[nodiscard]] auto operator()(ResponseId const& id) const -> std::size_t {
		return std::hash<std::uint64_t>{}(id.high ^ (id.low * 0x9e37'79b9'7f4a'7c15ULL));
	}
};

/**
 * The stored responses, least recently used first. Reading a response for a continuation, a retrieval or a listing
 * makes it the most recently used, so a parent many requests branch from stays while they run; a branch copies the
 * parent's sequence when it is submitted and needs nothing of the parent afterwards.
 */
export template<Bounds L>
class BasicStore {
public:
	static_assert(L.responses > 0 && L.bytes > 0);

	BasicStore() {
		entries_.reserve(L.responses);
	}

	/** Stores a response under a fresh id, evicting the least recently used until it fits. */
	auto insert(ResponseId id, Stored&& stored) -> void {
		auto entry = Entry{.stored = std::move(stored), .at = {}, .bytes = 0};
		entry.bytes = footprint(entry.stored);
		// A response is bounded by the context and the largest response body, far below the store's bound.
		core::invariant(entry.bytes <= L.bytes && !entries_.contains(id));
		while (entries_.size() == L.responses || bytes_ + entry.bytes > L.bytes) {
			evict();
		}
		order_.push_back(id);
		entry.at = std::prev(order_.end());
		bytes_ += entry.bytes;
		entries_.emplace(id, std::move(entry));
	}

	/** The stored response, now the most recently used; nothing when it is not stored. */
	[[nodiscard]] auto find(ResponseId id) -> std::optional<Stored const&> {
		auto const found = entries_.find(id);
		if (found == entries_.end()) {
			return std::nullopt;
		}
		order_.splice(order_.end(), order_, found->second.at);
		return found->second.stored;
	}

	[[nodiscard]] auto remove(ResponseId id) -> Removal {
		auto const found = entries_.find(id);
		if (found == entries_.end()) {
			return Removal::Absent;
		}
		bytes_ -= found->second.bytes;
		order_.erase(found->second.at);
		entries_.erase(found);
		return Removal::Removed;
	}

	[[nodiscard]] auto size() const -> std::size_t {
		return entries_.size();
	}

	[[nodiscard]] auto bytes() const -> std::size_t {
		return bytes_;
	}

private:
	struct Entry {
		Stored stored;
		std::list<ResponseId>::iterator at;
		std::size_t bytes;
	};

	auto evict() -> void {
		core::invariant(!order_.empty());
		std::ignore = remove(order_.front());
	}

	std::list<ResponseId> order_;
	std::unordered_map<ResponseId, Entry, Hash> entries_;
	std::size_t bytes_ = 0;
};

export using Store = BasicStore<server_bounds>;

}
