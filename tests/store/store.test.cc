// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.store;
import engine.testing;

namespace {

namespace json = engine::json;
namespace store = engine::store;

/** Small bounds, so that both run out within a test: a few entries, a few of the largest generated responses. */
constexpr auto bounds = store::Bounds{.responses = 8, .bytes = 40'000};

using Store = store::BasicStore<bounds>;

[[nodiscard]] auto stored(std::mt19937_64& rng, std::uint64_t tag) -> store::Stored {
	auto const tokens = std::uniform_int_distribution<std::size_t>{0, 3000}(rng);
	auto sequence = std::vector<std::uint32_t>(tokens);
	std::ranges::generate(sequence, [&rng] { return static_cast<std::uint32_t>(rng() % 248'077); });
	auto text = std::string(std::uniform_int_distribution<std::size_t>{0, 2000}(rng), 'x');
	return store::Stored{
	    .response =
	        json::Response{
	            .id = std::format("resp_{:032x}", tag),
	            .created_at = 1,
	            .completed_at = 2,
	            .status = json::ResponseStatus::Completed,
	            .instructions = std::nullopt,
	            .metadata = {},
	            .model = "m",
	            .output = {json::MessageOutput{.id = "msg_1", .text = std::move(text), .status = json::ItemStatus::Completed}},
	            .tools = {},
	            .tool_choice = json::ToolChoiceMode::Auto,
	            .temperature = 1.0,
	            .top_p = 1.0,
	            .format = json::TextFormatMode::Text,
	            .effort = json::Effort::XHigh,
	            .max_output_tokens = std::nullopt,
	            .previous_response_id = std::nullopt,
	            .storage = json::Storage::Stored,
	            .usage = json::Usage{.input_tokens = 1, .cached_tokens = 0, .output_tokens = 1, .reasoning_tokens = 0},
	        },
	    .input = {json::StoredInput{.id = "msg_0", .item = json::MessageItem{.role = json::Role::User, .text = "q"}}},
	    .sequence = std::move(sequence),
	};
}

/** The oracle: a plain list in recency order, oldest first, evicting from the front exactly as the bounds say. */
struct Model {
	std::vector<std::pair<store::ResponseId, store::Stored>> entries;

	[[nodiscard]] auto bytes() const -> std::size_t {
		auto total = std::size_t{0};
		for (auto const& entry : entries) {
			total += store::footprint(entry.second);
		}
		return total;
	}

	auto insert(store::ResponseId id, store::Stored value) -> void {
		auto const size = store::footprint(value);
		while (entries.size() == bounds.responses || bytes() + size > bounds.bytes) {
			entries.erase(entries.begin());
		}
		entries.emplace_back(id, std::move(value));
	}

	[[nodiscard]] auto find(store::ResponseId id) -> std::optional<store::Stored> {
		auto const found = std::ranges::find(entries, id, &std::pair<store::ResponseId, store::Stored>::first);
		if (found == entries.end()) {
			return std::nullopt;
		}
		auto entry = std::move(*found);
		entries.erase(found);
		entries.push_back(std::move(entry));
		return entries.back().second;
	}

	[[nodiscard]] auto remove(store::ResponseId id) -> store::Removal {
		auto const found = std::ranges::find(entries, id, &std::pair<store::ResponseId, store::Stored>::first);
		if (found == entries.end()) {
			return store::Removal::Absent;
		}
		entries.erase(found);
		return store::Removal::Removed;
	}
};

/** Random inserts, reads and deletes: the store agrees with the oracle at every step, and stays within its bounds. */
[[nodiscard]] auto agrees_with_the_model(std::uint64_t seed) -> bool {
	auto rng = engine::testing::generator(seed);
	auto held = Store{};
	auto model = Model{};
	auto issued = std::vector<store::ResponseId>{};
	for (auto step = 0; step < 4000; ++step) {
		auto const roll = rng() % 10;
		if (roll < 4 || issued.empty()) {
			auto const id = store::ResponseId{.high = rng(), .low = static_cast<std::uint64_t>(step)};
			auto value = stored(rng, id.low);
			model.insert(id, value);
			held.insert(id, std::move(value));
			issued.push_back(id);
		} else {
			auto const id = issued[rng() % issued.size()];
			if (roll < 8) {
				auto const expected = model.find(id);
				auto const found = held.find(id);
				if (found.has_value() != expected.has_value() || (found && *found != *expected)) {
					std::println("step {}: find disagrees", step);
					return false;
				}
			} else if (held.remove(id) != model.remove(id)) {
				std::println("step {}: remove disagrees", step);
				return false;
			}
		}
		if (held.size() != model.entries.size() || held.bytes() != model.bytes() || held.size() > bounds.responses ||
		    held.bytes() > bounds.bytes) {
			std::println("step {}: size or bytes disagree", step);
			return false;
		}
	}
	return true;
}

/** A parent read by every one of its branches stays while other responses come and go. */
[[nodiscard]] auto a_parent_read_by_its_branches_stays() -> bool {
	auto rng = engine::testing::generator(7);
	auto held = Store{};
	auto const parent = store::ResponseId{.high = 1, .low = 1};
	held.insert(parent, stored(rng, 1));
	for (auto branch = std::uint64_t{2}; branch < 200; ++branch) {
		if (!held.find(parent)) {
			return false;
		}
		held.insert(store::ResponseId{.high = 2, .low = branch}, stored(rng, branch));
	}
	return held.find(parent).has_value();
}

/** Ids round-trip through their spelling, and nothing else parses as one. */
[[nodiscard]] auto ids_round_trip(std::uint64_t seed) -> bool {
	auto rng = engine::testing::generator(seed);
	for (auto round = 0; round < 10'000; ++round) {
		auto const id = store::ResponseId{.high = rng(), .low = rng()};
		if (store::ResponseId::parse(id.text()) != id || id.text().size() != 37) {
			return false;
		}
	}
	auto const refused = std::array<std::string_view, 7>{
	    "", "resp_", "resp_0123456789abcdef0123456789abcde", "resp_0123456789abcdef0123456789abcdef0",
	    "resp_0123456789ABCDEF0123456789abcdef", "rsp_0123456789abcdef0123456789abcdef0", "resp_0123456789abcdef0123456789abcdeg",
	};
	return std::ranges::none_of(refused, [](std::string_view text) { return store::ResponseId::parse(text).has_value(); });
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the store agrees with a list oracle under random use (seed 1)", .holds = agrees_with_the_model(1)},
	    engine::testing::Case{.name = "the store agrees with a list oracle under random use (seed 2)", .holds = agrees_with_the_model(2)},
	    engine::testing::Case{.name = "a parent read by each of its branches is not evicted under them", .holds = a_parent_read_by_its_branches_stays()},
	    engine::testing::Case{.name = "response ids round-trip through their spelling, and nothing else parses (seed 1)", .holds = ids_round_trip(1)},
	};
	return engine::testing::report(cases);
}
