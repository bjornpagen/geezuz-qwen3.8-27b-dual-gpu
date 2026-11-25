// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.json_model;
import engine.testing;

namespace {

namespace json = engine::json;
namespace model = engine::json_model;

using json::ParseFault;

template<json::Bounds B>
[[nodiscard]] auto make_reader() -> std::unique_ptr<json::Reader<B>> {
	auto reader = json::Reader<B>::make();
	engine::core::invariant(reader.has_value());
	return std::make_unique<json::Reader<B>>(std::move(*reader));
}

/** Bounds small enough that generated documents cross every one of them often. */
auto constexpr small = json::Bounds{.document_bytes = 160, .depth = 3, .string_bytes = 6, .members = 4, .number_chars = 8};

/**
 * Under small bounds, a generated document reads whole exactly when it is within every bound; past one bound it
 * fails with that bound's fault; past several, with one of theirs. The oracle measures the model, not the text the
 * reader saw.
 */
[[nodiscard]] auto bounds_decide_reads(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader<small>();
	auto exceeded_each = std::array<std::size_t, 5>{};
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto const shape = model::Shape{.depth = rng() % 5, .width = 1 + rng() % 6, .text = rng() % 8};
		auto const value = model::generate(rng, shape);
		auto const style = model::Style{
		    .spelling = (rng() % 2) == 0 ? model::Spelling::Raw : model::Spelling::Escaped,
		    .spacing = (rng() % 2) == 0 ? model::Spacing::Compact : model::Spacing::Random,
		};
		auto const text = model::serialize(value, style, rng);
		auto const measured = model::measure(value);
		auto exceeded = std::vector<ParseFault>{};
		auto const note = [&](bool past, ParseFault fault, std::size_t index) {
			if (past) {
				exceeded.push_back(fault);
				++exceeded_each[index];
			}
		};
		note(text.size() > small.document_bytes, ParseFault::TooLarge, 0);
		note(measured.depth > small.depth, ParseFault::TooDeep, 1);
		note(measured.string_bytes > small.string_bytes, ParseFault::StringTooLong, 2);
		note(measured.members > small.members, ParseFault::TooManyMembers, 3);
		note(measured.number_chars > small.number_chars, ParseFault::NumberTooLong, 4);
		auto const read = model::read_model(*reader, text);
		if (exceeded.empty()) {
			if (!read || !model::same(value, *read)) {
				std::println("within bounds but not read back: {:?}", text);
				return false;
			}
			continue;
		}
		auto const fault = read ? std::nullopt : model::fault_of(read.error());
		if (!fault || std::ranges::find(exceeded, *fault) == exceeded.end()) {
			std::println("past {} bounds but read gave {}: {:?}", exceeded.size(), fault ? json::describe(*fault) : "success", text);
			return false;
		}
	}
	// The search is only evidence if it crossed every bound.
	return std::ranges::all_of(exceeded_each, [](std::size_t times) { return times > 0; });
}

auto constexpr bounds = json::request_bounds;

[[nodiscard]] auto reads_whole(json::Reader<bounds>& reader, std::string_view text) -> bool {
	return model::read_model(reader, text).has_value();
}

[[nodiscard]] auto fails_with(json::Reader<bounds>& reader, std::string_view text, ParseFault fault) -> bool {
	auto const read = model::read_model(reader, text);
	return !read && model::fault_of(read.error()) == fault;
}

[[nodiscard]] auto document_bound() -> bool {
	auto reader = make_reader<bounds>();
	auto const padded = [](std::size_t length) { return "[" + std::string(length - 2, ' ') + "]"; };
	return reads_whole(*reader, padded(bounds.document_bytes)) && fails_with(*reader, padded(bounds.document_bytes + 1), ParseFault::TooLarge);
}

[[nodiscard]] auto depth_bound() -> bool {
	auto reader = make_reader<bounds>();
	auto const nested = [](std::size_t depth, std::string_view open, std::string_view close) {
		auto text = std::string{};
		for (auto level = std::size_t{0}; level < depth; ++level) {
			text += open;
		}
		text += '0';
		for (auto level = std::size_t{0}; level < depth; ++level) {
			text += close;
		}
		return text;
	};
	return reads_whole(*reader, nested(bounds.depth, "[", "]")) && fails_with(*reader, nested(bounds.depth + 1, "[", "]"), ParseFault::TooDeep) &&
	       reads_whole(*reader, nested(bounds.depth, "{\"k\":", "}")) && fails_with(*reader, nested(bounds.depth + 1, "{\"k\":", "}"), ParseFault::TooDeep);
}

[[nodiscard]] auto string_bound() -> bool {
	auto reader = make_reader<bounds>();
	auto const quoted = [](std::size_t length) { return "\"" + std::string(length, 'x') + "\""; };
	// An escape is counted by the bytes it unescapes to, not by its spelling.
	auto const escaped = [](std::size_t length) {
		auto text = std::string{"[\""};
		for (auto index = std::size_t{0}; index < length / 2; ++index) {
			text += "\\u00e9";
		}
		return text + "\"]";
	};
	auto const named = [](std::size_t length) { return "{\"" + std::string(length, 'n') + "\":1}"; };
	return reads_whole(*reader, quoted(bounds.string_bytes)) && fails_with(*reader, quoted(bounds.string_bytes + 1), ParseFault::StringTooLong) &&
	       reads_whole(*reader, escaped(bounds.string_bytes)) && fails_with(*reader, escaped(bounds.string_bytes + 2), ParseFault::StringTooLong) &&
	       reads_whole(*reader, named(bounds.string_bytes)) && fails_with(*reader, named(bounds.string_bytes + 1), ParseFault::StringTooLong);
}

[[nodiscard]] auto member_bound() -> bool {
	auto reader = make_reader<bounds>();
	auto const array = [](std::size_t count) {
		auto text = std::string{"["};
		for (auto index = std::size_t{0}; index < count; ++index) {
			text += index == 0 ? "0" : ",0";
		}
		return text + "]";
	};
	auto const object = [](std::size_t count) {
		auto text = std::string{"{"};
		for (auto index = std::size_t{0}; index < count; ++index) {
			text += std::format("{}\"k{}\":0", index == 0 ? "" : ",", index);
		}
		return text + "}";
	};
	auto const counted = [&reader](std::string const& text) { return reader->read_copy(text)->as_array()->size(); };
	auto const over = counted(array(bounds.members + 1));
	return reads_whole(*reader, array(bounds.members)) && fails_with(*reader, array(bounds.members + 1), ParseFault::TooManyMembers) &&
	       reads_whole(*reader, object(bounds.members)) && fails_with(*reader, object(bounds.members + 1), ParseFault::TooManyMembers) &&
	       counted(array(bounds.members)) == bounds.members && !over && model::fault_of(over.error()) == ParseFault::TooManyMembers;
}

[[nodiscard]] auto number_bound() -> bool {
	auto reader = make_reader<bounds>();
	auto const integer = [](std::size_t length) { return "[1" + std::string(length - 1, '0') + "]"; };
	auto const fraction = [](std::size_t length) { return "[0." + std::string(length - 2, '5') + "]"; };
	return reads_whole(*reader, integer(bounds.number_chars)) && fails_with(*reader, integer(bounds.number_chars + 1), ParseFault::NumberTooLong) &&
	       reads_whole(*reader, fraction(bounds.number_chars)) && fails_with(*reader, fraction(bounds.number_chars + 1), ParseFault::NumberTooLong);
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "small bounds decide every generated read, each with its fault (seed 1)", .holds = bounds_decide_reads(1, 30000)},
	    engine::testing::Case{.name = "small bounds decide every generated read, each with its fault (seed 2)", .holds = bounds_decide_reads(2, 30000)},
	    engine::testing::Case{.name = "a request at its byte bound reads; one byte past is too large", .holds = document_bound()},
	    engine::testing::Case{.name = "a request at its depth bound reads; one level past is too deep", .holds = depth_bound()},
	    engine::testing::Case{.name = "a string or name at its byte bound reads; one byte past is too long", .holds = string_bound()},
	    engine::testing::Case{.name = "an array or object at its member bound reads; one past is too many", .holds = member_bound()},
	    engine::testing::Case{.name = "a number at its literal bound reads; one character past is too long", .holds = number_bound()},
	};
	return engine::testing::report(cases);
}
