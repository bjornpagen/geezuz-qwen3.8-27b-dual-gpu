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

auto constexpr bounds = json::request_bounds;
using Reader = json::Reader<bounds>;

[[nodiscard]] auto make_reader() -> std::unique_ptr<Reader> {
	auto reader = Reader::make();
	engine::core::invariant(reader.has_value());
	return std::make_unique<Reader>(std::move(*reader));
}

[[nodiscard]] auto random_style(std::mt19937_64& rng) -> model::Style {
	return model::Style{
	    .spelling = (rng() % 2) == 0 ? model::Spelling::Raw : model::Spelling::Escaped,
	    .spacing = (rng() % 2) == 0 ? model::Spacing::Compact : model::Spacing::Random,
	};
}

auto constexpr compact = model::Style{.spelling = model::Spelling::Raw, .spacing = model::Spacing::Compact};

/** Every generated document, in every spelling and spacing, reads back as exactly its model. */
[[nodiscard]] auto round_trips(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto const value = model::generate(rng, model::random_shape(rng));
		auto const text = model::serialize(value, random_style(rng), rng);
		auto const read = model::read_model(*reader, text);
		if (!read || !model::same(value, *read)) {
			std::println("round trip failed for {:?}", text);
			return false;
		}
	}
	return true;
}

/** Every proper prefix of a container document fails to read whole. */
[[nodiscard]] auto prefixes_fail(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto const value = model::generate_container(rng, model::random_shape(rng));
		auto text = model::serialize(value, random_style(rng), rng);
		// Whitespace after the root is not part of it: a prefix that drops only that is the whole document.
		text.erase(text.find_last_not_of(" \t\n\r") + 1);
		for (auto length = std::size_t{0}; length < text.size(); ++length) {
			if (model::read_model(*reader, text.substr(0, length)).has_value()) {
				std::println("prefix {} of {:?} read whole", length, text);
				return false;
			}
		}
	}
	return true;
}

enum class Mutation : std::uint8_t {
	Flip,
	Insert,
	Erase,
	Truncate,
};

[[nodiscard]] auto mutate(std::string text, std::mt19937_64& rng) -> std::string {
	auto constexpr interesting = std::string_view{"{}[],:\"\\-+.0123456789eEtfnu \x7F\x80\xBF\xC0\xED\xF4\xFF"};
	for (auto count = 1 + rng() % 4; count > 0; --count) {
		auto const at = text.empty() ? 0 : rng() % text.size();
		auto const byte = (rng() % 2) == 0 ? interesting[rng() % interesting.size()] : std::bit_cast<char>(static_cast<std::uint8_t>(rng()));
		switch (static_cast<Mutation>(rng() % 4)) {
		case Mutation::Flip:
			if (!text.empty()) {
				text[at] = byte;
			}
			break;
		case Mutation::Insert:
			text.insert(at, 1, byte);
			break;
		case Mutation::Erase:
			if (!text.empty()) {
				text.erase(at, 1);
			}
			break;
		case Mutation::Truncate:
			text.resize(at);
			break;
		}
	}
	return text;
}

/**
 * A mutated document never crashes the reader (the sanitizers are the oracle for that): it fails with a named
 * parse fault, or it reads whole to a model that, written out again, reads back as itself.
 */
[[nodiscard]] auto mutations_fail_cleanly(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto const value = model::generate(rng, model::random_shape(rng));
		auto const text = mutate(model::serialize(value, random_style(rng), rng), rng);
		auto const read = model::read_model(*reader, text);
		if (!read) {
			if (!model::fault_of(read.error())) {
				return false;
			}
			continue;
		}
		auto const again = model::read_model(*reader, model::serialize(*read, compact, rng));
		if (!again || !model::same(*read, *again)) {
			return false;
		}
	}
	return true;
}

/** Locates every closing bracket of a document, by walking its model alongside its compact text. */
auto closers_of(model::Model const& value, std::size_t at, std::vector<std::size_t>& closers, std::mt19937_64& rng) -> std::size_t {
	auto const length = model::serialize(value, compact, rng).size();
	if (std::holds_alternative<model::Elements>(value.value)) {
		auto inner = at + 1;
		for (auto const& element : std::get<model::Elements>(value.value)) {
			inner = closers_of(element, inner, closers, rng) + 1;
		}
		closers.push_back(at + length - 1);
	} else if (std::holds_alternative<model::Members>(value.value)) {
		auto inner = at + 1;
		for (auto const& [name, member] : std::get<model::Members>(value.value)) {
			auto key = std::string{};
			model::put_string(key, name, model::Spelling::Raw);
			inner = closers_of(member, inner + key.size() + 1, closers, rng) + 1;
		}
		closers.push_back(at + length - 1);
	}
	return at + length;
}

/** A comma before any closing bracket is refused as malformed structure. */
[[nodiscard]] auto trailing_commas_refused(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto const value = model::generate_container(rng, model::random_shape(rng));
		auto const text = model::serialize(value, compact, rng);
		auto closers = std::vector<std::size_t>{};
		std::ignore = closers_of(value, 0, closers, rng);
		auto broken = text;
		broken.insert(closers[rng() % closers.size()], ",");
		auto const read = model::read_model(*reader, broken);
		if (read || model::fault_of(read.error()) != ParseFault::Malformed) {
			std::println("trailing comma not refused as malformed: {:?}", broken);
			return false;
		}
	}
	return true;
}

/** Naming any member of an object a second time is refused. */
[[nodiscard]] auto duplicate_names_refused(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto members = model::Members{};
		for (auto size = 1 + rng() % 6; size > 0; --size) {
			members.emplace_back(model::random_text(rng, 3) + std::to_string(size), model::generate(rng, model::random_shape(rng)));
		}
		auto const original = rng() % members.size();
		auto repeated = members[original];
		repeated.second = model::generate(rng, model::random_shape(rng));
		members.insert(members.begin() + static_cast<std::ptrdiff_t>(rng() % (members.size() + 1)), std::move(repeated));
		auto const nested = (rng() % 2) == 0;
		auto const document = nested ? model::Model{.value = model::Elements{model::Model{.value = std::move(members)}}} : model::Model{.value = std::move(members)};
		auto const read = model::read_model(*reader, model::serialize(document, random_style(rng), rng));
		if (read || model::fault_of(read.error()) != ParseFault::DuplicateKey) {
			return false;
		}
	}
	return true;
}

/**
 * Anything but whitespace after a complete root is trailing content, whatever it is: a word, a number, a bracket, a
 * second document, an unclosed string.
 */
[[nodiscard]] auto trailing_content_named(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	auto constexpr junk = std::array<std::string_view, 12>{"x", "1", "]", "}", "[", "{", "{}", "[1]", "\"", "\"s\"", "tru", ","};
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto const value = model::generate(rng, model::random_shape(rng));
		auto const text = model::serialize(value, random_style(rng), rng) + " " + std::string{junk[rng() % junk.size()]};
		auto const read = model::read_model(*reader, text);
		if (read || model::fault_of(read.error()) != ParseFault::TrailingContent) {
			std::println("not trailing content: {:?}", text);
			return false;
		}
	}
	return true;
}

/** The grammar's refusals, each with its named fault: one representative per rule RFC 8259 states. */
[[nodiscard]] auto strict_grammar() -> bool {
	struct Refusal {
		std::string_view text;
		ParseFault fault;
	};
	auto constexpr refusals = std::array{
	    Refusal{.text = "", .fault = ParseFault::Empty},
	    Refusal{.text = "  ", .fault = ParseFault::Empty},
	    Refusal{.text = "[1,]", .fault = ParseFault::Malformed},
	    Refusal{.text = R"({"a":1,})", .fault = ParseFault::Malformed},
	    Refusal{.text = "[1 2]", .fault = ParseFault::Malformed},
	    Refusal{.text = "{a:1}", .fault = ParseFault::Malformed},
	    Refusal{.text = "{\"a\" 1}", .fault = ParseFault::Malformed},
	    Refusal{.text = "[Infinity]", .fault = ParseFault::Malformed},
	    Refusal{.text = "[1/*c*/]", .fault = ParseFault::InvalidNumber},
	    Refusal{.text = "[-Infinity]", .fault = ParseFault::InvalidNumber},
	    Refusal{.text = "01", .fault = ParseFault::InvalidNumber},
	    Refusal{.text = "[1.]", .fault = ParseFault::InvalidNumber},
	    Refusal{.text = "[-]", .fault = ParseFault::InvalidNumber},
	    Refusal{.text = "1e400", .fault = ParseFault::InvalidNumber},
	    Refusal{.text = "[tru]", .fault = ParseFault::InvalidLiteral},
	    Refusal{.text = "[fals]", .fault = ParseFault::InvalidLiteral},
	    Refusal{.text = "1 2", .fault = ParseFault::TrailingContent},
	    Refusal{.text = "{}}", .fault = ParseFault::TrailingContent},
	    Refusal{.text = R"({"a":[]} x)", .fault = ParseFault::TrailingContent},
	    Refusal{.text = "[] x", .fault = ParseFault::TrailingContent},
	    Refusal{.text = "[1] 2", .fault = ParseFault::TrailingContent},
	    Refusal{.text = R"({} ")", .fault = ParseFault::TrailingContent},
	    Refusal{.text = R"({"a":"]"} ])", .fault = ParseFault::TrailingContent},
	    Refusal{.text = R"({"a":"\"}"} [)", .fault = ParseFault::TrailingContent},
	    Refusal{.text = R"({"a":"}")", .fault = ParseFault::Truncated},
	    Refusal{.text = "[1", .fault = ParseFault::Truncated},
	    Refusal{.text = "{\"a\":", .fault = ParseFault::Truncated},
	    Refusal{.text = "\"abc", .fault = ParseFault::Truncated},
	    Refusal{.text = "\"a\tb\"", .fault = ParseFault::ControlCharacter},
	    Refusal{.text = R"("\x41")", .fault = ParseFault::InvalidString},
	    Refusal{.text = R"("\u12G4")", .fault = ParseFault::InvalidString},
	    Refusal{.text = R"("\uD800")", .fault = ParseFault::InvalidString},
	    Refusal{.text = R"("\uDC00\uD800")", .fault = ParseFault::InvalidString},
	    Refusal{.text = R"("\uD83D\n")", .fault = ParseFault::InvalidString},
	    Refusal{.text = R"("\uD83D\u0041")", .fault = ParseFault::InvalidString},
	    Refusal{.text = "\"\xC0\x80\"", .fault = ParseFault::InvalidUtf8},
	    Refusal{.text = "\"\xED\xA0\x80\"", .fault = ParseFault::InvalidUtf8},
	    Refusal{.text = "\"\xF4\x90\x80\x80\"", .fault = ParseFault::InvalidUtf8},
	    Refusal{.text = R"({"a":1,"a":1})", .fault = ParseFault::DuplicateKey},
	    Refusal{.text = R"({"\u0061":1,"a":2})", .fault = ParseFault::DuplicateKey},
	};
	auto reader = make_reader();
	return std::ranges::all_of(refusals, [&reader](Refusal const& refusal) {
		auto const read = model::read_model(*reader, refusal.text);
		auto const holds = !read && model::fault_of(read.error()) == refusal.fault;
		if (!holds) {
			std::println("not refused as {}: {:?} ({})", json::describe(refusal.fault), refusal.text, read ? "read" : json::describe(model::fault_of(read.error()).value_or(ParseFault::Empty)));
		}
		return holds;
	});
}

/** Words that are not JSON fail; which structural fault simdjson names for each is its choice, not the contract. */
[[nodiscard]] auto non_json_words_refused() -> bool {
	auto constexpr words = std::array<std::string_view, 9>{"NaN", "nan", "nul", "[nulL]", "True", "'a'", "// c\n1", ".5", "\x01"};
	auto reader = make_reader();
	return std::ranges::none_of(words, [&reader](std::string_view word) { return model::read_model(*reader, word).has_value(); });
}

/** Names unescape before they are compared, and distinct names that look alike stay distinct. */
[[nodiscard]] auto names_compare_unescaped() -> bool {
	auto reader = make_reader();
	auto const read = model::read_model(*reader, R"({"a":1,"A":2,"a ":3,"\u00e9":4,"é ":5})");
	return read.has_value() && std::get<model::Members>(read->value).size() == 5;
}

/** Each typed accessor succeeds exactly on its own kind, and names both kinds when it does not. */
[[nodiscard]] auto accessors_match_kinds() -> bool {
	struct Sample {
		std::string_view text;
		json::Kind kind;
	};
	auto constexpr samples = std::array{
	    Sample{.text = "null", .kind = json::Kind::Null},     Sample{.text = "true", .kind = json::Kind::Boolean},
	    Sample{.text = "-7", .kind = json::Kind::Integer},    Sample{.text = "2.5", .kind = json::Kind::Real},
	    Sample{.text = "\"s\"", .kind = json::Kind::String},  Sample{.text = "[1]", .kind = json::Kind::Array},
	    Sample{.text = "{\"k\":1}", .kind = json::Kind::Object},
	};
	auto reader = make_reader();
	auto holds = true;
	for (auto const& sample : samples) {
		auto const agrees = [&](auto read, json::Kind expected, bool succeeds) {
			auto const result = read(*reader->read_copy(sample.text));
			if (succeeds) {
				return result.has_value();
			}
			return !result.has_value() && result.error() == json::ReadError{.cause = json::KindMismatch{.expected = expected, .found = sample.kind}};
		};
		auto const kind = reader->read_copy(sample.text)->kind();
		holds = holds && kind == sample.kind;
		holds = holds && agrees([](auto value) { return value.as_null(); }, json::Kind::Null, sample.kind == json::Kind::Null);
		holds = holds && agrees([](auto value) { return value.as_boolean(); }, json::Kind::Boolean, sample.kind == json::Kind::Boolean);
		holds = holds && agrees([](auto value) { return value.as_integer(); }, json::Kind::Integer, sample.kind == json::Kind::Integer);
		holds = holds && agrees([](auto value) { return value.as_real(); }, json::Kind::Real, sample.kind == json::Kind::Real || sample.kind == json::Kind::Integer);
		holds = holds && agrees([](auto value) { return value.as_string(); }, json::Kind::String, sample.kind == json::Kind::String);
		holds = holds && agrees([](auto value) { return value.as_array(); }, json::Kind::Array, sample.kind == json::Kind::Array);
		holds = holds && agrees([](auto value) { return value.as_object(); }, json::Kind::Object, sample.kind == json::Kind::Object);
	}
	return holds;
}

/** An integer reads as a real exactly when the double format holds it, and then as that double. */
[[nodiscard]] auto integers_read_as_reals_exactly(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const integer = std::bit_cast<std::int64_t>(rng()) / static_cast<std::int64_t>(std::uint64_t{1} << (rng() % 63));
		auto const real = reader->read_copy(std::to_string(integer))->as_real();
		// The oracle: the integer is exact in a double when its odd part fits the 53-bit significand.
		auto const magnitude = integer == std::numeric_limits<std::int64_t>::min() ? std::uint64_t{1} << 63U : static_cast<std::uint64_t>(integer < 0 ? -integer : integer);
		auto const exact = magnitude == 0 || std::bit_width(magnitude >> static_cast<unsigned>(std::countr_zero(magnitude))) <= 53;
		if (real.has_value() != exact || (exact && *real != static_cast<double>(integer))) {
			return false;
		}
	}
	return true;
}

/** Lookup by name finds every member in any order, and a required absent member is named in the error. */
[[nodiscard]] auto members_found_by_name(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto members = model::Members{};
		for (auto size = rng() % 8; size > 0; --size) {
			auto name = model::random_text(rng, 3);
			if (std::ranges::none_of(members, [&name](auto const& member) { return member.first == name; })) {
				members.emplace_back(std::move(name), model::generate(rng, model::random_shape(rng)));
			}
		}
		auto const text = model::serialize(model::Model{.value = members}, random_style(rng), rng);
		auto order = members;
		std::ranges::shuffle(order, rng);
		auto object = reader->read_copy(text)->as_object();
		for (auto const& [name, value] : order) {
			auto found = object->member(name);
			if (!found) {
				std::println("lookup of {:?} failed in {:?}", name, text);
				return false;
			}
			auto const read = model::model_of(std::move(*found));
			if (!read || !model::same(value, *read)) {
				std::println("lookup of {:?} read wrong in {:?}: {}", name, text, read.has_value());
				return false;
			}
		}
		auto absent = model::random_text(rng, 3);
		while (std::ranges::any_of(members, [&absent](auto const& member) { return member.first == absent; })) {
			absent += '\x01';
		}
		auto const missing = object->member(absent);
		if (missing || missing.error() != json::ReadError{.cause = json::MissingMember{.name = absent}}) {
			std::println("absent lookup of {:?} in {:?}: found {} fault {}", absent, text, missing.has_value(), missing ? "" : json::describe(model::fault_of(missing.error()).value_or(ParseFault::Empty)));
			return false;
		}
	}
	return true;
}

/** An array's size, counted ahead, is its element count. */
[[nodiscard]] auto array_sizes_count_elements(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto elements = model::Elements{};
		for (auto size = rng() % 12; size > 0; --size) {
			elements.push_back(model::generate(rng, model::random_shape(rng)));
		}
		auto array = reader->read_copy(model::serialize(model::Model{.value = elements}, random_style(rng), rng))->as_array();
		if (array->size() != elements.size()) {
			return false;
		}
	}
	return true;
}

/** A text read in place, from storage padded by the caller, reads exactly as a copied one. */
[[nodiscard]] auto in_place_reads_agree(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto const value = model::generate(rng, model::random_shape(rng));
		auto storage = model::serialize(value, random_style(rng), rng);
		auto const length = storage.size();
		// Whatever the padding holds must not change the result.
		for (auto pad = std::size_t{0}; pad < json::padding; ++pad) {
			storage.push_back("]}\"0 x"[rng() % 6]);
		}
		auto root = reader->read(json::PaddedView::over(storage, length));
		auto const read = model::model_of(std::move(*root));
		if (!read || !model::same(value, *read) || !reader->finish()) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "generated documents read back exactly (seed 1)", .holds = round_trips(1, 4000)},
	    engine::testing::Case{.name = "generated documents read back exactly (seed 2)", .holds = round_trips(2, 4000)},
	    engine::testing::Case{.name = "every proper prefix of a container fails to read (seed 3)", .holds = prefixes_fail(3, 300)},
	    engine::testing::Case{.name = "mutated documents fail by name or read back as themselves (seed 4)", .holds = mutations_fail_cleanly(4, 20000)},
	    engine::testing::Case{.name = "mutated documents fail by name or read back as themselves (seed 5)", .holds = mutations_fail_cleanly(5, 20000)},
	    engine::testing::Case{.name = "a trailing comma before any closer is malformed (seed 6)", .holds = trailing_commas_refused(6, 2000)},
	    engine::testing::Case{.name = "a repeated member name is refused (seed 7)", .holds = duplicate_names_refused(7, 2000)},
	    engine::testing::Case{.name = "each grammar rule's refusal names its fault", .holds = strict_grammar()},
	    engine::testing::Case{.name = "anything after a complete root is trailing content (seed 12)", .holds = trailing_content_named(12, 20000)},
	    engine::testing::Case{.name = "words that are not JSON fail", .holds = non_json_words_refused()},
	    engine::testing::Case{.name = "member names compare after unescaping", .holds = names_compare_unescaped()},
	    engine::testing::Case{.name = "typed accessors succeed exactly on their kind", .holds = accessors_match_kinds()},
	    engine::testing::Case{.name = "an integer reads as a real exactly when a double holds it (seed 8)", .holds = integers_read_as_reals_exactly(8, 100000)},
	    engine::testing::Case{.name = "members are found by name in any order and a missing one is named (seed 9)", .holds = members_found_by_name(9, 2000)},
	    engine::testing::Case{.name = "an array's size counts its elements (seed 10)", .holds = array_sizes_count_elements(10, 2000)},
	    engine::testing::Case{.name = "a padded in-place read agrees with a copied one (seed 11)", .holds = in_place_reads_agree(11, 2000)},
	};
	return engine::testing::report(cases);
}
