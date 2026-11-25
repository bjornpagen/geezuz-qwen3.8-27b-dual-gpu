// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar_support;

import std;
import engine.core;
import engine.grammar;
import engine.json;
import engine.testing;
import engine.tokenizer;

namespace engine::grammar_support {

namespace grammar = engine::grammar;

/** The end token of the byte vocabulary. */
export inline constexpr grammar::Token byte_end = 256;

export inline constexpr grammar::Layout byte_layout{.tokens = 257, .rows = 257};

/** A vocabulary of the 256 single bytes and one end token: a matcher over it accepts or refuses text byte by byte. */
export [[nodiscard]] auto byte_vocabulary() -> grammar::Vocabulary {
	auto tokens = std::vector<grammar::TokenBytes>{};
	for (auto byte = 0U; byte < 256U; ++byte) {
		tokens.push_back(grammar::TokenBytes{.token = byte, .bytes = std::string(1, static_cast<char>(static_cast<std::uint8_t>(byte))), .row = byte});
	}
	auto const ends = std::array{grammar::EndToken{.token = byte_end, .row = byte_end}};
	return grammar::Vocabulary::make(tokens, ends, byte_layout);
}

/** How far a grammar takes a text. */
export enum class Verdict : std::uint8_t {
	/** Every byte is allowed and the output may end after them. */
	Accepted,
	/** Every byte is allowed but the output may not end there. */
	Prefix,
	/** Some byte is refused. */
	Refused,
};

export [[nodiscard]] auto judge(grammar::Grammar grammar, grammar::Vocabulary const& bytes, std::string_view text) -> Verdict {
	auto matcher = grammar::Matcher::make(std::move(grammar), byte_layout);
	for (auto const letter : text) {
		if (!matcher.advance(bytes, std::bit_cast<std::uint8_t>(letter))) {
			return Verdict::Refused;
		}
	}
	return matcher.accepting() ? Verdict::Accepted : Verdict::Prefix;
}

/**
 * llguidance's test form: each accepted text is accepted; each rejected text is refused at some byte, or, written
 * with a `FINAL_REJECT:` prefix, taken whole but not accepted at its end.
 */
export [[nodiscard]] auto agrees(grammar::Grammar const& grammar, grammar::Vocabulary const& bytes, std::span<std::string_view const> accepted,
                                 std::span<std::string_view const> rejected) -> bool {
	auto holds = true;
	for (auto const text : accepted) {
		if (judge(grammar, bytes, text) != Verdict::Accepted) {
			std::println("  expected accepted: {:?}", text);
			holds = false;
		}
	}
	constexpr auto final_reject = std::string_view{"FINAL_REJECT:"};
	for (auto const text : rejected) {
		auto const at_end = text.starts_with(final_reject);
		auto const body = at_end ? text.substr(final_reject.size()) : text;
		auto const verdict = judge(grammar, bytes, body);
		if (verdict != (at_end ? Verdict::Prefix : Verdict::Refused)) {
			std::println("  expected {}: {:?}", at_end ? "prefix only" : "refused", body);
			holds = false;
		}
	}
	return holds;
}

/** A JSON value as the oracle reads it: numbers as doubles, members in order. */
export struct Json;

export struct JsonMember;

export struct Json {
	std::variant<std::monostate, bool, double, std::string, std::vector<Json>, std::vector<JsonMember>> held;

	[[nodiscard]] auto find(std::string_view name) const -> std::optional<Json const&>;
};

export struct JsonMember {
	std::string name;
	Json value;
};

auto Json::find(std::string_view name) const -> std::optional<Json const&> {
	auto const members = engine::core::alternative<std::vector<JsonMember>>(held);
	if (!members) {
		return std::nullopt;
	}
	auto const found = std::ranges::find(*members, name, &JsonMember::name);
	if (found == members->end()) {
		return std::nullopt;
	}
	return found->value;
}

inline constexpr engine::json::Bounds oracle_bounds{
    .document_bytes = std::size_t{1} << 22U, .depth = 64, .string_bytes = std::size_t{1} << 22U, .members = std::size_t{1} << 16U, .number_chars = 128};

[[nodiscard]] auto read_json(engine::json::Value<oracle_bounds> value) -> std::optional<Json> {
	auto const kind = value.kind();
	if (!kind) {
		return std::nullopt;
	}
	switch (*kind) {
	case engine::json::Kind::Null:
		return value.as_null() ? std::optional<Json>{Json{.held = std::monostate{}}} : std::nullopt;
	case engine::json::Kind::Boolean: {
		auto const read = value.as_boolean();
		return read ? std::optional<Json>{Json{.held = *read}} : std::nullopt;
	}
	case engine::json::Kind::Integer:
	case engine::json::Kind::Real: {
		auto const literal = value.raw();
		if (!literal) {
			return std::nullopt;
		}
		auto number = 0.0;
		auto const read = std::from_chars(literal->data(), std::to_address(literal->end()), number);
		if (read.ptr != std::to_address(literal->end())) {
			return std::nullopt;
		}
		return Json{.held = number};
	}
	case engine::json::Kind::String: {
		auto const read = value.as_string();
		return read ? std::optional<Json>{Json{.held = std::string{*read}}} : std::nullopt;
	}
	case engine::json::Kind::Array: {
		auto array = value.as_array();
		if (!array) {
			return std::nullopt;
		}
		auto out = std::vector<Json>{};
		while (true) {
			auto element = array->next();
			if (!element) {
				return std::nullopt;
			}
			if (!*element) {
				return Json{.held = std::move(out)};
			}
			auto read = read_json(std::move(**element));
			if (!read) {
				return std::nullopt;
			}
			out.push_back(std::move(*read));
		}
	}
	case engine::json::Kind::Object: {
		auto object = value.as_object();
		if (!object) {
			return std::nullopt;
		}
		auto out = std::vector<JsonMember>{};
		while (true) {
			auto member = object->next();
			if (!member) {
				return std::nullopt;
			}
			if (!*member) {
				return Json{.held = std::move(out)};
			}
			auto name = std::string{(*member)->name};
			auto read = read_json(std::move((*member)->value));
			if (!read) {
				return std::nullopt;
			}
			out.push_back(JsonMember{.name = std::move(name), .value = std::move(*read)});
		}
	}
	}
	return std::nullopt;
}

/** Reads one whole JSON text strictly (repeated keys and trailing text refused), or nothing. */
export [[nodiscard]] auto parse_json(std::string_view text) -> std::optional<Json> {
	auto reader = engine::json::Reader<oracle_bounds>::make();
	if (!reader) {
		return std::nullopt;
	}
	auto root = reader->read_copy(text);
	if (!root) {
		return std::nullopt;
	}
	auto value = read_json(std::move(*root));
	if (!value || !reader->finish()) {
		return std::nullopt;
	}
	return value;
}

[[nodiscard]] auto equal(Json const& a, Json const& b) -> bool {
	if (a.held.index() != b.held.index()) {
		return false;
	}
	if (auto const array = engine::core::alternative<std::vector<Json>>(a.held)) {
		auto const& other = std::get<std::vector<Json>>(b.held);
		return array->size() == other.size() && std::ranges::equal(*array, other, equal);
	}
	if (auto const members = engine::core::alternative<std::vector<JsonMember>>(a.held)) {
		auto const& other = std::get<std::vector<JsonMember>>(b.held);
		return members->size() == other.size() && std::ranges::all_of(*members, [&b](JsonMember const& member) {
			       auto const found = b.find(member.name);
			       return found && equal(member.value, *found);
		       });
	}
	if (auto const number = engine::core::alternative<double>(a.held)) {
		return *number == std::get<double>(b.held);
	}
	if (auto const text = engine::core::alternative<std::string>(a.held)) {
		return *text == std::get<std::string>(b.held);
	}
	if (auto const flag = engine::core::alternative<bool>(a.held)) {
		return *flag == std::get<bool>(b.held);
	}
	return true;
}

[[nodiscard]] auto code_points(std::string_view text) -> std::size_t {
	return static_cast<std::size_t>(std::ranges::count_if(text, [](char c) { return (std::bit_cast<std::uint8_t>(c) & 0xc0U) != 0x80U; }));
}

[[nodiscard]] auto digits_at(std::string_view text, std::size_t at, std::size_t count) -> std::optional<int> {
	if (at + count > text.size()) {
		return std::nullopt;
	}
	auto value = 0;
	for (auto const c : text.substr(at, count)) {
		if (c < '0' || c > '9') {
			return std::nullopt;
		}
		value = value * 10 + (c - '0');
	}
	return value;
}

[[nodiscard]] auto valid_date(std::string_view text) -> bool {
	auto const year = digits_at(text, 0, 4);
	auto const month = digits_at(text, 5, 2);
	auto const day = digits_at(text, 8, 2);
	if (text.size() != 10 || text[4] != '-' || text[7] != '-' || !year || !month || !day) {
		return false;
	}
	return std::chrono::year_month_day{std::chrono::year{*year}, std::chrono::month{static_cast<unsigned>(*month)}, std::chrono::day{static_cast<unsigned>(*day)}}.ok();
}

[[nodiscard]] auto valid_time(std::string_view text) -> bool {
	auto const hour = digits_at(text, 0, 2);
	auto const minute = digits_at(text, 3, 2);
	auto const second = digits_at(text, 6, 2);
	if (text.size() < 9 || text[2] != ':' || text[5] != ':' || !hour || !minute || !second || *hour > 23 || *minute > 59 || *second > 59) {
		return false;
	}
	auto rest = text.substr(8);
	if (rest.starts_with('.')) {
		auto const digits = rest.find_first_not_of("0123456789", 1);
		if (digits == 1 || digits == std::string_view::npos) {
			return false;
		}
		rest = rest.substr(digits);
	}
	if (rest == "Z") {
		return true;
	}
	auto const offset_hour = digits_at(rest, 1, 2);
	auto const offset_minute = digits_at(rest, 4, 2);
	return rest.size() == 6 && (rest[0] == '+' || rest[0] == '-') && rest[3] == ':' && offset_hour && offset_minute && *offset_hour <= 23 && *offset_minute <= 59;
}

[[nodiscard]] auto valid_format(std::string_view format, std::string_view text) -> bool {
	if (format == "date") {
		return valid_date(text);
	}
	if (format == "time") {
		return valid_time(text);
	}
	if (format == "date-time") {
		auto const t = text.find('T');
		return t == 10 && valid_date(text.substr(0, 10)) && valid_time(text.substr(11));
	}
	if (format == "uuid") {
		return std::regex_match(std::string{text}, std::regex{"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"});
	}
	if (format == "ipv4") {
		auto parts = std::vector<std::string>{};
		auto part = std::string{};
		for (auto const c : text) {
			if (c == '.') {
				parts.push_back(std::exchange(part, {}));
			} else {
				part.push_back(c);
			}
		}
		parts.push_back(part);
		return parts.size() == 4 && std::ranges::all_of(parts, [](std::string const& piece) {
			       return !piece.empty() && piece.size() <= 3 && std::ranges::all_of(piece, [](char c) { return c >= '0' && c <= '9'; }) &&
			              (piece.size() == 1 || piece[0] != '0') && std::stoi(piece) <= 255;
		       });
	}
	if (format == "email") {
		auto const at = text.find('@');
		return at != 0 && at != std::string_view::npos && text.find('@', at + 1) == std::string_view::npos && text.substr(at + 1).contains('.') &&
		       !text.contains("..") && !text.starts_with('.') && !text.substr(0, at).ends_with('.');
	}
	if (format == "hostname") {
		return !text.empty() && text.size() <= 253 && std::regex_match(std::string{text}, std::regex{"[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)*"});
	}
	return false;
}

/**
 * An independent JSON-schema validator for the subset the engine compiles: the oracle the grammar's outputs must
 * satisfy. It reads keywords by their specification, not by the compiler's code: `type` (an integer is a whole
 * number), `enum`, `const`, `anyOf`, `oneOf`, `$ref` to `#` and `#/$defs/...`, the object, array, string and number
 * keywords, `format` by each format's own rules, and `pattern` with `std::regex` (ECMAScript, searched).
 */
export [[nodiscard]] auto validates(Json const& root, Json const& schema, Json const& instance) -> bool {
	if (auto const flag = engine::core::alternative<bool>(schema.held)) {
		return *flag;
	}
	auto const keyword = [&schema](std::string_view name) { return schema.find(name); };
	if (auto const ref = keyword("$ref")) {
		auto const& text = std::get<std::string>(ref->held);
		if (text == "#") {
			return validates(root, root, instance);
		}
		for (auto const prefix : {std::string_view{"#/$defs/"}, std::string_view{"#/definitions/"}}) {
			if (text.starts_with(prefix)) {
				auto const definitions = root.find(prefix.substr(2, prefix.size() - 3));
				auto const target = definitions ? definitions->find(std::string_view{text}.substr(prefix.size())) : std::nullopt;
				return target && validates(root, *target, instance);
			}
		}
		return false;
	}
	auto const nullable = keyword("nullable");
	if (nullable && std::get<bool>(nullable->held) && std::holds_alternative<std::monostate>(instance.held)) {
		return true;
	}
	if (auto const type = keyword("type")) {
		auto const matches = [&instance](Json const& name) {
			auto const& text = std::get<std::string>(name.held);
			auto const number = engine::core::alternative<double>(instance.held);
			return (text == "null" && std::holds_alternative<std::monostate>(instance.held)) || (text == "boolean" && std::holds_alternative<bool>(instance.held)) ||
			       (text == "number" && number.has_value()) || (text == "integer" && number.has_value() && *number == std::trunc(*number)) ||
			       (text == "string" && std::holds_alternative<std::string>(instance.held)) ||
			       (text == "array" && std::holds_alternative<std::vector<Json>>(instance.held)) ||
			       (text == "object" && std::holds_alternative<std::vector<JsonMember>>(instance.held));
		};
		auto const list = engine::core::alternative<std::vector<Json>>(type->held);
		if (list.has_value() ? std::ranges::none_of(*list, matches) : !matches(*type)) {
			return false;
		}
	}
	if (auto const values = keyword("enum")) {
		if (std::ranges::none_of(std::get<std::vector<Json>>(values->held), [&instance](Json const& value) { return equal(value, instance); })) {
			return false;
		}
	}
	if (auto const value = keyword("const"); value && !equal(*value, instance)) {
		return false;
	}
	if (auto const choices = keyword("anyOf")) {
		if (std::ranges::none_of(std::get<std::vector<Json>>(choices->held), [&](Json const& choice) { return validates(root, choice, instance); })) {
			return false;
		}
	}
	if (auto const choices = keyword("oneOf")) {
		if (std::ranges::count_if(std::get<std::vector<Json>>(choices->held), [&](Json const& choice) { return validates(root, choice, instance); }) != 1) {
			return false;
		}
	}
	if (auto const text = engine::core::alternative<std::string>(instance.held)) {
		auto const length = code_points(*text);
		if (auto const low = keyword("minLength"); low && length < static_cast<std::size_t>(std::get<double>(low->held))) {
			return false;
		}
		if (auto const high = keyword("maxLength"); high && length > static_cast<std::size_t>(std::get<double>(high->held))) {
			return false;
		}
		if (auto const pattern = keyword("pattern"); pattern && !std::regex_search(*text, std::regex{std::get<std::string>(pattern->held)})) {
			return false;
		}
		if (auto const format = keyword("format"); format && !valid_format(std::get<std::string>(format->held), *text)) {
			return false;
		}
	}
	if (auto const number = engine::core::alternative<double>(instance.held)) {
		auto const infinity = std::numeric_limits<double>::infinity();
		auto const bound = [&](std::string_view name, double absent) {
			return keyword(name).transform([](Json const& value) { return std::get<double>(value.held); }).value_or(absent);
		};
		if (*number < bound("minimum", -infinity) || *number > bound("maximum", infinity) || *number <= bound("exclusiveMinimum", -infinity) ||
		    *number >= bound("exclusiveMaximum", infinity)) {
			return false;
		}
	}
	if (auto const elements = engine::core::alternative<std::vector<Json>>(instance.held)) {
		if (auto const low = keyword("minItems"); low && elements->size() < static_cast<std::size_t>(std::get<double>(low->held))) {
			return false;
		}
		if (auto const high = keyword("maxItems"); high && elements->size() > static_cast<std::size_t>(std::get<double>(high->held))) {
			return false;
		}
		if (auto const items = keyword("items")) {
			if (!std::ranges::all_of(*elements, [&](Json const& element) { return validates(root, *items, element); })) {
				return false;
			}
		}
	}
	if (auto const members = engine::core::alternative<std::vector<JsonMember>>(instance.held)) {
		auto const properties = keyword("properties");
		if (auto const required = keyword("required")) {
			for (auto const& name : std::get<std::vector<Json>>(required->held)) {
				if (!instance.find(std::get<std::string>(name.held))) {
					return false;
				}
			}
		}
		auto const extra = keyword("additionalProperties");
		for (auto const& member : *members) {
			auto const declared = properties ? properties->find(member.name) : std::nullopt;
			if (declared) {
				if (!validates(root, *declared, member.value)) {
					return false;
				}
			} else if (extra && !validates(root, *extra, member.value)) {
				return false;
			}
		}
	}
	return true;
}


/** What a walk draws on: every token's bytes, and the tokens it leans toward to make progress. */
export struct Walker {
	std::vector<std::string> bytes;
	/** Tokens that close or open structure: markers, quotes, brackets, separators. */
	std::vector<bool> closer;
	/** Short tokens of structure or digits. */
	std::vector<bool> structural;
};

export [[nodiscard]] auto make_walker(engine::tokenizer::Tokenizer const& tokenizer) -> Walker {
	auto walker = Walker{.bytes = {}, .closer = {}, .structural = {}};
	for (auto id = engine::tokenizer::Id{0}; id < engine::tokenizer::vocabulary; ++id) {
		auto const text = std::string{tokenizer.token(id).value_or("")};
		auto const trimmed = std::string_view{text}.substr(std::min(text.size(), text.find_first_not_of(" \n")));
		auto const closes = trimmed.starts_with("</") || trimmed.starts_with("<tool_call>") || trimmed.starts_with("<parameter=") || trimmed == "\"" ||
		                    trimmed == "}" || trimmed == "]" || trimmed == "," || trimmed == ">\n" || trimmed == ">";
		walker.closer.push_back(closes && text.size() <= 16);
		walker.structural.push_back(text.size() <= 8 && text.find_first_of("<>{}[]\":,=/\n0123456789") != std::string::npos);
		walker.bytes.push_back(text);
	}
	return walker;
}

/** A walk's output: its text, whether it ended, its length, and how long each mask took. */
export struct Walk {
	std::string text;
	bool ended;
	std::size_t tokens;
	std::vector<std::chrono::nanoseconds> masks;
};

/** The texts a finishing walk tries first, in order: what closes the structures the format has. */
inline constexpr auto finishers = std::to_array<std::string_view>({
    "\n</parameter>\n", "</function>\n</tool_call>", "</think>", "<tool_call>", "\"", "}", "]", "1", "0", "a", ":", "\n",
});

/**
 * Samples a reply under the masks. First it explores: among the allowed tokens, uniformly, leaning toward tokens
 * that open or close structure. Then it finishes: it ends where it may, else writes the first finishing text the
 * grammar takes whole, else takes a token from a small allowed set (forced syntax) or a structural one. Every token
 * is one the mask allows; the walk fails if a mask is empty before the end.
 */
export [[nodiscard]] auto walk(grammar::Grammar const& compiled, grammar::Vocabulary const& vocabulary, Walker const& walker, std::uint64_t seed, std::size_t explore, std::size_t limit)
    -> std::optional<Walk> {
	auto random = engine::testing::generator(seed);
	auto matcher = grammar::Matcher::make(compiled, grammar::engine_layout);
	auto out = Walk{.text = {}, .ended = false, .tokens = 0, .masks = {}};
	auto const chance = [&random](double p) { return std::bernoulli_distribution{p}(random); };
	auto const byte_token = [&walker](char byte) {
		return static_cast<grammar::Token>(std::ranges::find(walker.bytes, std::string(1, byte)) - walker.bytes.begin());
	};
	for (auto step = std::size_t{0}; step < limit; ++step) {
		auto const began = std::chrono::steady_clock::now();
		auto const mask = matcher.mask(vocabulary);
		out.masks.push_back(std::chrono::steady_clock::now() - began);
		if (!mask) {
			return std::nullopt;
		}
		auto allowed = std::vector<grammar::Token>{};
		auto closers = std::vector<grammar::Token>{};
		auto structural = std::vector<grammar::Token>{};
		auto ends = std::vector<grammar::Token>{};
		for (auto const [index, word] : std::views::enumerate(mask->tokens)) {
			for (auto bits = word; bits != 0; bits &= bits - 1) {
				auto const token = static_cast<grammar::Token>((static_cast<std::size_t>(index) * 64) + static_cast<std::size_t>(std::countr_zero(bits)));
				if (vocabulary.ends_with(token)) {
					ends.push_back(token);
					continue;
				}
				allowed.push_back(token);
				if (walker.closer[token]) {
					closers.push_back(token);
				}
				if (walker.structural[token]) {
					structural.push_back(token);
				}
			}
		}
		if (allowed.empty() && ends.empty()) {
			std::println("  stuck after {:?}", out.text);
			return std::nullopt;
		}
		auto const pick = [&random](std::vector<grammar::Token> const& from) { return from[std::uniform_int_distribution<std::size_t>{0, from.size() - 1}(random)]; };
		auto const finishing = step >= explore;
		if (!ends.empty() && (allowed.empty() || finishing || chance(0.02))) {
			if (!matcher.advance(vocabulary, ends.front())) {
				return std::nullopt;
			}
			out.ended = true;
			return out;
		}
		if (finishing) {
			// Tried in a fresh order each time: in free text every finisher is taken, and a fixed first one would loop.
			auto order = std::vector<std::string_view>{finishers.begin(), finishers.end()};
			std::ranges::shuffle(order, random);
			auto const written = std::ranges::find_if(order, [&](std::string_view text) {
				for (auto const [count, byte] : std::views::enumerate(text)) {
					if (!matcher.advance(vocabulary, byte_token(byte))) {
						matcher.rollback(static_cast<std::size_t>(count));
						return false;
					}
				}
				return true;
			});
			if (written != order.end()) {
				out.text.append(*written);
				out.tokens += written->size();
				continue;
			}
		}
		auto chosen = grammar::Token{0};
		auto const forced = allowed.size() <= 64 && (finishing || allowed.size() <= 16);
		if (!forced && !closers.empty() && chance(finishing ? 0.5 : 0.15)) {
			chosen = pick(closers);
		} else if (!forced && !structural.empty() && chance(finishing ? 0.9 : 0.5)) {
			chosen = pick(structural);
		} else {
			chosen = pick(allowed);
		}
		if (!matcher.advance(vocabulary, chosen)) {
			std::println("  a masked token was refused");
			return std::nullopt;
		}
		++out.tokens;
		out.text.append(walker.bytes[chosen]);
	}
	return out;
}


}
