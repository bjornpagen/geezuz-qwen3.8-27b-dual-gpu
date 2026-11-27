// SPDX-License-Identifier: AGPL-3.0-only
export module engine.regex:syntax;

import std;
import engine.core;
import :expr;
import :utf8;

namespace engine::regex {

/** Why a regular expression was refused: a malformed one, or a feature the engine does not implement exactly. */
export enum class Fault : std::uint8_t {
	/** The text ends inside a construct. */
	UnexpectedEnd,
	/** A `)` without its `(`, or a `(` never closed. */
	UnbalancedGroup,
	/** A quantifier with nothing before it to repeat. */
	NothingToRepeat,
	/** A `{` that is no well-formed repetition, or one whose minimum exceeds its maximum. */
	BadRepetition,
	/** A repetition count past `max_repetition`. */
	RepetitionTooLarge,
	/** An escape the dialect does not define. */
	BadEscape,
	/** A class range whose start follows its end, or a class with nothing in it. */
	BadClass,
	/** The text is not utf-8, or an escape names no scalar value. */
	BadCharacter,
	/** Groups nest past `max_nesting`. */
	TooDeep,
	/** The expression grows past the grammar's node bound. */
	TooLarge,
	/** `(?=`, `(?!`, `(?<=`, `(?<!`: lookaround. */
	Lookaround,
	/** `*?`, `+?`, `??`, `{n,m}?`, or the `U` flag: lazy repetition. */
	LazyQuantifier,
	/** `^`, `$`, `\A`, `\z` where the dialect gives them no exact meaning here. */
	Anchor,
	/** `\b`, `\B`. */
	WordBoundary,
	/** `\1`, `\k<name>`: backreferences. */
	Backreference,
	/** `\p{...}`, `\P{...}`: Unicode property classes. */
	UnicodeProperty,
	/** A character past ascii under case-insensitivity: the engine folds ascii letters only, so it refuses the rest. */
	UnicodeCase,
	/** `[[...]]`, `[:alpha:]`, `&&`, `--`, `~~` inside a class, or a class starting with `]`: class set syntax. */
	ClassSyntax,
	/** A group flag other than `i`, `s`, `m`, `u`, `R`, or a group syntax other than `(?:`, `(?P<n>`, `(?<n>`. */
	GroupSyntax,
};

export struct [[nodiscard]] RegexError {
	Fault fault;
	/** Byte offset into the regex text where the fault was found. */
	std::size_t offset;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		// Every fault is of the regex as written.
		switch (fault) {
		case Fault::UnexpectedEnd:
		case Fault::UnbalancedGroup:
		case Fault::NothingToRepeat:
		case Fault::BadRepetition:
		case Fault::RepetitionTooLarge:
		case Fault::BadEscape:
		case Fault::BadClass:
		case Fault::BadCharacter:
		case Fault::TooDeep:
		case Fault::TooLarge:
		case Fault::Lookaround:
		case Fault::LazyQuantifier:
		case Fault::Anchor:
		case Fault::WordBoundary:
		case Fault::Backreference:
		case Fault::UnicodeProperty:
		case Fault::UnicodeCase:
		case Fault::ClassSyntax:
		case Fault::GroupSyntax:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(RegexError const&) const -> bool = default;
};

/** The largest repetition count accepted; past it a regex is refused by name rather than grown without bound. */
export inline constexpr std::uint32_t max_repetition = 100'000;
/** The deepest nesting of groups and classes accepted. */
export inline constexpr std::size_t max_nesting = 128;

/**
 * How a pattern is matched against a whole text. Lark terminals and regex tools match the whole text; a JSON-schema
 * `pattern` searches it, unless a leading `^` or trailing `$` anchors that end.
 */
export enum class Matching : std::uint8_t {
	Whole,
	Search,
};

enum class Case : std::uint8_t {
	Sensitive,
	Insensitive,
};

/** What `.` matches. */
enum class Dot : std::uint8_t {
	/** Every character but the line terminators of either dialect. */
	Line,
	/** Every character. */
	Any,
};

struct Flags {
	Case letters;
	Dot dot;
};

/** Unicode whitespace in both dialects: what `\s` matches. */
[[nodiscard]] auto space_class() -> CharClass {
	auto const ranges = std::to_array<Range>({
	    {.low = 0x09, .high = 0x0d},
	    {.low = 0x20, .high = 0x20},
	    {.low = 0xa0, .high = 0xa0},
	    {.low = 0x1680, .high = 0x1680},
	    {.low = 0x2000, .high = 0x200a},
	    {.low = 0x2028, .high = 0x2029},
	    {.low = 0x202f, .high = 0x202f},
	    {.low = 0x205f, .high = 0x205f},
	    {.low = 0x3000, .high = 0x3000},
	});
	return CharClass::of(ranges);
}

/** Unicode whitespace in either dialect: what `\S` never matches. */
[[nodiscard]] auto any_space_class() -> CharClass {
	auto out = space_class();
	out.add(Range{.low = 0x85, .high = 0x85});
	out.add(Range{.low = 0xfeff, .high = 0xfeff});
	return out;
}

[[nodiscard]] auto digit_class() -> CharClass {
	return CharClass::of(std::array{Range{.low = '0', .high = '9'}});
}

[[nodiscard]] auto word_class() -> CharClass {
	return CharClass::of(std::to_array<Range>({
	    {.low = '0', .high = '9'},
	    {.low = 'A', .high = 'Z'},
	    {.low = '_', .high = '_'},
	    {.low = 'a', .high = 'z'},
	}));
}

[[nodiscard]] auto ascii_class() -> CharClass {
	return CharClass::of(std::array{Range{.low = 0, .high = 0x7f}});
}

/** What `.` matches outside `s` mode: everything but `\n`, `\r`, U+2028 and U+2029, the terminators of either dialect. */
[[nodiscard]] auto line_class() -> CharClass {
	auto const terminators = CharClass::of(std::to_array<Range>({
	    {.low = '\n', .high = '\n'},
	    {.low = '\r', .high = '\r'},
	    {.low = 0x2028, .high = 0x2029},
	}));
	return CharClass::all().minus(terminators);
}

/** The class with every ascii letter's other case added. */
[[nodiscard]] auto case_closed(CharClass const& set) -> CharClass {
	auto out = set;
	for (auto letter = char32_t{'a'}; letter <= 'z'; ++letter) {
		auto const upper = letter - 'a' + 'A';
		if (set.contains(letter) || set.contains(upper)) {
			out.add(Range{.low = letter, .high = letter});
			out.add(Range{.low = upper, .high = upper});
		}
	}
	return out;
}

/** One character at the front of utf-8 text, with its length. */
struct Decoded {
	char32_t point;
	std::size_t length;
};

[[nodiscard]] auto decode_front(std::string_view text) -> std::optional<Decoded> {
	if (text.empty()) {
		return std::nullopt;
	}
	auto const lead = std::uint32_t{std::bit_cast<std::uint8_t>(text[0])};
	auto const length = lead < 0x80 ? 1U : lead >= 0xf0 && lead < 0xf5 ? 4U : lead >= 0xe0 && lead < 0xf0 ? 3U : lead >= 0xc2 && lead < 0xe0 ? 2U : 0U;
	if (length == 0 || text.size() < length) {
		return std::nullopt;
	}
	auto value = length == 1 ? lead : lead & (0x7fU >> length);
	for (auto at = std::size_t{1}; at < length; ++at) {
		auto const next = std::uint32_t{std::bit_cast<std::uint8_t>(text[at])};
		if ((next & 0xc0U) != 0x80U) {
			return std::nullopt;
		}
		value = (value << 6U) | (next & 0x3fU);
	}
	auto const point = char32_t{value};
	if (point >= scalar_end || (point >= 0xd800 && point <= 0xdfff) || encoded_length(point) != length) {
		return std::nullopt;
	}
	return Decoded{.point = point, .length = length};
}

/** Every character of a text must be utf-8 for a literal to match it as written. */
export [[nodiscard]] auto valid_utf8(std::string_view text) -> bool {
	while (!text.empty()) {
		auto const decoded = decode_front(text);
		if (!decoded) {
			return false;
		}
		text.remove_prefix(decoded->length);
	}
	return true;
}

/** A parsed piece and whether the text it came from was a top-level alternation. */
struct Parsed {
	Expr expr;
	bool alternation;
};

template<class T>
using Parse = core::Result<T, RegexError>;

/**
 * Reads a regular expression in the syntax both the Rust `regex` crate and ECMA-262 give one meaning, and refuses by
 * name what they give different meanings or what the engine does not implement. Where the two dialects' classes
 * differ, a class means the characters both dialects agree on (`\d` is ASCII digits, `\W` ASCII non-word
 * characters), so every match is a match in either dialect.
 */
class Reader {
public:
	Reader(Store& store, std::string_view text) : store_{store}, text_{text} {}

	[[nodiscard]] auto whole() -> Parse<Parsed> {
		auto parsed = alternation(Flags{.letters = Case::Sensitive, .dot = Dot::Line}, 0);
		if (!parsed) {
			return parsed;
		}
		if (at_ != text_.size()) {
			return fail(Fault::UnbalancedGroup);
		}
		if (store_.full()) {
			return fail(Fault::TooLarge);
		}
		return parsed;
	}

private:
	[[nodiscard]] auto fail(Fault fault) const -> std::unexpected<RegexError> {
		return std::unexpected(RegexError{.fault = fault, .offset = at_});
	}

	[[nodiscard]] auto peek() const -> std::optional<char> {
		if (at_ >= text_.size()) {
			return std::nullopt;
		}
		return text_[at_];
	}

	[[nodiscard]] auto rest() const -> std::string_view {
		return text_.substr(at_);
	}

	[[nodiscard]] auto alternation(Flags flags, std::size_t depth) -> Parse<Parsed> {
		auto choices = std::vector<Expr>{};
		while (true) {
			auto branch = sequence(flags, depth);
			if (!branch) {
				return std::unexpected(branch.error());
			}
			choices.push_back(*branch);
			if (peek() != '|') {
				break;
			}
			++at_;
		}
		return Parsed{.expr = store_.any_of(choices), .alternation = choices.size() > 1};
	}

	[[nodiscard]] auto sequence(Flags& flags, std::size_t depth) -> Parse<Expr> {
		auto parts = std::vector<Expr>{};
		while (at_ < text_.size() && peek() != '|' && peek() != ')') {
			auto atom = this->atom(flags, depth);
			if (!atom) {
				return std::unexpected(atom.error());
			}
			if (!*atom) {
				// A flag group sets flags for the rest of the sequence and matches nothing itself.
				continue;
			}
			auto repeated = quantified(**atom);
			if (!repeated) {
				return repeated;
			}
			parts.push_back(*repeated);
		}
		return store_.sequence(parts);
	}

	/** Reads one atom; empty for a flag group, which sets `flags` instead. */
	[[nodiscard]] auto atom(Flags& flags, std::size_t depth) -> Parse<std::optional<Expr>> {
		// The caller reads an atom only before the end of the text.
		auto const c = peek().value_or('\0');
		switch (c) {
		case '(':
			return group(flags, depth);
		case '[': {
			++at_;
			auto set = class_body(flags);
			if (!set) {
				return std::unexpected(set.error());
			}
			return characters(store_, *set);
		}
		case '.':
			++at_;
			return characters(store_, flags.dot == Dot::Any ? CharClass::all() : line_class());
		case '^':
		case '$':
			return fail(Fault::Anchor);
		case '*':
		case '+':
		case '?':
			return fail(Fault::NothingToRepeat);
		case '{':
			return fail(Fault::BadRepetition);
		case '\\': {
			auto set = escape(Context::Outside);
			if (!set) {
				return std::unexpected(set.error());
			}
			auto const point = single_point(*set);
			if (flags.letters == Case::Insensitive && point && *point >= 0x80) {
				return fail(Fault::UnicodeCase);
			}
			return characters(store_, flags.letters == Case::Insensitive ? case_closed(*set) : *set);
		}
		default: {
			auto const decoded = decode_front(rest());
			if (!decoded) {
				return fail(Fault::BadCharacter);
			}
			if (flags.letters == Case::Insensitive && decoded->point >= 0x80) {
				return fail(Fault::UnicodeCase);
			}
			at_ += decoded->length;
			auto const set = CharClass::single(decoded->point);
			return characters(store_, flags.letters == Case::Insensitive ? case_closed(set) : set);
		}
		}
	}

	[[nodiscard]] auto group(Flags& flags, std::size_t depth) -> Parse<std::optional<Expr>> {
		if (depth + 1 > max_nesting) {
			return fail(Fault::TooDeep);
		}
		++at_;
		auto inner = flags;
		if (rest().starts_with('?')) {
			auto const syntax = rest();
			if (syntax.starts_with("?=") || syntax.starts_with("?!") || syntax.starts_with("?<=") || syntax.starts_with("?<!")) {
				return fail(Fault::Lookaround);
			}
			if (syntax.starts_with("?:")) {
				at_ += 2;
			} else if (syntax.starts_with("?P<") || syntax.starts_with("?<")) {
				at_ += syntax.starts_with("?P<") ? 3 : 2;
				auto const close = rest().find('>');
				if (close == std::string_view::npos || close == 0) {
					return fail(Fault::GroupSyntax);
				}
				at_ += close + 1;
			} else {
				++at_;
				auto parsed = read_flags(inner);
				if (!parsed) {
					return std::unexpected(parsed.error());
				}
				if (peek() == ')') {
					++at_;
					flags = inner;
					return std::nullopt;
				}
				if (peek() != ':') {
					return fail(Fault::GroupSyntax);
				}
				++at_;
			}
		}
		auto body = alternation(inner, depth + 1);
		if (!body) {
			return std::unexpected(body.error());
		}
		if (peek() != ')') {
			return fail(Fault::UnbalancedGroup);
		}
		++at_;
		return body->expr;
	}

	[[nodiscard]] auto read_flags(Flags& flags) -> Parse<void> {
		auto negated = false;
		while (auto const c = peek()) {
			if (*c == ':' || *c == ')') {
				return {};
			}
			++at_;
			switch (*c) {
			case '-':
				negated = true;
				break;
			case 'i':
				flags.letters = negated ? Case::Sensitive : Case::Insensitive;
				break;
			case 's':
				flags.dot = negated ? Dot::Line : Dot::Any;
				break;
			// Multi-line mode changes only anchors, which are refused; Unicode mode is the meaning classes already
			// have; CRLF mode narrows `.` by `\r`, which it already excludes.
			case 'm':
			case 'u':
			case 'R':
				break;
			case 'U':
				return fail(Fault::LazyQuantifier);
			default:
				return fail(Fault::GroupSyntax);
			}
		}
		return fail(Fault::UnexpectedEnd);
	}

	[[nodiscard]] auto number() -> std::optional<std::uint64_t> {
		auto const begin = at_;
		auto value = std::uint64_t{0};
		while (at_ < text_.size() && text_[at_] >= '0' && text_[at_] <= '9') {
			value = std::min(value * 10 + static_cast<std::uint64_t>(text_[at_] - '0'), std::uint64_t{max_repetition} + 1);
			++at_;
		}
		if (at_ == begin) {
			return std::nullopt;
		}
		return value;
	}

	[[nodiscard]] auto quantified(Expr body) -> Parse<Expr> {
		while (auto const c = peek()) {
			auto min = std::uint64_t{0};
			auto max = std::uint64_t{unbounded};
			if (*c == '*') {
				++at_;
			} else if (*c == '+') {
				++at_;
				min = 1;
			} else if (*c == '?') {
				++at_;
				max = 1;
			} else if (*c == '{') {
				++at_;
				auto const low = number();
				if (!low) {
					return fail(Fault::BadRepetition);
				}
				min = *low;
				max = *low;
				if (peek() == ',') {
					++at_;
					max = number().value_or(unbounded);
				}
				if (peek() != '}') {
					return fail(Fault::BadRepetition);
				}
				++at_;
				if ((max != unbounded && max > max_repetition) || min > max_repetition) {
					return fail(Fault::RepetitionTooLarge);
				}
				if (min > max) {
					return fail(Fault::BadRepetition);
				}
			} else {
				return body;
			}
			if (peek() == '?') {
				return fail(Fault::LazyQuantifier);
			}
			if (peek() == '+') {
				return fail(Fault::BadRepetition);
			}
			// A quantifier of a quantifier is an error in ECMA-262 and redundant in Rust: refused, not guessed.
			if (auto const next = peek(); next == '*' || next == '+' || next == '?' || next == '{') {
				return fail(Fault::NothingToRepeat);
			}
			return store_.repeat(body, core::narrow<std::uint32_t>(min), core::narrow<std::uint32_t>(max));
		}
		return body;
	}

	enum class Context : std::uint8_t {
		Outside,
		InClass,
	};

	[[nodiscard]] auto hex_scalar(std::size_t digits_wanted) -> Parse<char32_t> {
		auto value = std::uint32_t{0};
		auto digits = std::size_t{0};
		auto const braced = peek() == '{';
		if (braced) {
			++at_;
		}
		while (auto const c = peek()) {
			if (braced && *c == '}') {
				++at_;
				break;
			}
			if (!braced && digits == digits_wanted) {
				break;
			}
			auto const digit = *c >= '0' && *c <= '9'   ? *c - '0'
			                   : *c >= 'a' && *c <= 'f' ? *c - 'a' + 10
			                   : *c >= 'A' && *c <= 'F' ? *c - 'A' + 10
			                                            : -1;
			if (digit < 0 || digits >= 8) {
				return fail(Fault::BadEscape);
			}
			value = value * 16 + static_cast<std::uint32_t>(digit);
			++digits;
			++at_;
		}
		if (digits == 0 || (!braced && digits != digits_wanted)) {
			return fail(Fault::BadEscape);
		}
		auto const point = char32_t{value};
		if (point >= scalar_end || (point >= 0xd800 && point <= 0xdfff)) {
			return fail(Fault::BadCharacter);
		}
		return point;
	}

	/** Reads an escape after its backslash: the class of characters it stands for. */
	[[nodiscard]] auto escape(Context context) -> Parse<CharClass> {
		++at_;
		auto const c = peek();
		if (!c) {
			return fail(Fault::UnexpectedEnd);
		}
		++at_;
		auto const single = [](char32_t point) { return CharClass::single(point); };
		switch (*c) {
		case 'n':
			return single('\n');
		case 't':
			return single('\t');
		case 'r':
			return single('\r');
		case 'f':
			return single('\f');
		case 'v':
			return single('\v');
		case 'a':
			return single('\a');
		case 'd':
			return digit_class();
		case 'D':
			return ascii_class().minus(digit_class());
		case 'w':
			return word_class();
		case 'W':
			return ascii_class().minus(word_class());
		case 's':
			return space_class();
		case 'S':
			return CharClass::all().minus(any_space_class());
		case 'x': {
			auto const point = hex_scalar(2);
			if (!point) {
				return std::unexpected(point.error());
			}
			return single(*point);
		}
		case 'u': {
			auto const point = hex_scalar(4);
			if (!point) {
				return std::unexpected(point.error());
			}
			return single(*point);
		}
		case 'U': {
			auto const point = hex_scalar(8);
			if (!point) {
				return std::unexpected(point.error());
			}
			return single(*point);
		}
		case 'p':
		case 'P':
			return fail(Fault::UnicodeProperty);
		case 'b':
		case 'B':
			return context == Context::InClass ? fail(Fault::BadEscape) : fail(Fault::WordBoundary);
		case 'A':
		case 'z':
		case 'Z':
			return fail(Fault::Anchor);
		case 'k':
			return fail(Fault::Backreference);
		default:
			if (*c >= '0' && *c <= '9') {
				return fail(Fault::Backreference);
			}
			// Both dialects read an escaped ascii punctuation character as itself.
			if (std::string_view{"\\.+*?()|[]{}^$#&-~/\"'<>!@%,:;=_`"}.contains(*c)) {
				return single(char32_t{static_cast<unsigned char>(*c)});
			}
			return fail(Fault::BadEscape);
		}
	}

	/** One member of a class: a character, or an escape's class. */
	[[nodiscard]] auto class_atom() -> Parse<CharClass> {
		if (peek() == '\\') {
			return escape(Context::InClass);
		}
		auto const decoded = decode_front(rest());
		if (!decoded) {
			return at_ >= text_.size() ? fail(Fault::UnexpectedEnd) : fail(Fault::BadCharacter);
		}
		at_ += decoded->length;
		return CharClass::single(decoded->point);
	}

	[[nodiscard]] static auto single_point(CharClass const& set) -> std::optional<char32_t> {
		if (set.ranges().size() == 1 && set.ranges().front().low == set.ranges().front().high) {
			return set.ranges().front().low;
		}
		return std::nullopt;
	}

	/** Reads a class after its `[`, through its `]`. */
	[[nodiscard]] auto class_body(Flags flags) -> Parse<CharClass> {
		auto const negated = peek() == '^';
		if (negated) {
			++at_;
		}
		if (peek() == ']') {
			return fail(Fault::ClassSyntax);
		}
		auto set = CharClass{};
		// Whether a member written out (not an escape's class) lies past ascii.
		auto wide = false;
		while (true) {
			auto const c = peek();
			if (!c) {
				return fail(Fault::UnexpectedEnd);
			}
			if (*c == ']') {
				++at_;
				break;
			}
			if (*c == '[' || rest().starts_with("&&") || rest().starts_with("--") || rest().starts_with("~~")) {
				return fail(Fault::ClassSyntax);
			}
			auto first = class_atom();
			if (!first) {
				return first;
			}
			if (peek() == '-' && !rest().starts_with("-]")) {
				++at_;
				auto last = class_atom();
				if (!last) {
					return last;
				}
				auto const low = single_point(*first);
				auto const high = single_point(*last);
				if (!low || !high || *low > *high) {
					return fail(Fault::BadClass);
				}
				wide = wide || *high >= 0x80;
				set.add(Range{.low = *low, .high = *high});
			} else {
				auto const point = single_point(*first);
				wide = wide || (point && *point >= 0x80);
				set.add(*first);
			}
		}
		if (flags.letters == Case::Insensitive) {
			if (wide) {
				return fail(Fault::UnicodeCase);
			}
			set = case_closed(set);
		}
		if (negated) {
			set = set.complemented();
		}
		if (set.empty()) {
			return fail(Fault::BadClass);
		}
		return set;
	}

	Store& store_;
	std::string_view text_;
	std::size_t at_ = 0;
};

/** Whether the character at `end - 1` is escaped: an odd run of backslashes stands before it. */
[[nodiscard]] auto escaped_at(std::string_view text, std::size_t at) -> bool {
	auto count = std::size_t{0};
	while (at > count && text[at - count - 1] == '\\') {
		++count;
	}
	return count % 2 == 1;
}

/**
 * The language of a regular expression: the whole texts it matches. With `Matching::Search`, a pattern matches every
 * text containing a match, except that a leading `^` and a trailing `$` pin the match to that end; an anchor anywhere
 * else, or anchors around a top-level alternation (`^a|b$`), is refused.
 */
export [[nodiscard]] auto parse(Store& store, std::string_view text, Matching matching) -> core::Result<Expr, RegexError> {
	if (matching == Matching::Whole) {
		auto parsed = Reader{store, text}.whole();
		if (!parsed) {
			return std::unexpected(parsed.error());
		}
		return parsed->expr;
	}
	auto body = text;
	auto const anchored_start = body.starts_with('^');
	if (anchored_start) {
		body.remove_prefix(1);
	}
	auto const anchored_end = body.ends_with('$') && !escaped_at(body, body.size() - 1);
	if (anchored_end) {
		body.remove_suffix(1);
	}
	auto parsed = Reader{store, body}.whole();
	if (!parsed) {
		auto error = parsed.error();
		error.offset += anchored_start ? 1 : 0;
		return std::unexpected(error);
	}
	if ((anchored_start || anchored_end) && parsed->alternation) {
		return std::unexpected(RegexError{.fault = Fault::Anchor, .offset = 0});
	}
	auto const text_any = any_text(store);
	auto const parts = std::array{anchored_start ? empty : text_any, parsed->expr, anchored_end ? empty : text_any};
	auto const out = store.sequence(parts);
	if (store.full()) {
		return std::unexpected(RegexError{.fault = Fault::TooLarge, .offset = 0});
	}
	return out;
}

}
