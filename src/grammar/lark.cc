// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:lark;

import std;
import engine.core;
import engine.regex;
import :error;
import :builder;
import :document;
import :strings;
import :schema;
import :lark_syntax;

namespace engine::grammar::lark {

struct ExpansionsRef {
	std::uint32_t index;
};

struct AtomRef {
	std::uint32_t index;
};

/** Whether a string literal matches its letters in either case (`"..."i`). */
enum class Letters : std::uint8_t {
	Exact,
	AnyCase,
};

struct Literal {
	std::string text;
	Letters letters;
};

struct Pattern {
	std::string text;
	std::string flags;
};

struct Name {
	std::string name;
	Kind kind;
	std::size_t line;
};

/** `"a".."z"`: one character from a range. */
struct CharRange {
	std::string low;
	std::string high;
	std::size_t line;
};

struct Group {
	ExpansionsRef inner;
};

struct Maybe {
	ExpansionsRef inner;
};

/** `~atom`: every text the atom does not match. */
struct Negated {
	AtomRef inner;
	std::size_t line;
};

/** `%json { ... }`: the values a JSON schema admits, as JSON text. */
struct Embedded {
	std::string schema;
	std::size_t line;
};

using Atom = std::variant<Literal, Pattern, Name, CharRange, Group, Maybe, Negated, Embedded>;

struct Item {
	AtomRef atom;
	std::uint32_t min;
	std::uint32_t max;
};

struct Sequence {
	std::vector<Item> items;
};

/** One alternative: sequences joined by `&`. */
struct Alternative {
	std::vector<Sequence> conjuncts;
	std::size_t line;
};

struct Expansions {
	std::vector<Alternative> alternatives;
};

/** How a rule's single lexeme ends: `[lazy]`, or `[suffix=...]` with the suffix's atom. */
struct Options {
	Mode mode;
	std::optional<AtomRef> suffix;
};

struct Definition {
	std::string name;
	ExpansionsRef body;
	std::size_t line;
	std::optional<Options> options;
};

/** A parsed lark grammar. */
struct Ast {
	std::vector<Atom> atoms;
	std::vector<Expansions> expansions;
	std::vector<Definition> rules;
	std::vector<Definition> terminals;
	std::vector<ExpansionsRef> ignores;
	/** Imported terminals: the name they go by, and their name in `common`. */
	std::vector<std::pair<std::string, std::string>> imports;
};

/** Lark's `common` terminals, as llguidance defines them. */
inline constexpr auto common_terminals = std::to_array<std::pair<std::string_view, std::string_view>>({
    {"DIGIT", "[0-9]"},
    {"HEXDIGIT", "[a-fA-F0-9]"},
    {"INT", "[0-9]+"},
    {"SIGNED_INT", R"((\+|-)?[0-9]+)"},
    {"DECIMAL", R"(([0-9]+\.[0-9]*)|(\.[0-9]+))"},
    {"_EXP", R"([eE](\+|-)?[0-9]+)"},
    {"FLOAT", R"(([0-9]+\.[0-9]*|\.[0-9]+)([eE](\+|-)?[0-9]+)?|[0-9]+[eE](\+|-)?[0-9]+)"},
    {"SIGNED_FLOAT", R"((\+|-)?(([0-9]+\.[0-9]*|\.[0-9]+)([eE](\+|-)?[0-9]+)?|[0-9]+[eE](\+|-)?[0-9]+))"},
    {"NUMBER", R"(([0-9]+)|([0-9]+\.[0-9]*|\.[0-9]+)([eE](\+|-)?[0-9]+)?|[0-9]+[eE](\+|-)?[0-9]+)"},
    {"SIGNED_NUMBER", R"((\+|-)?(([0-9]+)|([0-9]+\.[0-9]*|\.[0-9]+)([eE](\+|-)?[0-9]+)?|[0-9]+[eE](\+|-)?[0-9]+))"},
    {"ESCAPED_STRING", R"("([^"\\]|\\.)*")"},
    {"LCASE_LETTER", "[a-z]"},
    {"UCASE_LETTER", "[A-Z]"},
    {"LETTER", "[A-Za-z]"},
    {"WORD", "[A-Za-z]+"},
    {"CNAME", "[_A-Za-z][_A-Za-z0-9]*"},
    {"WS_INLINE", "[ \\t]+"},
    {"WS", R"([ \t\f\r\n]+)"},
    {"CR", "\\r"},
    {"LF", "\\n"},
    {"NEWLINE", "(\\r?\\n)+"},
    {"SH_COMMENT", "#[^\\n]*"},
    {"CPP_COMMENT", "//[^\\n]*"},
    {"C_COMMENT", R"(\/\*[^*]*\*+(?:[^/*][^*]*\*+)*\/)"},
    {"SQL_COMMENT", "--[^\\n]*"},
});

/** Reads tokens into an `Ast`: definitions end at a newline, unless the next line continues them with `|`. */
class Reader {
public:
	explicit Reader(std::vector<Token> tokens) : tokens_{std::move(tokens)} {}

	[[nodiscard]] auto file(std::size_t max_definitions) -> core::Result<Ast, GrammarError> {
		while (true) {
			skip_newlines();
			auto const& token = peek();
			if (token.kind == Kind::End) {
				return std::move(ast_);
			}
			auto read = token.kind == Kind::Directive ? statement() : definition();
			if (!read) {
				return std::unexpected(read.error());
			}
			if (ast_.rules.size() + ast_.terminals.size() > max_definitions) {
				return refuse(Refusal::LarkTooLarge, std::format("line {}", token.line));
			}
		}
	}

private:
	[[nodiscard]] auto peek() const -> Token const& {
		return tokens_[at_];
	}

	[[nodiscard]] auto next() -> Token const& {
		auto const& token = tokens_[at_];
		if (token.kind != Kind::End) {
			++at_;
		}
		return token;
	}

	auto skip_newlines() -> void {
		while (peek().kind == Kind::Newline) {
			++at_;
		}
	}

	[[nodiscard]] auto fail() const -> std::unexpected<GrammarError> {
		return syntax_error(peek().line);
	}

	[[nodiscard]] auto expect(Kind kind) -> core::Result<Token, GrammarError> {
		if (peek().kind != kind) {
			return fail();
		}
		return next();
	}

	[[nodiscard]] auto end_of_line() -> core::Result<void, GrammarError> {
		if (peek().kind != Kind::Newline && peek().kind != Kind::End) {
			return fail();
		}
		return {};
	}

	[[nodiscard]] auto statement() -> core::Result<void, GrammarError> {
		auto const directive = next();
		if (directive.text == "%ignore") {
			auto body = expansions(0);
			if (!body) {
				return std::unexpected(body.error());
			}
			ast_.ignores.push_back(*body);
			return end_of_line();
		}
		if (directive.text == "%import") {
			return import();
		}
		return refuse(Refusal::LarkDirective, std::format("line {}: {}", directive.line, directive.text));
	}

	[[nodiscard]] auto import() -> core::Result<void, GrammarError> {
		auto const module = next();
		if (module.kind != Kind::Rule || module.text != "common") {
			return refuse(Refusal::LarkDirective, std::format("line {}: %import", module.line));
		}
		if (peek().kind == Kind::OpenParen) {
			std::ignore = next();
			while (true) {
				auto const name = expect(Kind::Terminal);
				if (!name) {
					return std::unexpected(name.error());
				}
				ast_.imports.emplace_back(name->text, name->text);
				if (peek().kind == Kind::Comma) {
					std::ignore = next();
					continue;
				}
				if (auto const close = expect(Kind::CloseParen); !close) {
					return std::unexpected(close.error());
				}
				return end_of_line();
			}
		}
		if (auto const dot = expect(Kind::Dot); !dot) {
			return std::unexpected(dot.error());
		}
		auto const name = expect(Kind::Terminal);
		if (!name) {
			return std::unexpected(name.error());
		}
		auto alias = name->text;
		if (peek().kind == Kind::Arrow) {
			std::ignore = next();
			auto const renamed = expect(Kind::Terminal);
			if (!renamed) {
				return std::unexpected(renamed.error());
			}
			alias = renamed->text;
		}
		ast_.imports.emplace_back(alias, name->text);
		return end_of_line();
	}

	[[nodiscard]] auto definition() -> core::Result<void, GrammarError> {
		auto const head = next();
		if (head.kind != Kind::Rule && head.kind != Kind::Terminal) {
			return syntax_error(head.line);
		}
		if (peek().kind == Kind::Dot) {
			return refuse(Refusal::LarkPriority, std::format("line {}", head.line));
		}
		if (peek().kind == Kind::OpenBrace) {
			return refuse(Refusal::LarkTemplate, std::format("line {}", head.line));
		}
		auto options = std::optional<Options>{};
		if (head.kind == Kind::Rule && peek().kind == Kind::OpenBracket) {
			auto read = rule_options(head.line);
			if (!read) {
				return std::unexpected(read.error());
			}
			options = *read;
		}
		if (auto const colon = expect(Kind::Colon); !colon) {
			return std::unexpected(colon.error());
		}
		auto body = expansions(0);
		if (!body) {
			return std::unexpected(body.error());
		}
		auto const name = std::string{std::string_view{head.text}.substr(head.text.find_first_not_of("!?"))};
		auto& definitions = head.kind == Kind::Rule ? ast_.rules : ast_.terminals;
		definitions.push_back(Definition{.name = name, .body = *body, .line = head.line, .options = options});
		return end_of_line();
	}

	/** `[lazy]`, `[suffix="..."]`, `[capture]`, `[capture="..."]`, comma-separated. */
	[[nodiscard]] auto rule_options(std::size_t line) -> core::Result<std::optional<Options>, GrammarError> {
		std::ignore = next();
		auto out = std::optional<Options>{};
		while (true) {
			auto const key = next();
			if (key.kind != Kind::Rule) {
				return syntax_error(line);
			}
			auto value = std::optional<AtomRef>{};
			if (peek().kind == Kind::Equals) {
				std::ignore = next();
				if (peek().kind == Kind::Number) {
					// Numeric options (`max_tokens`, `temperature`) are all refused below.
					std::ignore = next();
					return refuse(Refusal::LarkOption, std::format("line {}: {}", line, key.text));
				}
				auto read = atom();
				if (!read) {
					return std::unexpected(read.error());
				}
				value = *read;
			}
			if (key.text == "lazy" && !value) {
				out = Options{.mode = Mode::Lazy, .suffix = out ? out->suffix : std::nullopt};
			} else if (key.text == "suffix" && value) {
				out = Options{.mode = Mode::Lazy, .suffix = value};
			} else if (key.text != "capture") {
				return refuse(Refusal::LarkOption, std::format("line {}: {}", line, key.text));
			}
			if (peek().kind == Kind::Comma) {
				std::ignore = next();
				continue;
			}
			if (auto const close = expect(Kind::CloseBracket); !close) {
				return std::unexpected(close.error());
			}
			return out;
		}
	}

	/** Alternatives; inside brackets (`nesting` > 0) newlines are spaces, at the top a newline then `|` continues. */
	[[nodiscard]] auto expansions(std::size_t nesting) -> core::Result<ExpansionsRef, GrammarError> {
		auto out = Expansions{};
		while (true) {
			if (nesting > 0) {
				skip_newlines();
			}
			auto read = alternative(nesting);
			if (!read) {
				return std::unexpected(read.error());
			}
			out.alternatives.push_back(std::move(*read));
			auto const before = at_;
			skip_newlines();
			if (peek().kind == Kind::Pipe) {
				std::ignore = next();
				continue;
			}
			if (nesting == 0) {
				at_ = before;
			}
			break;
		}
		ast_.expansions.push_back(std::move(out));
		return ExpansionsRef{.index = core::narrow<std::uint32_t>(ast_.expansions.size() - 1)};
	}

	[[nodiscard]] auto alternative(std::size_t nesting) -> core::Result<Alternative, GrammarError> {
		auto out = Alternative{.conjuncts = {}, .line = peek().line};
		while (true) {
			auto read = sequence(nesting);
			if (!read) {
				return std::unexpected(read.error());
			}
			out.conjuncts.push_back(std::move(*read));
			if (peek().kind != Kind::Ampersand) {
				break;
			}
			std::ignore = next();
		}
		if (peek().kind == Kind::Arrow) {
			// An alias names a tree node; it changes nothing a decoder accepts.
			std::ignore = next();
			if (auto const name = expect(Kind::Rule); !name) {
				return std::unexpected(name.error());
			}
		}
		return out;
	}

	[[nodiscard]] auto sequence(std::size_t nesting) -> core::Result<Sequence, GrammarError> {
		auto out = Sequence{};
		while (true) {
			if (nesting > 0) {
				skip_newlines();
			}
			auto const kind = peek().kind;
			if (kind == Kind::Pipe || kind == Kind::CloseParen || kind == Kind::CloseBracket || kind == Kind::Newline || kind == Kind::End ||
			    kind == Kind::Arrow || kind == Kind::Ampersand) {
				return out;
			}
			auto read = atom();
			if (!read) {
				return std::unexpected(read.error());
			}
			auto const item = Item{.atom = *read, .min = 1, .max = 1};
			auto quantified = quantifier(item);
			if (!quantified) {
				return std::unexpected(quantified.error());
			}
			out.items.push_back(*quantified);
		}
	}

	[[nodiscard]] auto count() -> core::Result<std::uint32_t, GrammarError> {
		auto const number = expect(Kind::Number);
		if (!number) {
			return std::unexpected(number.error());
		}
		if (number->text.size() > 6 || std::stoul(number->text) > regex::max_repetition) {
			return refuse(Refusal::LarkTooLarge, std::format("line {}", number->line));
		}
		return core::narrow<std::uint32_t>(std::stoul(number->text));
	}

	[[nodiscard]] auto quantifier(Item item) -> core::Result<Item, GrammarError> {
		auto const& token = peek();
		if (token.kind == Kind::Operator) {
			std::ignore = next();
			item.min = token.text == "+" ? 1 : 0;
			item.max = token.text == "?" ? 1 : regex::unbounded;
		} else if (token.kind == Kind::Tilde) {
			std::ignore = next();
			auto const low = count();
			if (!low) {
				return std::unexpected(low.error());
			}
			item.min = *low;
			item.max = *low;
			if (peek().kind == Kind::DotDot) {
				std::ignore = next();
				auto const high = count();
				if (!high) {
					return std::unexpected(high.error());
				}
				item.max = *high;
			}
		} else if (token.kind == Kind::OpenBrace) {
			std::ignore = next();
			if (peek().kind != Kind::Number && peek().kind != Kind::Comma) {
				return refuse(Refusal::LarkTemplate, std::format("line {}", token.line));
			}
			auto low = std::uint32_t{0};
			if (peek().kind == Kind::Number) {
				auto const read = count();
				if (!read) {
					return std::unexpected(read.error());
				}
				low = *read;
			}
			auto high = low;
			if (peek().kind == Kind::Comma) {
				std::ignore = next();
				high = regex::unbounded;
				if (peek().kind == Kind::Number) {
					auto const read = count();
					if (!read) {
						return std::unexpected(read.error());
					}
					high = *read;
				}
			}
			if (auto const close = expect(Kind::CloseBrace); !close) {
				return std::unexpected(close.error());
			}
			item.min = low;
			item.max = high;
		} else {
			return item;
		}
		if (item.min > item.max) {
			return fail();
		}
		return item;
	}

	[[nodiscard]] auto add(Atom atom) -> AtomRef {
		ast_.atoms.push_back(std::move(atom));
		return AtomRef{.index = core::narrow<std::uint32_t>(ast_.atoms.size() - 1)};
	}

	[[nodiscard]] auto atom() -> core::Result<AtomRef, GrammarError> {
		auto const token = next();
		switch (token.kind) {
		case Kind::OpenParen:
		case Kind::OpenBracket: {
			auto inner = expansions(1);
			if (!inner) {
				return std::unexpected(inner.error());
			}
			skip_newlines();
			if (auto const close = expect(token.kind == Kind::OpenParen ? Kind::CloseParen : Kind::CloseBracket); !close) {
				return std::unexpected(close.error());
			}
			return token.kind == Kind::OpenParen ? add(Group{.inner = *inner}) : add(Maybe{.inner = *inner});
		}
		case Kind::Tilde: {
			auto inner = atom();
			if (!inner) {
				return inner;
			}
			return add(Negated{.inner = *inner, .line = token.line});
		}
		case Kind::String:
			if (peek().kind == Kind::DotDot) {
				std::ignore = next();
				auto const high = expect(Kind::String);
				if (!high) {
					return std::unexpected(high.error());
				}
				return add(CharRange{.low = token.text, .high = high->text, .line = token.line});
			}
			return add(Literal{.text = token.text, .letters = token.extra == "i" ? Letters::AnyCase : Letters::Exact});
		case Kind::Regexp:
			return add(Pattern{.text = token.text, .flags = token.extra});
		case Kind::Rule:
		case Kind::Terminal:
			return add(Name{.name = token.text, .kind = token.kind, .line = token.line});
		case Kind::Directive:
			if (token.text == "%json") {
				return add(Embedded{.schema = token.extra, .line = token.line});
			}
			return refuse(Refusal::LarkDirective, std::format("line {}: {}", token.line, token.text));
		case Kind::Special:
		case Kind::Reference:
			return refuse(Refusal::LarkToken, std::format("line {}: {}", token.line, token.text));
		default:
			return syntax_error(token.line);
		}
	}

	std::vector<Token> tokens_;
	std::size_t at_ = 0;
	Ast ast_;
};

/** Where a lark grammar's lexemes run: under a region's guard or free, and what its embedded JSON may contain. */
struct Setting {
	std::optional<Region> region;
	Place json;
};

/**
 * Compiles a lark grammar into the builder: terminals become regular expressions, rules nonterminals, `%ignore`
 * an optional run of its lexeme between every two symbols. Borrows the builder and the tree for one compile.
 */
class Compiler {
public:
	Compiler(Builder& builder, Ast const& ast, Setting setting, Usage& usage) : builder_{builder}, store_{builder.store()}, ast_{ast}, setting_{setting}, usage_{usage} {}

	[[nodiscard]] auto start() -> core::Result<Symbol, GrammarError> {
		for (auto const& rule : ast_.rules) {
			if (std::ranges::count(ast_.rules, rule.name, &Definition::name) > 1) {
				return refuse(Refusal::LarkUndefined, std::format("line {}: {}", rule.line, rule.name));
			}
		}
		if (!ast_.ignores.empty()) {
			auto patterns = std::vector<regex::Expr>{};
			for (auto const ignore : ast_.ignores) {
				auto pattern = terminal_expansions(ignore);
				if (!pattern) {
					return std::unexpected(pattern.error());
				}
				patterns.push_back(*pattern);
			}
			ignore_ = builder_.star(builder_.lexeme(store_.any_of(patterns), Mode::Greedy, setting_.region));
		}
		auto const found = std::ranges::find(ast_.rules, std::string_view{"start"}, &Definition::name);
		if (found == ast_.rules.end()) {
			return refuse(Refusal::LarkNoStart, "");
		}
		auto body = rule("start", found->line);
		if (!body) {
			return body;
		}
		if (!ignore_) {
			return *body;
		}
		auto const out = builder_.nonterminal();
		builder_.rule(out, {*ignore_, *body, *ignore_});
		return out;
	}

private:
	[[nodiscard]] auto at(std::size_t line, std::string_view what) const -> std::string {
		return std::format("line {}: {}", line, what);
	}

	/** A terminal's language, by name: defined here, or imported from `common`. */
	[[nodiscard]] auto terminal(std::string const& name, std::size_t line) -> core::Result<regex::Expr, GrammarError> {
		if (auto const found = terminals_.find(name); found != terminals_.end()) {
			if (!found->second) {
				return refuse(Refusal::LarkRecursiveTerminal, at(line, name));
			}
			return *found->second;
		}
		auto const definition = std::ranges::find(ast_.terminals, name, &Definition::name);
		auto out = regex::nothing;
		if (definition != ast_.terminals.end()) {
			if (std::ranges::count(ast_.terminals, name, &Definition::name) > 1) {
				return refuse(Refusal::LarkUndefined, at(line, name));
			}
			terminals_.emplace(name, std::nullopt);
			auto read = terminal_expansions(definition->body);
			if (!read) {
				return read;
			}
			out = *read;
		} else {
			auto const imported = std::ranges::find(ast_.imports, name, &std::pair<std::string, std::string>::first);
			auto const common = imported == ast_.imports.end()
			                        ? common_terminals.end()
			                        : std::ranges::find(common_terminals, imported->second, &std::pair<std::string_view, std::string_view>::first);
			if (common == common_terminals.end()) {
				return refuse(Refusal::LarkUndefined, at(line, name));
			}
			auto parsed = regex::parse(store_, common->second, regex::Matching::Whole);
			core::invariant(parsed.has_value());
			out = *parsed;
		}
		terminals_.insert_or_assign(name, out);
		return out;
	}

	[[nodiscard]] auto terminal_expansions(ExpansionsRef ref) -> core::Result<regex::Expr, GrammarError> {
		auto choices = std::vector<regex::Expr>{};
		for (auto const& alternative : ast_.expansions[ref.index].alternatives) {
			auto conjuncts = std::vector<regex::Expr>{};
			for (auto const& sequence : alternative.conjuncts) {
				auto parts = std::vector<regex::Expr>{};
				for (auto const& item : sequence.items) {
					auto part = terminal_atom(item.atom);
					if (!part) {
						return part;
					}
					parts.push_back(store_.repeat(*part, item.min, item.max));
				}
				conjuncts.push_back(store_.sequence(parts));
			}
			choices.push_back(store_.all_of(conjuncts));
		}
		return store_.any_of(choices);
	}

	[[nodiscard]] auto literal(Literal const& literal) -> regex::Expr {
		if (literal.letters == Letters::Exact) {
			return store_.literal(literal.text);
		}
		auto parts = std::vector<regex::Expr>{};
		for (auto const c : literal.text) {
			auto set = regex::ByteSet::none();
			set.add(std::bit_cast<std::uint8_t>(c));
			if (c >= 'a' && c <= 'z') {
				set.add(static_cast<std::uint8_t>(c - 'a' + 'A'));
			} else if (c >= 'A' && c <= 'Z') {
				set.add(static_cast<std::uint8_t>(c - 'A' + 'a'));
			}
			parts.push_back(store_.bytes(set));
		}
		return store_.sequence(parts);
	}

	[[nodiscard]] auto pattern(Pattern const& pattern, std::size_t line) -> core::Result<regex::Expr, GrammarError> {
		if (pattern.text.size() > builder_.bounds().regex_bytes) {
			return refuse(Refusal::LarkTooLarge, at(line, "regex"));
		}
		auto flags = std::string{};
		for (auto const flag : pattern.flags) {
			if (flag == 'i' || flag == 's') {
				flags.push_back(flag);
			} else if (flag != 'm' && flag != 'u') {
				return refuse_regex(regex::RegexError{.fault = regex::Fault::GroupSyntax, .offset = 0}, at(line, "regex flags"));
			}
		}
		auto const text = flags.empty() ? pattern.text : std::format("(?{}:{})", flags, pattern.text);
		auto parsed = regex::parse(store_, text, regex::Matching::Whole);
		if (!parsed) {
			return refuse_regex(parsed.error(), at(line, "regex"));
		}
		return *parsed;
	}

	[[nodiscard]] auto terminal_atom(AtomRef ref) -> core::Result<regex::Expr, GrammarError> {
		auto const& atom = ast_.atoms[ref.index];
		if (auto const held = core::alternative<Literal>(atom)) {
			auto const wide = std::ranges::any_of(held->text, [](char c) { return std::bit_cast<std::uint8_t>(c) >= 0x80; });
			if (held->letters == Letters::AnyCase && wide) {
				// Only ascii letters fold; a literal past ascii under `i` would match less than lark means.
				return refuse_regex(regex::RegexError{.fault = regex::Fault::UnicodeCase, .offset = 0}, "case-insensitive literal");
			}
			return literal(*held);
		}
		if (auto const held = core::alternative<Pattern>(atom)) {
			return pattern(*held, 0);
		}
		if (auto const held = core::alternative<Name>(atom)) {
			if (held->kind != Kind::Terminal) {
				return refuse(Refusal::LarkSyntax, at(held->line, held->name));
			}
			return terminal(held->name, held->line);
		}
		if (auto const held = core::alternative<CharRange>(atom)) {
			auto const low = single_scalar(held->low);
			auto const high = single_scalar(held->high);
			if (!low || !high || *low > *high) {
				return refuse(Refusal::LarkSyntax, at(held->line, "range"));
			}
			return regex::characters(store_, regex::CharClass::of(std::array{regex::Range{.low = *low, .high = *high}}));
		}
		if (auto const held = core::alternative<Group>(atom)) {
			return terminal_expansions(held->inner);
		}
		if (auto const held = core::alternative<Maybe>(atom)) {
			auto inner = terminal_expansions(held->inner);
			if (!inner) {
				return inner;
			}
			return store_.optional(*inner);
		}
		if (auto const held = core::alternative<Negated>(atom)) {
			auto inner = terminal_atom(held->inner);
			if (!inner) {
				return inner;
			}
			return store_.complement(*inner);
		}
		return refuse(Refusal::LarkSyntax, at(std::get<Embedded>(atom).line, "%json in a terminal"));
	}

	[[nodiscard]] static auto single_scalar(std::string_view text) -> std::optional<char32_t> {
		if (text.empty()) {
			return std::nullopt;
		}
		auto const lead = std::uint32_t{std::bit_cast<std::uint8_t>(text[0])};
		auto const length = lead < 0x80 ? 1U : lead >= 0xf0 ? 4U : lead >= 0xe0 ? 3U : 2U;
		if (text.size() != length) {
			return std::nullopt;
		}
		auto value = length == 1 ? lead : lead & (0x7fU >> length);
		for (auto const c : text.substr(1)) {
			value = (value << 6U) | (std::uint32_t{std::bit_cast<std::uint8_t>(c)} & 0x3fU);
		}
		return char32_t{value};
	}

	/** A rule by name, its nonterminal made on first use, so rules may refer to each other in any order. */
	[[nodiscard]] auto rule(std::string const& name, std::size_t line) -> core::Result<Symbol, GrammarError> {
		if (auto const found = rules_.find(name); found != rules_.end()) {
			return found->second;
		}
		auto const definition = std::ranges::find(ast_.rules, name, &Definition::name);
		if (definition == ast_.rules.end()) {
			return refuse(Refusal::LarkUndefined, at(line, name));
		}
		if (definition->options) {
			// A rule with lexeme options is one lexeme: its body must be a terminal's.
			auto body = terminal_expansions(definition->body);
			if (!body) {
				return refuse(Refusal::LarkOption, at(definition->line, name));
			}
			auto language = *body;
			if (definition->options->suffix) {
				auto suffix = terminal_atom(*definition->options->suffix);
				if (!suffix) {
					return std::unexpected(suffix.error());
				}
				language = store_.concat(language, *suffix);
			}
			auto const symbol = builder_.lexeme(language, definition->options->mode, setting_.region);
			rules_.emplace(name, symbol);
			return symbol;
		}
		auto const out = builder_.nonterminal();
		rules_.emplace(name, out);
		auto read = rule_body(out, definition->body);
		if (!read) {
			return std::unexpected(read.error());
		}
		return out;
	}

	[[nodiscard]] auto rule_body(Nonterminal out, ExpansionsRef ref) -> core::Result<void, GrammarError> {
		for (auto const& alternative : ast_.expansions[ref.index].alternatives) {
			if (alternative.conjuncts.size() > 1) {
				return refuse(Refusal::LarkOperator, at(alternative.line, "&"));
			}
			auto rhs = std::vector<Symbol>{};
			for (auto const& item : alternative.conjuncts.front().items) {
				auto symbol = rule_item(item, alternative.line);
				if (!symbol) {
					return std::unexpected(symbol.error());
				}
				if (ignore_ && !rhs.empty()) {
					rhs.push_back(*ignore_);
				}
				rhs.push_back(*symbol);
			}
			builder_.rule(out, std::move(rhs));
		}
		return {};
	}

	[[nodiscard]] auto rule_item(Item const& item, std::size_t line) -> core::Result<Symbol, GrammarError> {
		auto base = rule_atom(item.atom, line);
		if (!base) {
			return base;
		}
		if (item.min == 1 && item.max == 1) {
			return *base;
		}
		if (item.max == 0) {
			auto const out = builder_.nonterminal();
			builder_.rule(out, {});
			return out;
		}
		auto const separator = ignore_ ? std::optional<Symbol>{*ignore_} : std::nullopt;
		return builder_.separated(*base, separator, item.min, item.max);
	}

	[[nodiscard]] auto rule_atom(AtomRef ref, std::size_t line) -> core::Result<Symbol, GrammarError> {
		auto const& atom = ast_.atoms[ref.index];
		if (auto const held = core::alternative<Name>(atom); held && held->kind == Kind::Rule) {
			return rule(held->name, held->line);
		}
		if (auto const held = core::alternative<Group>(atom)) {
			auto const out = builder_.nonterminal();
			auto read = rule_body(out, held->inner);
			if (!read) {
				return std::unexpected(read.error());
			}
			return out;
		}
		if (auto const held = core::alternative<Maybe>(atom)) {
			auto const out = builder_.nonterminal();
			auto read = rule_body(out, held->inner);
			if (!read) {
				return std::unexpected(read.error());
			}
			builder_.rule(out, {});
			return out;
		}
		if (auto const held = core::alternative<Negated>(atom)) {
			return refuse(Refusal::LarkOperator, at(held->line, "~"));
		}
		if (auto const held = core::alternative<Embedded>(atom)) {
			auto document = read_document(held->schema, at(held->line, "%json"));
			if (!document) {
				return std::unexpected(document.error());
			}
			documents_.push_back(std::make_unique<Value>(std::move(*document)));
			auto compiler = SchemaCompiler{builder_, *documents_.back(), "", setting_.json, usage_};
			return compiler.value(*documents_.back(), at(held->line, "%json"), Types::all(), 0);
		}
		auto language = terminal_atom(ref);
		if (!language) {
			return std::unexpected(language.error());
		}
		std::ignore = line;
		return builder_.lexeme(*language, Mode::Greedy, setting_.region);
	}

	Builder& builder_;
	regex::Store& store_;
	Ast const& ast_;
	Setting setting_;
	Usage& usage_;
	std::optional<Symbol> ignore_;
	/** Terminal languages by name; nothing while one is being defined, which a reference back to it finds. */
	std::map<std::string, std::optional<regex::Expr>> terminals_;
	std::map<std::string, Symbol> rules_;
	/** Embedded schema documents, kept while their compile reads them. */
	std::vector<std::unique_ptr<Value>> documents_;
};

/** Compiles lark text into the builder; returns the nonterminal of its whole language. */
[[nodiscard]] auto compile(Builder& builder, std::string_view text, Setting setting, Usage& usage) -> core::Result<Nonterminal, GrammarError> {
	if (text.size() > builder.bounds().lark_bytes) {
		return refuse(Refusal::LarkTooLarge, "");
	}
	auto tokens = tokenize(text);
	if (!tokens) {
		return std::unexpected(tokens.error());
	}
	auto ast = Reader{std::move(*tokens)}.file(builder.bounds().lark_rules);
	if (!ast) {
		return std::unexpected(ast.error());
	}
	auto start = Compiler{builder, *ast, setting, usage}.start();
	if (!start) {
		return std::unexpected(start.error());
	}
	auto const symbol = *start;
	if (auto const nonterminal = core::alternative<Nonterminal>(symbol)) {
		return *nonterminal;
	}
	auto const out = builder.nonterminal();
	builder.rule(out, {symbol});
	return out;
}

}
