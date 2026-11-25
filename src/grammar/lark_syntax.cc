// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:lark_syntax;

import std;
import engine.core;
import :error;

namespace engine::grammar::lark {

enum class Kind : std::uint8_t {
	Rule,
	Terminal,
	String,
	Regexp,
	Number,
	Operator,
	Colon,
	Pipe,
	OpenParen,
	CloseParen,
	OpenBracket,
	CloseBracket,
	OpenBrace,
	CloseBrace,
	Comma,
	Dot,
	DotDot,
	Arrow,
	Tilde,
	Ampersand,
	Equals,
	Directive,
	Special,
	Reference,
	Newline,
	End,
};

/** A lark source token: its kind, its text (a string's or regex's body decoded), and its line. */
struct Token {
	Kind kind;
	std::string text;
	std::size_t line;
	/** A string's `i` suffix, a regex's flags, or the JSON object after `%json` and its kin. */
	std::string extra;
};

[[nodiscard]] auto syntax_error(std::size_t line) -> std::unexpected<GrammarError> {
	return refuse(Refusal::LarkSyntax, std::format("line {}", line));
}

[[nodiscard]] auto identifier_char(char c) -> bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
}

[[nodiscard]] auto hex_value(char c) -> std::optional<std::uint32_t> {
	if (c >= '0' && c <= '9') {
		return static_cast<std::uint32_t>(c - '0');
	}
	if (c >= 'a' && c <= 'f') {
		return static_cast<std::uint32_t>(c - 'a' + 10);
	}
	if (c >= 'A' && c <= 'F') {
		return static_cast<std::uint32_t>(c - 'A' + 10);
	}
	return std::nullopt;
}

/** Appends a scalar value's utf-8; false for a surrogate or a value past the last scalar. */
[[nodiscard]] auto put_scalar(std::string& out, std::uint32_t value) -> bool {
	if (value >= 0x110000 || (value >= 0xd800 && value <= 0xdfff)) {
		return false;
	}
	auto const put = [&out](std::uint32_t byte) { out.push_back(static_cast<char>(static_cast<std::uint8_t>(byte))); };
	if (value < 0x80) {
		put(value);
	} else if (value < 0x800) {
		put(0xc0U | (value >> 6U));
		put(0x80U | (value & 0x3fU));
	} else if (value < 0x10000) {
		put(0xe0U | (value >> 12U));
		put(0x80U | ((value >> 6U) & 0x3fU));
		put(0x80U | (value & 0x3fU));
	} else {
		put(0xf0U | (value >> 18U));
		put(0x80U | ((value >> 12U) & 0x3fU));
		put(0x80U | ((value >> 6U) & 0x3fU));
		put(0x80U | (value & 0x3fU));
	}
	return true;
}

/**
 * Splits lark text into tokens. Strings take JSON's escapes and `\xHH`; a regex runs to its unescaped `/`, `\/`
 * standing for `/`; `//` and `#` start comments; newlines are tokens, since a definition ends at one.
 */
[[nodiscard]] auto tokenize(std::string_view text) -> core::Result<std::vector<Token>, GrammarError> {
	auto out = std::vector<Token>{};
	auto line = std::size_t{1};
	auto at = std::size_t{0};
	auto const push = [&](Kind kind, std::string body) { out.push_back(Token{.kind = kind, .text = std::move(body), .line = line, .extra = {}}); };
	while (at < text.size()) {
		auto const c = text[at];
		if (c == ' ' || c == '\t' || c == '\r') {
			++at;
			continue;
		}
		if (c == '\n') {
			if (out.empty() || out.back().kind != Kind::Newline) {
				push(Kind::Newline, "");
			}
			++line;
			++at;
			continue;
		}
		if (c == '#' || text.substr(at).starts_with("//")) {
			while (at < text.size() && text[at] != '\n') {
				++at;
			}
			continue;
		}
		if (c == '"') {
			auto body = std::string{};
			++at;
			while (true) {
				if (at >= text.size() || text[at] == '\n') {
					return syntax_error(line);
				}
				auto const d = text[at];
				if (d == '"') {
					++at;
					break;
				}
				if (std::bit_cast<std::uint8_t>(d) < 0x20) {
					return syntax_error(line);
				}
				if (d != '\\') {
					body.push_back(d);
					++at;
					continue;
				}
				if (at + 1 >= text.size()) {
					return syntax_error(line);
				}
				auto const e = text[at + 1];
				at += 2;
				switch (e) {
				case '"':
				case '\\':
				case '/':
					body.push_back(e);
					break;
				case 'b':
					body.push_back('\b');
					break;
				case 'f':
					body.push_back('\f');
					break;
				case 'n':
					body.push_back('\n');
					break;
				case 'r':
					body.push_back('\r');
					break;
				case 't':
					body.push_back('\t');
					break;
				case 'x':
				case 'u': {
					auto const digits = e == 'x' ? std::size_t{2} : std::size_t{4};
					auto value = std::uint32_t{0};
					for (auto index = std::size_t{0}; index < digits; ++index) {
						auto const digit = at < text.size() ? hex_value(text[at]) : std::nullopt;
						if (!digit) {
							return syntax_error(line);
						}
						value = value * 16 + *digit;
						++at;
					}
					if (!put_scalar(body, value)) {
						return syntax_error(line);
					}
					break;
				}
				default:
					return syntax_error(line);
				}
			}
			push(Kind::String, std::move(body));
			if (at < text.size() && text[at] == 'i') {
				out.back().extra = "i";
				++at;
			}
			continue;
		}
		if (c == '/') {
			auto body = std::string{};
			++at;
			while (true) {
				if (at >= text.size() || text[at] == '\n') {
					return syntax_error(line);
				}
				if (text[at] == '/') {
					++at;
					break;
				}
				if (text[at] == '\\' && at + 1 < text.size()) {
					if (text[at + 1] == '/') {
						body.push_back('/');
					} else {
						body.append(text.substr(at, 2));
					}
					at += 2;
					continue;
				}
				body.push_back(text[at]);
				++at;
			}
			push(Kind::Regexp, std::move(body));
			while (at < text.size() && std::string_view{"imslux"}.contains(text[at])) {
				out.back().extra.push_back(text[at]);
				++at;
			}
			continue;
		}
		if (c == '<') {
			auto const close = text.find('>', at);
			if (close == std::string_view::npos) {
				return syntax_error(line);
			}
			push(Kind::Special, std::string{text.substr(at, close + 1 - at)});
			at = close + 1;
			continue;
		}
		if (c == '%' || c == '@') {
			auto const begin = at;
			++at;
			while (at < text.size() && identifier_char(text[at])) {
				++at;
			}
			push(c == '%' ? Kind::Directive : Kind::Reference, std::string{text.substr(begin, at - begin)});
			auto const& directive = out.back().text;
			if (directive == "%json" || directive == "%regex" || directive == "%llguidance" || directive == "%int_ranges") {
				// The directive's JSON object follows whole: kept as text, braces matched outside strings.
				while (at < text.size() && (text[at] == ' ' || text[at] == '\t')) {
					++at;
				}
				if (at >= text.size() || text[at] != '{') {
					return syntax_error(line);
				}
				auto const begin_json = at;
				auto depth = std::size_t{0};
				auto in_string = false;
				for (; at < text.size(); ++at) {
					auto const d = text[at];
					if (d == '\n') {
						++line;
					}
					if (in_string) {
						if (d == '\\') {
							++at;
						} else if (d == '"') {
							in_string = false;
						}
					} else if (d == '"') {
						in_string = true;
					} else if (d == '{') {
						++depth;
					} else if (d == '}' && --depth == 0) {
						++at;
						break;
					}
				}
				if (depth != 0) {
					return syntax_error(line);
				}
				out.back().extra = std::string{text.substr(begin_json, at - begin_json)};
			}
			continue;
		}
		if (c >= '0' && c <= '9') {
			auto const begin = at;
			while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
				++at;
			}
			push(Kind::Number, std::string{text.substr(begin, at - begin)});
			continue;
		}
		auto const prefixed = (c == '!' || c == '?') && at + 1 < text.size() && identifier_char(text[at + 1]);
		if (c == '?' && !prefixed) {
			push(Kind::Operator, "?");
			++at;
			continue;
		}
		if ((identifier_char(c) && c != '-') || prefixed) {
			auto const begin = at;
			++at;
			while (at < text.size() && identifier_char(text[at])) {
				++at;
			}
			auto name = std::string{text.substr(begin, at - begin)};
			// A rule's `!` and `?` shape lark's trees, which constrained decoding does not build.
			auto const letters = name.find_first_not_of("!?_");
			if (letters == std::string::npos) {
				return syntax_error(line);
			}
			auto const bare = std::string_view{name}.substr(letters);
			if (bare.front() == '-' || (bare.front() >= '0' && bare.front() <= '9')) {
				return syntax_error(line);
			}
			auto const upper = std::ranges::none_of(bare, [](char letter) { return letter >= 'a' && letter <= 'z'; });
			if (upper && (name.front() == '!' || name.front() == '?')) {
				return syntax_error(line);
			}
			push(upper ? Kind::Terminal : Kind::Rule, std::move(name));
			continue;
		}
		if (text.substr(at).starts_with("..")) {
			push(Kind::DotDot, "..");
			at += 2;
			continue;
		}
		if (text.substr(at).starts_with("->")) {
			push(Kind::Arrow, "->");
			at += 2;
			continue;
		}
		auto const single = std::to_array<std::pair<char, Kind>>({
		    {':', Kind::Colon},
		    {'|', Kind::Pipe},
		    {'(', Kind::OpenParen},
		    {')', Kind::CloseParen},
		    {'[', Kind::OpenBracket},
		    {']', Kind::CloseBracket},
		    {'{', Kind::OpenBrace},
		    {'}', Kind::CloseBrace},
		    {',', Kind::Comma},
		    {'.', Kind::Dot},
		    {'~', Kind::Tilde},
		    {'&', Kind::Ampersand},
		    {'=', Kind::Equals},
		    {'+', Kind::Operator},
		    {'*', Kind::Operator},
		});
		auto const found = std::ranges::find(single, c, &std::pair<char, Kind>::first);
		if (found == single.end()) {
			return syntax_error(line);
		}
		push(found->second, std::string(1, c));
		++at;
	}
	if (out.empty() || out.back().kind != Kind::Newline) {
		push(Kind::Newline, "");
	}
	push(Kind::End, "");
	return out;
}

}
