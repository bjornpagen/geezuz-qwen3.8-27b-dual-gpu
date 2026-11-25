// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:error;

import std;
import engine.core;
import engine.json;
import engine.regex;

namespace engine::grammar {

/**
 * Why a request's grammar was refused. Each case is a feature the engine does not implement exactly, a malformed
 * input, or an input past a named compile-time bound; the endpoint answers each with its own 400.
 */
export enum class Refusal : std::uint8_t {
	// JSON schema.
	/** A schema is neither an object nor a boolean. */
	SchemaShape,
	/** A keyword the engine does not implement; `at` names it. */
	SchemaKeyword,
	/** A keyword's value has the wrong shape (a `required` that is no list of strings, a negative `maxLength`). */
	SchemaValue,
	/** A `type` naming no JSON type. */
	SchemaType,
	/** A `$ref` that is not `#`, `#/$defs/<name>` or `#/definitions/<name>`, or names no definition. */
	SchemaReference,
	/** A string `format` the engine does not implement. */
	SchemaFormat,
	/** `oneOf` whose choices the engine cannot prove disjoint. */
	SchemaOneOf,
	/** `multipleOf`, which the engine does not implement. */
	SchemaMultipleOf,
	/** A numeric bound the engine cannot hold exactly (an exclusive real bound past 15 significant digits). */
	SchemaBound,
	/** The grammar admits no output at all: a schema no value satisfies, a lark grammar with no finite derivation. */
	Empty,
	/** A tool argument may hold an object with additional properties: the chat format's reader refuses repeated keys. */
	SchemaOpenArgument,
	/** A tool argument's type list mixes `string` with a type the chat format would read the text as. */
	SchemaStringUnion,
	/** A tool argument's schema is recursive: it could nest deeper than the chat format reads arguments back. */
	SchemaRecursiveArgument,
	/** A function tool's `parameters` is not an object schema. */
	ToolParameters,
	/** A tool or parameter name the chat format cannot write: empty, or holding `<`, `>` or a newline. */
	ToolName,
	/** Two tools share a name. */
	ToolDuplicate,
	/** A tool choice names a tool the request does not define, or requires a call when no tool may be called. */
	ToolUnknown,
	// Lark.
	/** Lark text that does not parse. */
	LarkSyntax,
	/** A rule or terminal used but never defined, or defined twice. */
	LarkUndefined,
	/** A terminal defined in terms of itself. */
	LarkRecursiveTerminal,
	/** No `start` rule. */
	LarkNoStart,
	/** Terminal or rule priorities. */
	LarkPriority,
	/** Templates. */
	LarkTemplate,
	/** `%import` of anything but `common`, `%declare`, `%override`, `%llguidance` and the other directives. */
	LarkDirective,
	/** A special-token reference `<...>`, or a grammar reference `@...`. */
	LarkToken,
	/** A lexeme option other than `lazy` and `suffix` (`stop`, `max_tokens`, `temperature`), or one on a rule whose body is not a terminal. */
	LarkOption,
	/** `&` or `~` outside a terminal definition. */
	LarkOperator,
	// Regular expressions.
	/** A regular expression refused: `fault` says why. */
	Regex,
	// Bounds.
	/** The schema text is past `Bounds::schema_bytes`, or nests past `Bounds::schema_depth`. */
	SchemaTooLarge,
	/** Properties, enum values or definitions past their bounds. */
	SchemaTooMany,
	/** The lark text is past `Bounds::lark_bytes`, or has more rules than `Bounds::lark_rules`. */
	LarkTooLarge,
	/** A lexeme's automaton has more states than `Bounds::lexeme_states`, or all together more than `Bounds::states`. */
	AutomatonTooLarge,
	/** The grammar has more lexemes, rules or symbols than its bounds, or its expressions more nodes. */
	GrammarTooLarge,
};

/** Where a regex fault was found, when the refusal is `Refusal::Regex`. */
export using RegexDetail = std::optional<regex::RegexError>;

export struct [[nodiscard]] GrammarError {
	Refusal refusal;
	/** Where in the request: a JSON pointer into a schema, a lark line, a tool's name; for the message, never parsed. */
	std::string at;
	RegexDetail regex;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		// Every refusal is of the request as written: sent again unchanged, it is refused again.
		switch (refusal) {
		case Refusal::SchemaShape:
		case Refusal::SchemaKeyword:
		case Refusal::SchemaValue:
		case Refusal::SchemaType:
		case Refusal::SchemaReference:
		case Refusal::SchemaFormat:
		case Refusal::SchemaOneOf:
		case Refusal::SchemaMultipleOf:
		case Refusal::SchemaBound:
		case Refusal::Empty:
		case Refusal::SchemaOpenArgument:
		case Refusal::SchemaStringUnion:
		case Refusal::SchemaRecursiveArgument:
		case Refusal::ToolParameters:
		case Refusal::ToolName:
		case Refusal::ToolDuplicate:
		case Refusal::ToolUnknown:
		case Refusal::LarkSyntax:
		case Refusal::LarkUndefined:
		case Refusal::LarkRecursiveTerminal:
		case Refusal::LarkNoStart:
		case Refusal::LarkPriority:
		case Refusal::LarkTemplate:
		case Refusal::LarkDirective:
		case Refusal::LarkToken:
		case Refusal::LarkOption:
		case Refusal::LarkOperator:
		case Refusal::Regex:
		case Refusal::SchemaTooLarge:
		case Refusal::SchemaTooMany:
		case Refusal::LarkTooLarge:
		case Refusal::AutomatonTooLarge:
		case Refusal::GrammarTooLarge:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] auto operator==(GrammarError const&) const -> bool = default;
};

[[nodiscard]] inline auto refuse(Refusal refusal, std::string at) -> std::unexpected<GrammarError> {
	return std::unexpected(GrammarError{.refusal = refusal, .at = std::move(at), .regex = std::nullopt});
}

[[nodiscard]] inline auto refuse_regex(regex::RegexError error, std::string at) -> std::unexpected<GrammarError> {
	return std::unexpected(GrammarError{.refusal = Refusal::Regex, .at = std::move(at), .regex = error});
}

/** The compile-time bounds on a request's grammar: every input is refused past them, by name, before admission. */
export struct Bounds {
	/** The longest schema text, in bytes. */
	std::size_t schema_bytes;
	/** The deepest nesting of schemas within one schema. */
	std::size_t schema_depth;
	/** Properties across every object schema of a request. */
	std::size_t properties;
	/** Enum and const values across a request. */
	std::size_t enum_values;
	/** The longest lark text, in bytes. */
	std::size_t lark_bytes;
	/** Rules and terminals in one lark grammar. */
	std::size_t lark_rules;
	/** The longest regex text, in bytes. */
	std::size_t regex_bytes;
	/** Expression nodes across a grammar. */
	std::size_t expr_nodes;
	/** States of one lexeme's automaton. */
	std::size_t lexeme_states;
	/** States across every automaton of a grammar. */
	std::size_t states;
	/** Lexemes in a grammar. */
	std::size_t lexemes;
	/** Rules in a grammar. */
	std::size_t rules;
	/** Right-hand-side symbols across a grammar's rules. */
	std::size_t symbols;
};

/**
 * The bounds every request compiles under. They admit the schemas and grammars real tool definitions use (OpenAI's
 * own structured-output limits are 5,000 properties and 1,000 enum values) with room to spare, and keep the worst
 * compile a few milliseconds on the reactor thread.
 */
export inline constexpr Bounds request_bounds{
    .schema_bytes = std::size_t{256} << 10U,
    .schema_depth = 32,
    .properties = 5000,
    .enum_values = 2000,
    .lark_bytes = std::size_t{64} << 10U,
    .lark_rules = 1000,
    .regex_bytes = std::size_t{16} << 10U,
    .expr_nodes = std::size_t{1} << 20U,
    .lexeme_states = 10000,
    .states = 120000,
    .lexemes = 20000,
    .rules = 100000,
    .symbols = 400000,
};

/** A JSON value that could not be read back (a bound of the reader). */
[[nodiscard]] inline auto refuse_json(json::ReadError const& /*error*/, std::string at) -> std::unexpected<GrammarError> {
	return refuse(Refusal::SchemaTooLarge, std::move(at));
}

}
