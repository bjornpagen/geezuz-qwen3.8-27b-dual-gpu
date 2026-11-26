// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:arguments;

import std;
import engine.chat;
import engine.core;
import engine.regex;
import :error;
import :builder;
import :document;
import :strings;
import :schema;

namespace engine::grammar {

/** Python's `str.isspace`, which the template's `strip` removes around a boolean or null argument. */
[[nodiscard]] auto python_spaces() -> regex::CharClass {
	return regex::CharClass::of(std::to_array<regex::Range>({
	    {.low = 0x09, .high = 0x0d},
	    {.low = 0x1c, .high = 0x20},
	    {.low = 0x85, .high = 0x85},
	    {.low = 0xa0, .high = 0xa0},
	    {.low = 0x1680, .high = 0x1680},
	    {.low = 0x2000, .high = 0x200a},
	    {.low = 0x2028, .high = 0x2029},
	    {.low = 0x202f, .high = 0x202f},
	    {.low = 0x205f, .high = 0x205f},
	    {.low = 0x3000, .high = 0x3000},
	}));
}

/** The texts a cast reads as a boolean or null: those the string branch must leave out. */
[[nodiscard]] auto captured_words(regex::Store& store, std::span<chat::Cast const> casts) -> regex::Expr {
	auto words = std::vector<regex::Expr>{};
	if (std::ranges::contains(casts, chat::Cast::Null)) {
		words.push_back(store.literal("null"));
		words.push_back(store.literal("None"));
	}
	if (std::ranges::contains(casts, chat::Cast::Boolean)) {
		auto parsed = regex::parse(store, "(?i:true|false)|1|0", regex::Matching::Whole);
		core::invariant(parsed.has_value());
		words.push_back(*parsed);
	}
	auto const space = store.repeat(regex::characters(store, python_spaces()), 0, regex::unbounded);
	return store.sequence(std::array{space, store.any_of(words), space});
}

/** The cast that reads a JSON value of a type back as itself, and whether the list has it. */
[[nodiscard]] auto reads_back(std::span<chat::Cast const> casts, Type type) -> bool {
	switch (type) {
	case Type::Null:
		return std::ranges::contains(casts, chat::Cast::Null);
	case Type::Boolean:
		return std::ranges::contains(casts, chat::Cast::Boolean);
	case Type::Integer:
		return std::ranges::contains(casts, chat::Cast::Integer) || std::ranges::contains(casts, chat::Cast::Number);
	case Type::Number:
		return std::ranges::contains(casts, chat::Cast::Number);
	case Type::Object:
		return std::ranges::contains(casts, chat::Cast::Object);
	case Type::Array:
		return std::ranges::contains(casts, chat::Cast::Array);
	case Type::String:
		return false;
	}
	std::unreachable();
}

/** Whether a boolean cast comes before every numeric one: then it reads the numbers 0 and 1 as booleans. */
[[nodiscard]] auto booleans_first(std::span<chat::Cast const> casts) -> bool {
	auto const boolean = std::ranges::find(casts, chat::Cast::Boolean);
	auto const numeric = std::ranges::find_if(casts, [](chat::Cast cast) { return cast == chat::Cast::Integer || cast == chat::Cast::Number; });
	return boolean != casts.end() && boolean < numeric;
}

/** The markers the chat format reads inside a tool call: a parameter's text may contain none of them. */
inline constexpr auto call_markers = std::to_array<std::string_view>({"<tool_call>", "</tool_call>", "<parameter=", "</parameter>", "</function>"});

/** How a parameter's text ends, as the template writes it. */
inline constexpr std::string_view parameter_close = "\n</parameter>\n";

/**
 * The text of one argument, through its closing tag, such that the chat format's parser reads back a value the
 * parameter's schema admits. A string is written as itself; any other type as JSON, read back by the template's
 * cast for that type. A text two casts would read differently is left to the first, as the template does.
 */
[[nodiscard]] auto argument(Builder& builder, Value const& root, Value const& schema, std::span<chat::Cast const> casts, std::string const& path, Usage& usage)
    -> core::Result<Symbol, GrammarError> {
	auto& store = builder.store();
	auto const forbidden = builder.memo("call markers", [&store] { return any_literal(store, call_markers); });
	auto compiler = SchemaCompiler{builder, root, "", Place{.markers = call_markers, .readback = Readback::ChatFormat, .top_numbers_excluded = regex::nothing}, usage};
	auto strings = compiler.strings(schema, path, 0);
	if (!strings) {
		return std::unexpected(strings.error());
	}
	auto const numeric_or_container = std::ranges::any_of(casts, [](chat::Cast cast) { return cast != chat::Cast::Null && cast != chat::Cast::Boolean; });
	if (*strings && numeric_or_container) {
		return refuse(Refusal::SchemaStringUnion, path);
	}
	auto const out = builder.nonterminal();
	auto const close = builder.literal(parameter_close);
	if (*strings) {
		auto values = std::holds_alternative<AnyString>(**strings) ? builder.memo("text", [&store] { return regex::any_text(store); }) : std::get<SomeStrings>(**strings).values;
		if (!casts.empty()) {
			values = store.both(values, store.complement(captured_words(store, casts)));
		}
		auto const text = store.both(values, avoiding(store, forbidden));
		builder.rule(out, {builder.lexeme(store.concat(text, store.literal(parameter_close)), Mode::Lazy, std::nullopt)});
	}
	auto json_types = Types::none();
	for (auto const type : all_types) {
		if (type != Type::String && reads_back(casts, type)) {
			json_types = json_types.with(type);
		}
	}
	if (!json_types.empty()) {
		auto const excluded = booleans_first(casts) ? store.either(store.literal("0"), store.literal("1")) : regex::nothing;
		auto typed = SchemaCompiler{builder, root, "", Place{.markers = call_markers, .readback = Readback::ChatFormat, .top_numbers_excluded = excluded}, usage};
		auto value = typed.value(schema, path, json_types, 0);
		if (!value) {
			return value;
		}
		builder.rule(out, {*value, close});
	}
	if (!*strings && json_types.empty()) {
		// No value of the schema survives the template's casts: the parameter could never be written.
		return refuse(Refusal::Empty, path);
	}
	return out;
}

/** Whether the chat format can write a name between `=` and `>`: not empty, without `<`, `>` or a newline. */
[[nodiscard]] auto writable_name(std::string_view name) -> bool {
	return !name.empty() && !name.contains('<') && !name.contains('>') && !name.contains('\n');
}

/**
 * A function tool's parameters, as the template writes them inside a call: each named property in the schema's
 * order as `<parameter=NAME>\nTEXT\n</parameter>\n`, the required ones always, the others as the model chooses, and
 * then, when the schema allows other properties, more parameters of other names whose text stays a string.
 */
[[nodiscard]] auto parameters(Builder& builder, Value const& root, std::string_view schema_text, std::string const& path, Usage& usage) -> core::Result<Symbol, GrammarError> {
	// The casts are the chat format's own: what its parser will read each argument's text as.
	auto reader = SchemaReader::make();
	if (!reader) {
		return refuse_json(reader.error(), path);
	}
	auto const declared = chat::parameter_casts(*reader, schema_text);
	if (!declared) {
		return refuse_json(declared.error(), path);
	}
	auto const object = core::alternative<Object>(root.held);
	if (!object) {
		return refuse(Refusal::ToolParameters, path);
	}
	auto facets = read_facets(*object, path);
	if (!facets) {
		return std::unexpected(facets.error());
	}
	if (facets->ref || facets->choices || facets->values || (facets->types && *facets->types != Types::only(Type::Object)) || facets->nullable) {
		return refuse(Refusal::ToolParameters, path);
	}
	auto const properties = facets->properties.value_or(std::span<Member const>{});
	usage.properties += properties.size();
	if (usage.properties > builder.bounds().properties) {
		return refuse(Refusal::SchemaTooMany, path);
	}
	for (auto const& name : facets->required) {
		if (!std::ranges::contains(properties, name, &Member::name)) {
			return refuse(Refusal::SchemaKeyword, pointer(path, "required"));
		}
	}
	auto& store = builder.store();
	auto const count = properties.size();
	auto rest = std::vector<Nonterminal>(count + 1);
	rest[count] = builder.nonterminal();
	if (facets->extra == Extra::Closed) {
		builder.rule(rest[count], {});
	} else {
		// Other parameters stay strings: the template casts only the properties the schema names.
		auto values = std::optional<StringLanguage>{AnyString{}};
		if (facets->extra == Extra::Schema) {
			auto compiler = SchemaCompiler{builder, root, "", Place{.markers = {}, .readback = Readback::ChatFormat, .top_numbers_excluded = regex::nothing}, usage};
			auto read = compiler.strings(facets->extra_schema.value(), pointer(path, "additionalProperties"), 0);
			if (!read) {
				return std::unexpected(read.error());
			}
			values = *read;
		}
		auto names = std::vector<regex::Expr>{};
		for (auto const& property : properties) {
			names.push_back(store.literal(property.name));
		}
		auto const name_characters = regex::characters(store, regex::CharClass::all().minus(regex::CharClass::of(std::to_array<regex::Range>({
		                                                          {.low = '\n', .high = '\n'},
		                                                          {.low = '<', .high = '<'},
		                                                          {.low = '>', .high = '>'},
		                                                      }))));
		auto const other_names = store.both(store.repeat(name_characters, 1, regex::unbounded), store.complement(store.any_of(names)));
		auto const extra = builder.nonterminal();
		if (values) {
			auto const forbidden = any_literal(store, call_markers);
			auto const text = std::holds_alternative<AnyString>(*values) ? regex::any_text(store) : std::get<SomeStrings>(*values).values;
			auto const open = builder.lexeme(store.sequence(std::array{store.literal("<parameter="), other_names, store.literal(">\n")}), Mode::Greedy, std::nullopt);
			auto const value = builder.lexeme(store.concat(store.both(text, avoiding(store, forbidden)), store.literal(parameter_close)), Mode::Lazy, std::nullopt);
			builder.rule(extra, {open, value});
		}
		builder.rule(rest[count], {builder.star(extra)});
	}
	for (auto index = count; index > 0; --index) {
		auto const& property = properties[index - 1];
		auto const at = pointer(pointer(path, "properties"), property.name);
		if (!writable_name(property.name)) {
			return refuse(Refusal::ToolName, at);
		}
		auto const named = std::ranges::find(*declared, property.name, &chat::ParameterCasts::name);
		auto const casts = named == declared->end() ? std::span<chat::Cast const>{} : std::span<chat::Cast const>{named->casts};
		auto value = argument(builder, root, property.value, casts, at, usage);
		if (!value) {
			return value;
		}
		rest[index - 1] = builder.nonterminal();
		auto const open = builder.literal(std::format("<parameter={}>\n", property.name));
		builder.rule(rest[index - 1], {open, *value, rest[index]});
		if (!std::ranges::contains(facets->required, property.name)) {
			builder.rule(rest[index - 1], {rest[index]});
		}
	}
	return rest[0];
}

}
