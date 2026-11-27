// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:schema;

import std;
import engine.core;
import engine.regex;
import :error;
import :builder;
import :document;
import :numbers;
import :strings;

namespace engine::grammar {

/** A JSON type. Integers are numbers too; the distinction is a schema's. */
enum class Type : std::uint8_t {
	Null,
	Boolean,
	Integer,
	Number,
	String,
	Array,
	Object,
};

inline constexpr auto all_types = std::to_array<Type>({Type::Null, Type::Boolean, Type::Integer, Type::Number, Type::String, Type::Array, Type::Object});

/** A set of JSON types. */
struct Types {
	std::uint8_t bits;

	[[nodiscard]] static constexpr auto none() -> Types {
		return Types{.bits = 0};
	}

	[[nodiscard]] static constexpr auto all() -> Types {
		return Types{.bits = 0x7f};
	}

	[[nodiscard]] static constexpr auto only(Type type) -> Types {
		return Types{.bits = static_cast<std::uint8_t>(1U << std::to_underlying(type))};
	}

	[[nodiscard]] constexpr auto contains(Type type) const -> bool {
		return ((unsigned{bits} >> std::to_underlying(type)) & 1U) != 0;
	}

	[[nodiscard]] constexpr auto with(Type type) const -> Types {
		return Types{.bits = static_cast<std::uint8_t>(bits | only(type).bits)};
	}

	[[nodiscard]] constexpr auto meet(Types other) const -> Types {
		return Types{.bits = static_cast<std::uint8_t>(bits & other.bits)};
	}

	[[nodiscard]] constexpr auto empty() const -> bool {
		return bits == 0;
	}

	[[nodiscard]] constexpr auto operator==(Types const&) const -> bool = default;
};

[[nodiscard]] auto type_named(std::string_view name) -> std::optional<Type> {
	auto const names = std::to_array<std::pair<std::string_view, Type>>({
	    {"null", Type::Null},
	    {"boolean", Type::Boolean},
	    {"integer", Type::Integer},
	    {"number", Type::Number},
	    {"string", Type::String},
	    {"array", Type::Array},
	    {"object", Type::Object},
	});
	auto const found = std::ranges::find(names, name, &std::pair<std::string_view, Type>::first);
	if (found == names.end()) {
		return std::nullopt;
	}
	return found->second;
}

/** The type a JSON value has; a number is an integer when its value is whole. */
[[nodiscard]] auto type_of(Value const& value) -> Type {
	return std::visit(
	    [](auto const& held) -> Type {
		    using Held = std::remove_cvref_t<decltype(held)>;
		    if constexpr (std::is_same_v<Held, Null>) {
			    return Type::Null;
		    } else if constexpr (std::is_same_v<Held, bool>) {
			    return Type::Boolean;
		    } else if constexpr (std::is_same_v<Held, Number>) {
			    auto const decimal = read_decimal(held.literal);
			    return decimal && decimal->exponent >= 0 ? Type::Integer : Type::Number;
		    } else if constexpr (std::is_same_v<Held, std::string>) {
			    return Type::String;
		    } else if constexpr (std::is_same_v<Held, Array>) {
			    return Type::Array;
		    } else {
			    return Type::Object;
		    }
	    },
	    value.held);
}

/** Whether a schema type set admits a value of type `type` (an integer is a number). */
[[nodiscard]] auto admits(Types types, Type type) -> bool {
	return types.contains(type) || (type == Type::Integer && types.contains(Type::Number));
}

/** Who reads a JSON text back, which decides what the grammar may let it hold. */
enum class Readback : std::uint8_t {
	/** The client: objects may hold members their schema does not name, and values may nest without bound. */
	Client,
	/**
	 * The chat format's argument reader, which refuses repeated keys and nests at most 64 deep: a grammar cannot keep
	 * the names of members it does not know apart, nor count nesting through a recursive schema, so open objects and
	 * recursive schemas are refused.
	 */
	ChatFormat,
};

/** Where a JSON text is written: what its strings may not contain, and who reads it back. */
struct Place {
	/** Words no string of the text may contain. */
	std::span<std::string_view const> markers;
	Readback readback;
	/** Texts a number at the top level may not be: those an earlier cast would read as another type. */
	regex::Expr top_numbers_excluded;
};

/** What `additionalProperties` says. */
enum class Extra : std::uint8_t {
	/** Absent: members past the named ones may hold any value. */
	Unspecified,
	Closed,
	Schema,
};

enum class Union : std::uint8_t {
	AnyOf,
	OneOf,
};

/** A schema object's keywords, read and checked once. */
struct Facets {
	std::optional<Types> types;
	bool nullable = false;
	std::optional<std::vector<Value>> values;
	std::optional<std::span<Value const>> choices;
	Union choice_kind = Union::AnyOf;
	std::optional<std::string> ref;
	std::optional<std::uint32_t> min_length;
	std::optional<std::uint32_t> max_length;
	std::optional<std::string> pattern;
	std::optional<Format> format;
	std::optional<Bound> minimum;
	std::optional<Bound> maximum;
	std::optional<Value const&> items;
	std::optional<std::uint32_t> min_items;
	std::optional<std::uint32_t> max_items;
	std::optional<std::span<Member const>> properties;
	std::vector<std::string> required;
	Extra extra = Extra::Unspecified;
	std::optional<Value const&> extra_schema;
	/** Keywords that constrain a value, beyond `type`, `nullable` and annotations. */
	std::vector<std::string_view> constraints;
};

/** Keywords that annotate and never constrain: read past. */
inline constexpr auto annotations = std::to_array<std::string_view>({
    "title", "description", "default", "examples", "$schema", "$id", "$comment", "deprecated", "readOnly", "writeOnly", "$defs", "definitions",
});

[[nodiscard]] auto as_count(Value const& value) -> std::optional<std::uint32_t> {
	auto const number = core::alternative<Number>(value.held);
	if (!number) {
		return std::nullopt;
	}
	auto const decimal = read_decimal(number->literal);
	if (!decimal || decimal->negative || decimal->exponent < 0) {
		return std::nullopt;
	}
	auto const parts = split(*decimal);
	if (!parts || parts->whole.size() > 9) {
		return std::nullopt;
	}
	return core::narrow<std::uint32_t>(std::stoul(parts->whole));
}

[[nodiscard]] auto pointer(std::string const& path, std::string_view key) -> std::string {
	auto out = path + "/";
	for (auto const c : key) {
		out.append(c == '~' ? "~0" : c == '/' ? "~1" : std::string(1, c));
	}
	return out;
}

/** Reads a schema object's keywords; refuses one the engine does not implement, or one of the wrong shape. */
[[nodiscard]] auto read_facets(Object const& schema, std::string const& path) -> core::Result<Facets, GrammarError> {
	auto facets = Facets{};
	auto const bad = [&path](std::string_view key) { return refuse(Refusal::SchemaValue, pointer(path, key)); };
	auto exclusive_minimum = std::optional<Value const&>{};
	auto exclusive_maximum = std::optional<Value const&>{};
	auto minimum = std::optional<Value const&>{};
	auto maximum = std::optional<Value const&>{};
	for (auto const& [key, value] : schema.members) {
		if (std::ranges::contains(annotations, key)) {
			continue;
		}
		if (key == "type") {
			auto types = Types::none();
			auto const add = [&types](Value const& name) -> bool {
				auto const text = core::alternative<std::string>(name.held);
				auto const type = text ? type_named(*text) : std::nullopt;
				if (!type) {
					return false;
				}
				types = types.with(*type);
				return true;
			};
			if (auto const list = core::alternative<Array>(value.held)) {
				if (list->empty() || !std::ranges::all_of(*list, add)) {
					return refuse(Refusal::SchemaType, pointer(path, key));
				}
			} else if (!add(value)) {
				return refuse(Refusal::SchemaType, pointer(path, key));
			}
			facets.types = types;
		} else if (key == "nullable") {
			auto const flag = core::alternative<bool>(value.held);
			if (!flag) {
				return bad(key);
			}
			facets.nullable = *flag;
		} else if (key == "enum" || key == "const") {
			auto members = std::vector<Value>{};
			if (key == "const") {
				members.push_back(value);
			} else if (auto const list = core::alternative<Array>(value.held)) {
				members = *list;
			} else {
				return bad(key);
			}
			if (facets.values) {
				// Both: the values in both lists.
				std::erase_if(members, [&facets](Value const& member) { return !std::ranges::contains(*facets.values, member); });
			}
			facets.values = std::move(members);
			facets.constraints.push_back(key);
		} else if (key == "anyOf" || key == "oneOf") {
			auto const list = core::alternative<Array>(value.held);
			if (!list || list->empty() || facets.choices) {
				return bad(key);
			}
			facets.choices = std::span<Value const>{*list};
			facets.choice_kind = key == "anyOf" ? Union::AnyOf : Union::OneOf;
			facets.constraints.push_back(key);
		} else if (key == "$ref") {
			auto const text = core::alternative<std::string>(value.held);
			if (!text) {
				return bad(key);
			}
			facets.ref = *text;
			facets.constraints.push_back(key);
		} else if (key == "minLength" || key == "maxLength" || key == "minItems" || key == "maxItems") {
			auto const count = as_count(value);
			if (!count) {
				return bad(key);
			}
			(key == "minLength" ? facets.min_length : key == "maxLength" ? facets.max_length : key == "minItems" ? facets.min_items : facets.max_items) = *count;
			facets.constraints.push_back(key);
		} else if (key == "pattern") {
			auto const text = core::alternative<std::string>(value.held);
			if (!text) {
				return bad(key);
			}
			facets.pattern = *text;
			facets.constraints.push_back(key);
		} else if (key == "format") {
			auto const text = core::alternative<std::string>(value.held);
			if (!text) {
				return bad(key);
			}
			auto const format = format_named(*text);
			if (!format) {
				return refuse(Refusal::SchemaFormat, pointer(path, key));
			}
			facets.format = *format;
			facets.constraints.push_back(key);
		} else if (key == "minimum" || key == "maximum") {
			if (!core::alternative<Number>(value.held)) {
				return bad(key);
			}
			(key == "minimum" ? minimum : maximum) = value;
			facets.constraints.push_back(key);
		} else if (key == "exclusiveMinimum" || key == "exclusiveMaximum") {
			if (!core::alternative<Number>(value.held) && !core::alternative<bool>(value.held)) {
				return bad(key);
			}
			(key == "exclusiveMinimum" ? exclusive_minimum : exclusive_maximum) = value;
			facets.constraints.push_back(key);
		} else if (key == "multipleOf") {
			return refuse(Refusal::SchemaMultipleOf, pointer(path, key));
		} else if (key == "items") {
			if (!core::alternative<Object>(value.held) && !core::alternative<bool>(value.held)) {
				return bad(key);
			}
			facets.items = value;
			facets.constraints.push_back(key);
		} else if (key == "properties") {
			auto const members = core::alternative<Object>(value.held);
			if (!members) {
				return bad(key);
			}
			facets.properties = std::span<Member const>{members->members};
			facets.constraints.push_back(key);
		} else if (key == "required") {
			auto const list = core::alternative<Array>(value.held);
			if (!list) {
				return bad(key);
			}
			for (auto const& name : *list) {
				auto const text = core::alternative<std::string>(name.held);
				if (!text) {
					return bad(key);
				}
				facets.required.push_back(*text);
			}
			facets.constraints.push_back(key);
		} else if (key == "additionalProperties") {
			if (auto const flag = core::alternative<bool>(value.held)) {
				facets.extra = *flag ? Extra::Unspecified : Extra::Closed;
			} else if (core::alternative<Object>(value.held)) {
				facets.extra = Extra::Schema;
				facets.extra_schema = value;
			} else {
				return bad(key);
			}
			facets.constraints.push_back(key);
		} else {
			return refuse(Refusal::SchemaKeyword, pointer(path, key));
		}
	}
	// A boolean exclusive bound (draft 4) marks the plain bound; a numeric one (draft 6 on) is a bound itself.
	auto const bound = [&](std::optional<Value const&> plain, std::optional<Value const&> exclusive, std::string_view key) -> core::Result<std::optional<Bound>, GrammarError> {
		auto out = std::optional<Bound>{};
		if (plain) {
			out = Bound{.literal = std::get<Number>(plain->held).literal, .exclusive = false};
		}
		if (exclusive) {
			if (auto const flag = core::alternative<bool>(exclusive->held)) {
				if (!out) {
					return refuse(Refusal::SchemaValue, pointer(path, key));
				}
				out->exclusive = *flag;
			} else if (out) {
				// Both a bound and an exclusive bound: the tighter one, which needs comparing; refused instead.
				return refuse(Refusal::SchemaValue, pointer(path, key));
			} else {
				out = Bound{.literal = std::get<Number>(exclusive->held).literal, .exclusive = true};
			}
		}
		return out;
	};
	auto low = bound(minimum, exclusive_minimum, "exclusiveMinimum");
	if (!low) {
		return std::unexpected(low.error());
	}
	auto high = bound(maximum, exclusive_maximum, "exclusiveMaximum");
	if (!high) {
		return std::unexpected(high.error());
	}
	facets.minimum = *low;
	facets.maximum = *high;
	return facets;
}

/** Request-wide counts the bounds limit. */
struct Usage {
	std::size_t properties = 0;
	std::size_t enum_values = 0;
};

/** Strings a schema admits, as a language over the value's text. */
struct AnyString {};
struct SomeStrings {
	regex::Expr values;
};
/** Every string of `min` to `max` characters (max `regex::unbounded` for none), and no other constraint. */
struct BoundedStrings {
	std::uint32_t min;
	std::uint32_t max;
};
using StringLanguage = std::variant<AnyString, SomeStrings, BoundedStrings>;

/** A string language as an expression over the value's text: a length bound becomes counted characters. */
[[nodiscard]] auto values_of(regex::Store& store, StringLanguage const& language) -> regex::Expr {
	if (auto const bounded = core::alternative<BoundedStrings>(language)) {
		return store.repeat(regex::any_character(store), bounded->min, bounded->max);
	}
	if (auto const some = core::alternative<SomeStrings>(language)) {
		return some->values;
	}
	return regex::any_text(store);
}

/**
 * Compiles JSON schemas into a grammar's rules: each schema into a nonterminal deriving the JSON texts of exactly
 * the values it admits, written with `json_whitespace` between tokens and numbers as `numbers` writes them. Lives for
 * one compile, borrowing its builder and the schema document.
 */
class SchemaCompiler {
public:
	SchemaCompiler(Builder& builder, Value const& root, std::string root_path, Place place, Usage& usage)
	    : builder_{builder}, store_{builder.store()}, root_{root}, root_path_{std::move(root_path)}, place_{place}, forbidden_{any_literal(builder.store(), place.markers)}, usage_{usage} {}

	/** The JSON texts of the values `schema` admits whose types are in `filter`. */
	[[nodiscard]] auto value(Value const& schema, std::string const& path, Types filter, std::size_t depth) -> core::Result<Symbol, GrammarError> {
		if (depth > builder_.bounds().schema_depth) {
			return refuse(Refusal::SchemaTooLarge, path);
		}
		if (auto const flag = core::alternative<bool>(schema.held)) {
			return *flag ? any(filter, path) : builder_.nonterminal();
		}
		auto const object = core::alternative<Object>(schema.held);
		if (!object) {
			return refuse(Refusal::SchemaShape, path);
		}
		auto facets = read_facets(*object, path);
		if (!facets) {
			return std::unexpected(facets.error());
		}
		if (facets->ref) {
			return reference(*facets, path, filter, depth);
		}
		if (facets->choices) {
			return choices(*facets, path, filter, depth);
		}
		auto types = facets->types.value_or(Types::all());
		if (facets->nullable) {
			types = types.with(Type::Null);
		}
		if (facets->values) {
			return enumeration(*facets, path, types.meet(filter));
		}
		types = types.meet(filter);
		auto const out = builder_.nonterminal();
		for (auto const type : all_types) {
			if (!types.contains(type) || (type == Type::Integer && types.contains(Type::Number))) {
				continue;
			}
			auto alternative = typed(*facets, path, type, depth);
			if (!alternative) {
				return std::unexpected(alternative.error());
			}
			if (*alternative) {
				builder_.rule(out, {**alternative});
			}
		}
		return out;
	}

	/** The strings `schema` admits, as a language over their text; nothing when it admits no string. */
	[[nodiscard]] auto strings(Value const& schema, std::string const& path, std::size_t depth) -> core::Result<std::optional<StringLanguage>, GrammarError> {
		if (depth > builder_.bounds().schema_depth) {
			return refuse(Refusal::SchemaTooLarge, path);
		}
		if (auto const flag = core::alternative<bool>(schema.held)) {
			return *flag ? std::optional<StringLanguage>{AnyString{}} : std::nullopt;
		}
		auto const object = core::alternative<Object>(schema.held);
		if (!object) {
			return refuse(Refusal::SchemaShape, path);
		}
		auto facets = read_facets(*object, path);
		if (!facets) {
			return std::unexpected(facets.error());
		}
		if (facets->ref) {
			auto target = resolve(*facets, path);
			if (!target) {
				return std::unexpected(target.error());
			}
			return strings(target->first, target->second, depth + 1);
		}
		if (facets->choices) {
			if (auto const extra = constrains_besides(*facets, {"anyOf", "oneOf"})) {
				return refuse(Refusal::SchemaKeyword, pointer(path, *extra));
			}
			auto languages = std::vector<regex::Expr>{};
			auto any_string = false;
			for (auto const [index, choice] : std::views::enumerate(*facets->choices)) {
				auto language = strings(choice, std::format("{}/{}/{}", path, facets->choice_kind == Union::AnyOf ? "anyOf" : "oneOf", index), depth + 1);
				if (!language) {
					return language;
				}
				if (!*language) {
					continue;
				}
				if (std::holds_alternative<AnyString>(**language)) {
					any_string = true;
				} else {
					// A union's choices are one lexeme: a length bound among them goes into its expression.
					languages.push_back(values_of(store_, **language));
				}
			}
			if (any_string) {
				return std::optional<StringLanguage>{AnyString{}};
			}
			if (languages.empty()) {
				return std::nullopt;
			}
			return std::optional<StringLanguage>{SomeStrings{.values = store_.any_of(languages)}};
		}
		auto types = facets->types.value_or(Types::all());
		if (facets->nullable) {
			types = types.with(Type::Null);
		}
		if (facets->values) {
			auto literals = std::vector<regex::Expr>{};
			for (auto const& member : *facets->values) {
				if (auto const text = core::alternative<std::string>(member.held); text && types.contains(Type::String)) {
					literals.push_back(store_.literal(*text));
				}
			}
			if (literals.empty()) {
				return std::nullopt;
			}
			return std::optional<StringLanguage>{SomeStrings{.values = store_.any_of(literals)}};
		}
		if (!types.contains(Type::String)) {
			return std::nullopt;
		}
		return string_facets(*facets, path);
	}

	/** The types a schema may admit, from its `type`, `nullable`, `enum`, `const` and unions; all when unknown. */
	[[nodiscard]] auto types_of(Value const& schema, std::size_t depth) -> Types {
		if (depth > builder_.bounds().schema_depth) {
			return Types::all();
		}
		if (auto const flag = core::alternative<bool>(schema.held)) {
			return *flag ? Types::all() : Types::none();
		}
		auto const object = core::alternative<Object>(schema.held);
		if (!object) {
			return Types::all();
		}
		auto facets = read_facets(*object, "");
		if (!facets) {
			return Types::all();
		}
		if (facets->ref) {
			auto target = resolve(*facets, "");
			return target ? types_of(target->first, depth + 1) : Types::all();
		}
		if (facets->choices) {
			auto out = Types::none();
			for (auto const& choice : *facets->choices) {
				out = Types{.bits = static_cast<std::uint8_t>(out.bits | types_of(choice, depth + 1).bits)};
			}
			return out;
		}
		auto types = facets->types.value_or(Types::all());
		if (facets->nullable) {
			types = types.with(Type::Null);
		}
		if (facets->values) {
			auto out = Types::none();
			for (auto const& member : *facets->values) {
				auto const type = type_of(member);
				if (admits(types, type)) {
					out = out.with(type);
				}
			}
			return out;
		}
		return types;
	}

	/** The JSON text of any value whose type is in `filter`. */
	[[nodiscard]] auto any(Types filter, std::string const& path) -> core::Result<Symbol, GrammarError> {
		auto const cached = std::ranges::find(any_, filter, &std::pair<Types, Nonterminal>::first);
		if (cached != any_.end()) {
			return cached->second;
		}
		if (place_.readback == Readback::ChatFormat && (filter.contains(Type::Object) || filter.contains(Type::Array))) {
			// Any array may hold any object, and any object is open.
			return refuse(Refusal::SchemaOpenArgument, path);
		}
		auto const out = builder_.nonterminal();
		any_.emplace_back(filter, out);
		++nested_;
		auto const everything = Types::all();
		for (auto const type : all_types) {
			if (!filter.contains(type) || type == Type::Integer) {
				continue;
			}
			auto const facets = Facets{};
			if (type == Type::Array) {
				auto element = any(everything, path);
				if (!element) {
					return element;
				}
				builder_.rule(out, {list(*element, 0, regex::unbounded)});
			} else if (type == Type::Object) {
				auto member_value = any(everything, path);
				if (!member_value) {
					return member_value;
				}
				builder_.rule(out, {open_object(*member_value)});
			} else {
				auto alternative = typed(facets, path, type, 0);
				if (!alternative) {
					return std::unexpected(alternative.error());
				}
				if (*alternative) {
					builder_.rule(out, {**alternative});
				}
			}
		}
		--nested_;
		return out;
	}

	/** A lexeme of `language` whose strings avoid the place's marker words. */
	[[nodiscard]] auto guarded(regex::Expr language) -> Symbol {
		return builder_.lexeme(store_.both(language, avoiding(store_, forbidden_)), Mode::Greedy, std::nullopt);
	}

	/** A lexeme of one text, then `after` (which holds no marker): none at all when the text holds a marker word. */
	[[nodiscard]] auto guarded_literal(std::string_view text, regex::Expr after) -> Symbol {
		if (std::ranges::any_of(place_.markers, [text](std::string_view marker) { return text.contains(marker); })) {
			return builder_.nonterminal();
		}
		return builder_.lexeme(store_.concat(store_.literal(text), after), Mode::Greedy, std::nullopt);
	}

	[[nodiscard]] auto place() const -> Place {
		return place_;
	}

private:
	/** A number lexeme's language: at the top level, without the texts an earlier cast would take. */
	[[nodiscard]] auto numeric(regex::Expr language) -> regex::Expr {
		return nested_ == 0 ? store_.both(language, store_.complement(place_.top_numbers_excluded)) : language;
	}

	/** A constraining keyword other than `allowed`, if the facets have one. */
	[[nodiscard]] static auto constrains_besides(Facets const& facets, std::initializer_list<std::string_view> allowed) -> std::optional<std::string_view> {
		for (auto const keyword : facets.constraints) {
			if (!std::ranges::contains(allowed, keyword)) {
				return keyword;
			}
		}
		if (facets.types || facets.nullable) {
			if (!std::ranges::contains(allowed, std::string_view{"type"})) {
				return "type";
			}
		}
		return std::nullopt;
	}

	/** The schema a `$ref` names, and its path. */
	[[nodiscard]] auto resolve(Facets const& facets, std::string const& path) const -> core::Result<std::pair<Value const&, std::string>, GrammarError> {
		auto const& ref = facets.ref.value();
		auto const at = pointer(path, "$ref");
		if (ref == "#") {
			return std::pair<Value const&, std::string>{root_, root_path_};
		}
		for (auto const prefix : {std::string_view{"#/$defs/"}, std::string_view{"#/definitions/"}}) {
			if (!ref.starts_with(prefix)) {
				continue;
			}
			auto name = std::string{};
			for (auto index = prefix.size(); index < ref.size(); ++index) {
				if (ref[index] == '/') {
					return refuse(Refusal::SchemaReference, at);
				}
				if (ref[index] == '~' && index + 1 < ref.size() && (ref[index + 1] == '0' || ref[index + 1] == '1')) {
					name.push_back(ref[index + 1] == '0' ? '~' : '/');
					++index;
				} else {
					name.push_back(ref[index]);
				}
			}
			auto const root_object = core::alternative<Object>(root_.held);
			auto const container = root_object ? root_object->find(prefix.substr(2, prefix.size() - 3)) : std::nullopt;
			auto const definitions = container ? core::alternative<Object>(container->held) : std::nullopt;
			auto const target = definitions ? definitions->find(name) : std::nullopt;
			if (!target) {
				return refuse(Refusal::SchemaReference, at);
			}
			return std::pair<Value const&, std::string>{*target, pointer(root_path_ + std::string{prefix.substr(1, prefix.size() - 2)}, name)};
		}
		return refuse(Refusal::SchemaReference, at);
	}

	[[nodiscard]] auto reference(Facets const& facets, std::string const& path, Types filter, std::size_t depth) -> core::Result<Symbol, GrammarError> {
		if (auto const extra = constrains_besides(facets, {"$ref"})) {
			return refuse(Refusal::SchemaKeyword, pointer(path, *extra));
		}
		auto target = resolve(facets, path);
		if (!target) {
			return std::unexpected(target.error());
		}
		auto const key = std::tuple{target->second, filter.bits, nested_ == 0};
		if (auto const found = references_.find(key); found != references_.end()) {
			if (place_.readback == Readback::ChatFormat && compiling_.contains(target->second)) {
				// A recursive argument could nest past the depth the chat format's reader takes, and come back a string.
				return refuse(Refusal::SchemaRecursiveArgument, pointer(path, "$ref"));
			}
			return found->second;
		}
		// The nonterminal exists before its body, so a recursive schema refers back to it.
		auto const out = builder_.nonterminal();
		references_.emplace(key, out);
		compiling_.insert(target->second);
		auto body = value(target->first, target->second, filter, depth + 1);
		compiling_.erase(target->second);
		if (!body) {
			return body;
		}
		builder_.rule(out, {*body});
		return out;
	}

	[[nodiscard]] auto choices(Facets const& facets, std::string const& path, Types filter, std::size_t depth) -> core::Result<Symbol, GrammarError> {
		if (auto const extra = constrains_besides(facets, {"anyOf", "oneOf"})) {
			return refuse(Refusal::SchemaKeyword, pointer(path, *extra));
		}
		auto const keyword = facets.choice_kind == Union::AnyOf ? "anyOf" : "oneOf";
		if (facets.choice_kind == Union::OneOf) {
			// Exactly one choice must match: exact only where no value can match two, which disjoint types prove.
			auto seen = Types::none();
			for (auto const& choice : facets.choices.value()) {
				auto const types = types_of(choice, depth + 1);
				auto const numeric = Types::only(Type::Integer).with(Type::Number);
				auto const overlaps = !seen.meet(types).empty() || (!seen.meet(numeric).empty() && !types.meet(numeric).empty());
				if (overlaps) {
					return refuse(Refusal::SchemaOneOf, pointer(path, keyword));
				}
				seen = Types{.bits = static_cast<std::uint8_t>(seen.bits | types.bits)};
			}
		}
		auto const out = builder_.nonterminal();
		for (auto const [index, choice] : std::views::enumerate(facets.choices.value())) {
			auto alternative = value(choice, std::format("{}/{}/{}", path, keyword, index), filter, depth + 1);
			if (!alternative) {
				return alternative;
			}
			builder_.rule(out, {*alternative});
		}
		return out;
	}

	[[nodiscard]] auto enumeration(Facets const& facets, std::string const& path, Types types) -> core::Result<Symbol, GrammarError> {
		if (auto const extra = constrains_besides(facets, {"enum", "const", "type"})) {
			return refuse(Refusal::SchemaKeyword, pointer(path, *extra));
		}
		usage_.enum_values += facets.values.value().size();
		if (usage_.enum_values > builder_.bounds().enum_values) {
			return refuse(Refusal::SchemaTooMany, pointer(path, "enum"));
		}
		auto const out = builder_.nonterminal();
		for (auto const& member : facets.values.value()) {
			if (!admits(types, type_of(member))) {
				continue;
			}
			if (place_.readback == Readback::ChatFormat && type_of(member) == Type::Number) {
				// A real is read back as a double and written in its shortest form: the literal may not survive.
				auto const decimal = read_decimal(std::get<Number>(member.held).literal);
				if (!decimal || decimal->digits.size() > 15) {
					return refuse(Refusal::SchemaBound, pointer(path, "enum"));
				}
			}
			auto const text = json_text(member);
			auto const numeric_member = type_of(member) == Type::Integer || type_of(member) == Type::Number;
			builder_.rule(out, {numeric_member ? builder_.lexeme(numeric(store_.literal(text)), Mode::Greedy, std::nullopt) : guarded_literal(text, regex::empty)});
		}
		return out;
	}

	/**
	 * The lexeme of the JSON text of a string value language. A bound on the length alone is counted in JSON
	 * characters beside the automaton (the quotes are two of them), so no bound grows the automaton.
	 */
	[[nodiscard]] auto string_lexeme(StringLanguage const& language) -> Symbol {
		if (std::holds_alternative<AnyString>(language)) {
			return guarded(builder_.memo("json string", [this] { return any_json_string(store_); }));
		}
		if (auto const bounded = core::alternative<BoundedStrings>(language)) {
			auto const avoid = avoiding(store_, forbidden_);
			auto const text = store_.both(builder_.memo("json string", [this] { return any_json_string(store_); }), avoid);
			auto const fallback = store_.both(json_strings(store_, bounded->min, bounded->max), avoid);
			auto const quotes = std::uint32_t{2};
			return builder_.counted(text, Mode::Greedy, Count{.unit = Unit::JsonCharacter, .min = widened(bounded->min, quotes), .max = widened(bounded->max, quotes), .fallback = fallback});
		}
		return guarded(plain_json_string(store_, std::get<SomeStrings>(language).values));
	}

	[[nodiscard]] auto string_facets(Facets const& facets, std::string const& path) -> core::Result<std::optional<StringLanguage>, GrammarError> {
		if (!facets.min_length && !facets.max_length && !facets.pattern && !facets.format) {
			return std::optional<StringLanguage>{AnyString{}};
		}
		auto parts = std::vector<regex::Expr>{regex::any_text(store_)};
		if (facets.min_length || facets.max_length) {
			auto const low = facets.min_length.value_or(0);
			auto const high = facets.max_length.value_or(regex::unbounded);
			if (low > high) {
				return std::optional<StringLanguage>{SomeStrings{.values = regex::nothing}};
			}
			if (low > regex::max_repetition || (high != regex::unbounded && high > regex::max_repetition)) {
				return refuse(Refusal::SchemaTooLarge, pointer(path, "maxLength"));
			}
			if (!facets.pattern && !facets.format) {
				return std::optional<StringLanguage>{BoundedStrings{.min = low, .max = high}};
			}
			// With a pattern or a format, the bound joins their expression: no count tracks it beside a pattern.
			parts.push_back(store_.repeat(regex::any_character(store_), low, high));
		}
		if (facets.pattern) {
			if (facets.pattern->size() > builder_.bounds().regex_bytes) {
				return refuse(Refusal::SchemaTooLarge, pointer(path, "pattern"));
			}
			auto parsed = regex::parse(store_, *facets.pattern, regex::Matching::Search);
			if (!parsed) {
				return refuse_regex(parsed.error(), pointer(path, "pattern"));
			}
			parts.push_back(*parsed);
		}
		if (facets.format) {
			parts.push_back(format_language(store_, *facets.format));
		}
		return std::optional<StringLanguage>{SomeStrings{.values = store_.all_of(parts)}};
	}

	/** One type's alternative of a schema without unions, references or enumerations; nothing when it admits none. */
	[[nodiscard]] auto typed(Facets const& facets, std::string const& path, Type type, std::size_t depth) -> core::Result<std::optional<Symbol>, GrammarError> {
		switch (type) {
		case Type::Null:
			return std::optional<Symbol>{builder_.literal("null")};
		case Type::Boolean: {
			auto const out = builder_.nonterminal();
			builder_.rule(out, {builder_.literal("true")});
			builder_.rule(out, {builder_.literal("false")});
			return std::optional<Symbol>{out};
		}
		case Type::Integer: {
			auto const bound = [](std::optional<Bound> const& value) { return value ? std::format("{}{}", value->exclusive ? "(" : "[", value->literal) : std::string{}; };
			auto const range = builder_.memo(std::format("integer {} {}", bound(facets.minimum), bound(facets.maximum)), [&] {
				return integer_range(store_, facets.minimum, facets.maximum).value_or(regex::nothing);
			});
			if (range == regex::nothing) {
				return std::optional<Symbol>{};
			}
			return std::optional<Symbol>{builder_.lexeme(numeric(range), Mode::Greedy, std::nullopt)};
		}
		case Type::Number: {
			if (!facets.minimum && !facets.maximum) {
				return std::optional<Symbol>{builder_.lexeme(numeric(builder_.memo("number", [this] { return any_number(store_); })), Mode::Greedy, std::nullopt)};
			}
			auto const range = real_range(store_, facets.minimum, facets.maximum);
			if (!range) {
				if (range.error() == RealFault::Precision) {
					return refuse(Refusal::SchemaBound, path);
				}
				return std::optional<Symbol>{};
			}
			return std::optional<Symbol>{builder_.lexeme(numeric(*range), Mode::Greedy, std::nullopt)};
		}
		case Type::String: {
			auto language = string_facets(facets, path);
			if (!language) {
				return std::unexpected(language.error());
			}
			return std::optional<Symbol>{string_lexeme(language->value())};
		}
		case Type::Array: {
			++nested_;
			auto element = facets.items ? value(*facets.items, pointer(path, "items"), Types::all(), depth + 1) : any(Types::all(), pointer(path, "items"));
			--nested_;
			if (!element) {
				return std::unexpected(element.error());
			}
			auto const low = facets.min_items.value_or(0);
			auto const high = facets.max_items.value_or(regex::unbounded);
			if (low > high) {
				return std::optional<Symbol>{};
			}
			return std::optional<Symbol>{list(*element, low, high)};
		}
		case Type::Object:
			return object(facets, path, depth);
		}
		std::unreachable();
	}

	/** `[ item, ... ]` with between `low` and `high` items. */
	[[nodiscard]] auto list(Symbol item, std::uint32_t low, std::uint32_t high) -> Symbol {
		auto const out = builder_.nonterminal();
		auto const open = builder_.lexeme(punctuation(store_, "", "[", " "), Mode::Greedy, std::nullopt);
		auto const close = builder_.lexeme(punctuation(store_, " ", "]", ""), Mode::Greedy, std::nullopt);
		if (high == 0) {
			builder_.rule(out, {open, close});
			return out;
		}
		auto const items = builder_.separated(item, comma(), low, high);
		builder_.rule(out, {open, items, close});
		return out;
	}

	[[nodiscard]] auto comma() -> Symbol {
		return builder_.lexeme(punctuation(store_, " ", ",", " "), Mode::Greedy, std::nullopt);
	}

	/** The lexeme of a member's name and the colon after it. */
	[[nodiscard]] auto key(regex::Expr name_text) -> Symbol {
		auto const out = builder_.nonterminal();
		builder_.rule(out, {guarded(name_text), colon()});
		return out;
	}

	/** The colon after a member's name: one lexeme every member shares, so names stay small automata. */
	[[nodiscard]] auto colon() -> Symbol {
		return builder_.lexeme(punctuation(store_, " ", ":", " "), Mode::Greedy, std::nullopt);
	}

	/** `{ "name": value, ... }` with any names: the object of an open schema without named properties. */
	[[nodiscard]] auto open_object(Symbol member_value) -> Symbol {
		auto const out = builder_.nonterminal();
		auto const member = builder_.nonterminal();
		builder_.rule(member, {key(any_json_string(store_)), member_value});
		auto const open = builder_.lexeme(punctuation(store_, "", "{", " "), Mode::Greedy, std::nullopt);
		auto const close = builder_.lexeme(punctuation(store_, " ", "}", ""), Mode::Greedy, std::nullopt);
		builder_.rule(out, {open, builder_.separated(member, comma(), 0, regex::unbounded), close});
		return out;
	}

	[[nodiscard]] auto object(Facets const& facets, std::string const& path, std::size_t depth) -> core::Result<std::optional<Symbol>, GrammarError> {
		auto const properties = facets.properties.value_or(std::span<Member const>{});
		usage_.properties += properties.size();
		if (usage_.properties > builder_.bounds().properties) {
			return refuse(Refusal::SchemaTooMany, pointer(path, "properties"));
		}
		auto const declared = [&properties](std::string_view name) { return std::ranges::contains(properties, name, &Member::name); };
		for (auto const& name : facets.required) {
			if (!declared(name)) {
				// A required member the schema does not describe: refused rather than modelled.
				return refuse(Refusal::SchemaKeyword, pointer(path, "required"));
			}
		}
		auto const open = facets.extra != Extra::Closed;
		if (open && place_.readback == Readback::ChatFormat) {
			return refuse(Refusal::SchemaOpenArgument, path);
		}
		auto const open_brace = builder_.lexeme(punctuation(store_, "", "{", " "), Mode::Greedy, std::nullopt);
		auto const close_brace = builder_.lexeme(punctuation(store_, " ", "}", ""), Mode::Greedy, std::nullopt);
		// rest[i][first]: the members from property i on, `first` when no member was written yet.
		auto const count = properties.size();
		auto rest = std::vector<std::array<Nonterminal, 2>>(count + 1);
		rest[count] = {builder_.nonterminal(), builder_.nonterminal()};
		if (open) {
			++nested_;
			auto extra_value = facets.extra == Extra::Schema ? value(facets.extra_schema.value(), pointer(path, "additionalProperties"), Types::all(), depth + 1)
			                                                  : any(Types::all(), pointer(path, "additionalProperties"));
			--nested_;
			if (!extra_value) {
				return std::unexpected(extra_value.error());
			}
			// Extra names are plain strings other than the named ones, so no member is named twice by an escape.
			auto names = std::vector<regex::Expr>{};
			for (auto const& property : properties) {
				names.push_back(store_.literal(property.name));
			}
			auto const extra_names = store_.both(regex::any_text(store_), store_.complement(store_.any_of(names)));
			auto const member = builder_.nonterminal();
			builder_.rule(member, {key(plain_json_string(store_, extra_names)), *extra_value});
			auto const with_comma = builder_.nonterminal();
			builder_.rule(with_comma, {comma(), member});
			auto const tail = builder_.star(with_comma);
			builder_.rule(rest[count][1], {});
			builder_.rule(rest[count][1], {member, tail});
			builder_.rule(rest[count][0], {tail});
		} else {
			builder_.rule(rest[count][0], {});
			builder_.rule(rest[count][1], {});
		}
		for (auto index = count; index > 0; --index) {
			auto const& property = properties[index - 1];
			++nested_;
			auto property_value = value(property.value, pointer(pointer(path, "properties"), property.name), Types::all(), depth + 1);
			--nested_;
			if (!property_value) {
				return std::unexpected(property_value.error());
			}
			auto name = std::string{};
			put_json_string(name, property.name);
			auto const member = builder_.nonterminal();
			builder_.rule(member, {guarded_literal(name, regex::empty), colon(), *property_value});
			auto const required = std::ranges::contains(facets.required, property.name);
			rest[index - 1] = {builder_.nonterminal(), builder_.nonterminal()};
			for (auto const first : {0, 1}) {
				auto const self = rest[index - 1][core::narrow<std::size_t>(first)];
				auto written = first == 1 ? std::vector<Symbol>{member, rest[index][0]} : std::vector<Symbol>{comma(), member, rest[index][0]};
				builder_.rule(self, std::move(written));
				if (!required) {
					builder_.rule(self, {rest[index][core::narrow<std::size_t>(first)]});
				}
			}
		}
		auto const out = builder_.nonterminal();
		builder_.rule(out, {open_brace, rest[0][1], close_brace});
		return std::optional<Symbol>{out};
	}

	Builder& builder_;
	regex::Store& store_;
	Value const& root_;
	std::string root_path_;
	Place place_;
	regex::Expr forbidden_;
	Usage& usage_;
	std::vector<std::pair<Types, Nonterminal>> any_;
	std::map<std::tuple<std::string, std::uint8_t, bool>, Nonterminal> references_;
	/** The definitions whose bodies are being compiled: a reference to one of them is recursion. */
	std::set<std::string> compiling_;
	/** Containers around the value being compiled: zero at the top level. */
	std::size_t nested_ = 0;
};

}
