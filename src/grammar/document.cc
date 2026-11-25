// SPDX-License-Identifier: AGPL-3.0-only
export module engine.grammar:document;

import std;
import engine.core;
import engine.json;
import :error;

namespace engine::grammar {

struct Value;

struct Null {
	[[nodiscard]] auto operator==(Null const&) const -> bool = default;
};

/** A number as written: its literal text, which a bound keeps exactly. */
struct Number {
	std::string literal;

	[[nodiscard]] auto operator==(Number const&) const -> bool = default;
};

struct Member;

using Array = std::vector<Value>;

struct Object {
	std::vector<Member> members;

	[[nodiscard]] auto find(std::string_view name) const -> std::optional<Value const&>;
	[[nodiscard]] auto operator==(Object const&) const -> bool;
};

/** A schema document's value, members in the order written. */
struct Value {
	std::variant<Null, bool, Number, std::string, Array, Object> held;

	[[nodiscard]] auto operator==(Value const&) const -> bool = default;
};

struct Member {
	std::string name;
	Value value;

	[[nodiscard]] auto operator==(Member const&) const -> bool = default;
};

auto Object::find(std::string_view name) const -> std::optional<Value const&> {
	auto const found = std::ranges::find(members, name, &Member::name);
	if (found == members.end()) {
		return std::nullopt;
	}
	return found->value;
}

auto Object::operator==(Object const& other) const -> bool {
	return members == other.members;
}

/** The bounds a schema is read under: the document is `request_bounds.schema_bytes`, its nesting twice the schema depth. */
inline constexpr json::Bounds schema_reading{
    .document_bytes = request_bounds.schema_bytes,
    .depth = request_bounds.schema_depth * 2,
    .string_bytes = request_bounds.schema_bytes,
    .members = request_bounds.properties,
    .number_chars = 128,
};

static_assert(json::Enforceable<schema_reading>);

using SchemaReader = json::Reader<schema_reading>;

[[nodiscard]] auto read_value(json::Value<schema_reading> value) -> core::Result<Value, json::ReadError> {
	auto const kind = value.kind();
	if (!kind) {
		return std::unexpected(kind.error());
	}
	switch (*kind) {
	case json::Kind::Null:
		return value.as_null().transform([] { return Value{.held = Null{}}; });
	case json::Kind::Boolean:
		return value.as_boolean().transform([](bool read) { return Value{.held = read}; });
	case json::Kind::Integer:
	case json::Kind::Real:
		return value.raw().transform([](std::string_view literal) { return Value{.held = Number{.literal = std::string{literal}}}; });
	case json::Kind::String:
		return value.as_string().transform([](std::string_view read) { return Value{.held = std::string{read}}; });
	case json::Kind::Array: {
		auto array = value.as_array();
		if (!array) {
			return std::unexpected(array.error());
		}
		auto elements = Array{};
		while (true) {
			auto element = array->next();
			if (!element) {
				return std::unexpected(element.error());
			}
			if (!*element) {
				return Value{.held = std::move(elements)};
			}
			auto read = read_value(std::move(**element));
			if (!read) {
				return read;
			}
			elements.push_back(std::move(*read));
		}
	}
	case json::Kind::Object: {
		auto object = value.as_object();
		if (!object) {
			return std::unexpected(object.error());
		}
		auto members = Object{};
		while (true) {
			auto member = object->next();
			if (!member) {
				return std::unexpected(member.error());
			}
			if (!*member) {
				return Value{.held = std::move(members)};
			}
			auto name = std::string{(*member)->name};
			auto read = read_value(std::move((*member)->value));
			if (!read) {
				return read;
			}
			members.members.push_back(Member{.name = std::move(name), .value = std::move(*read)});
		}
	}
	}
	std::unreachable();
}

/** Reads a schema's text whole, under the schema bounds. */
[[nodiscard]] auto read_document(std::string_view text, std::string_view at) -> core::Result<Value, GrammarError> {
	if (text.size() > request_bounds.schema_bytes) {
		return refuse(Refusal::SchemaTooLarge, std::string{at});
	}
	auto reader = SchemaReader::make();
	if (!reader) {
		return refuse_json(reader.error(), std::string{at});
	}
	auto root = reader->read_copy(text);
	if (!root) {
		return refuse_json(root.error(), std::string{at});
	}
	auto value = read_value(std::move(*root));
	if (!value) {
		return refuse_json(value.error(), std::string{at});
	}
	if (auto const finished = reader->finish(); !finished) {
		return refuse_json(finished.error(), std::string{at});
	}
	return std::move(*value);
}

/** Appends a string as JSON writes it: the quote, the backslash and the controls escaped, everything else as itself. */
auto put_json_string(std::string& out, std::string_view text) -> void {
	out.push_back('"');
	for (auto const byte : text) {
		switch (byte) {
		case '"':
			out.append("\\\"");
			break;
		case '\\':
			out.append("\\\\");
			break;
		case '\n':
			out.append("\\n");
			break;
		case '\r':
			out.append("\\r");
			break;
		case '\t':
			out.append("\\t");
			break;
		case '\b':
			out.append("\\b");
			break;
		case '\f':
			out.append("\\f");
			break;
		default:
			if (auto const code = std::bit_cast<std::uint8_t>(byte); code < 0x20U) {
				std::format_to(std::back_inserter(out), "\\u{:04x}", code);
			} else {
				out.push_back(byte);
			}
		}
	}
	out.push_back('"');
}

/** A value as JSON text, with `", "` and `": "` between parts as Python's `json.dumps` writes them. */
auto put_json(std::string& out, Value const& value) -> void {
	std::visit(
	    [&out](auto const& held) {
		    using Held = std::remove_cvref_t<decltype(held)>;
		    if constexpr (std::is_same_v<Held, Null>) {
			    out.append("null");
		    } else if constexpr (std::is_same_v<Held, bool>) {
			    out.append(held ? "true" : "false");
		    } else if constexpr (std::is_same_v<Held, Number>) {
			    out.append(held.literal);
		    } else if constexpr (std::is_same_v<Held, std::string>) {
			    put_json_string(out, held);
		    } else if constexpr (std::is_same_v<Held, Array>) {
			    out.push_back('[');
			    for (auto const [index, element] : std::views::enumerate(held)) {
				    out.append(index == 0 ? "" : ", ");
				    put_json(out, element);
			    }
			    out.push_back(']');
		    } else {
			    out.push_back('{');
			    for (auto const [index, member] : std::views::enumerate(held.members)) {
				    out.append(index == 0 ? "" : ", ");
				    put_json_string(out, member.name);
				    out.append(": ");
				    put_json(out, member.value);
			    }
			    out.push_back('}');
		    }
	    },
	    value.held);
}

[[nodiscard]] auto json_text(Value const& value) -> std::string {
	auto out = std::string{};
	put_json(out, value);
	return out;
}

}
