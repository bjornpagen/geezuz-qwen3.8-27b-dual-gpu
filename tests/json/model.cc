// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json_model;

import std;
import engine.core;
import engine.json;

/**
 * The oracle side of the json tests: a plain recursive model of a JSON value, a serializer for it written
 * independently of the engine's writer, a generator, and the facts each bound measures. The generator keeps only the
 * harness's invariant (member names unique within an object, so a model is one document); everything the parser
 * decides stays generable.
 */
namespace engine::json_model {

export struct Model;

export using Elements = std::vector<Model>;
export using Members = std::vector<std::pair<std::string, Model>>;

export struct Model {
	std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Elements, Members> value;
};

/** How the serializer spells strings: UTF-8 as is, or every non-ASCII code point as `\u` escapes. */
export enum class Spelling : std::uint8_t {
	Raw,
	Escaped,
};

/** Where the serializer puts whitespace: nowhere, or a random run of the four JSON whitespace bytes. */
export enum class Spacing : std::uint8_t {
	Compact,
	Random,
};

export struct Style {
	Spelling spelling;
	Spacing spacing;
};

/** The independent UTF-8 encoder: arithmetic on the scalar value, byte by byte. */
export [[nodiscard]] auto utf8(char32_t scalar) -> std::string {
	auto const value = std::uint32_t{scalar};
	auto bytes = std::string{};
	auto const put = [&bytes](std::uint32_t byte) { bytes.push_back(std::bit_cast<char>(static_cast<std::uint8_t>(byte))); };
	if (value < 0x80) {
		put(value);
	} else if (value < 0x800) {
		put(0xC0 + value / 64);
		put(0x80 + value % 64);
	} else if (value < 0x10000) {
		put(0xE0 + value / 4096);
		put(0x80 + value / 64 % 64);
		put(0x80 + value % 64);
	} else {
		put(0xF0 + value / 262144);
		put(0x80 + value / 4096 % 64);
		put(0x80 + value / 64 % 64);
		put(0x80 + value % 64);
	}
	return bytes;
}

/**
 * The independent UTF-8 decoder: reads the scalar value a sequence claims by arithmetic, then accepts it only if it
 * is a scalar value (not a surrogate, at most U+10FFFF) and the sequence is its shortest encoding. Returns each
 * scalar, or nullopt for an ill-formed text.
 */
export [[nodiscard]] auto decode_utf8(std::string_view text) -> std::optional<std::vector<char32_t>> {
	auto scalars = std::vector<char32_t>{};
	auto at = std::size_t{0};
	while (at < text.size()) {
		auto const lead = std::uint32_t{std::bit_cast<std::uint8_t>(text[at])};
		auto const length = lead < 0x80 ? 1U : lead >= 0xF0 && lead < 0xF8 ? 4U : lead >= 0xE0 && lead < 0xF0 ? 3U : lead >= 0xC0 && lead < 0xE0 ? 2U : 0U;
		if (length == 0 || at + length > text.size()) {
			return std::nullopt;
		}
		auto value = length == 1 ? lead : length == 2 ? lead & 0x1FU : length == 3 ? lead & 0x0FU : lead & 0x07U;
		for (auto index = std::size_t{1}; index < length; ++index) {
			auto const byte = std::uint32_t{std::bit_cast<std::uint8_t>(text[at + index])};
			if ((byte & 0xC0U) != 0x80U) {
				return std::nullopt;
			}
			value = (value << 6U) | (byte & 0x3FU);
		}
		if (value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF) || utf8(char32_t{value}).size() != length) {
			return std::nullopt;
		}
		scalars.push_back(char32_t{value});
		at += length;
	}
	return scalars;
}

/** The text the serializer writes for a number: shortest round trip, with `.0` on a double that has no fraction. */
export [[nodiscard]] auto number_text(Model const& model) -> std::string {
	auto digits = std::array<char, 40>{};
	auto const written = std::holds_alternative<double>(model.value)
	                         ? std::to_chars(digits.data(), std::to_address(digits.end()), std::get<double>(model.value))
	                         : std::to_chars(digits.data(), std::to_address(digits.end()), std::get<std::int64_t>(model.value));
	auto text = std::string{digits.data(), written.ptr};
	if (std::holds_alternative<double>(model.value) && text.find_first_of(".e") == std::string::npos) {
		text += ".0";
	}
	return text;
}

auto put_space(std::string& out, Style style, std::mt19937_64& rng) -> void {
	if (style.spacing == Spacing::Compact) {
		return;
	}
	auto constexpr spaces = std::string_view{" \t\n\r"};
	for (auto count = rng() % 3; count > 0; --count) {
		out.push_back(spaces[rng() % spaces.size()]);
	}
}

auto put_unit(std::string& out, std::uint32_t unit) -> void {
	out += std::format("\\u{:04X}", unit);
}

/** Writes a JSON string; controls take a short escape where one exists, so both escape forms are exercised. */
export auto put_string(std::string& out, std::string_view text, Spelling spelling) -> void {
	out.push_back('"');
	auto const scalars = decode_utf8(text);
	for (auto const scalar : scalars.value()) {
		auto const value = std::uint32_t{scalar};
		if (value == '"' || value == '\\') {
			out.push_back('\\');
			out.push_back(static_cast<char>(value));
		} else if (value == '\n') {
			out += "\\n";
		} else if (value == '\t') {
			out += "\\t";
		} else if (value < 0x20 || (value >= 0x80 && value < 0x10000 && spelling == Spelling::Escaped)) {
			put_unit(out, value);
		} else if (value < 0x80 || spelling == Spelling::Raw) {
			out += utf8(scalar);
		} else {
			put_unit(out, 0xD800 + ((value - 0x10000) >> 10U));
			put_unit(out, 0xDC00 + ((value - 0x10000) & 0x3FFU));
		}
	}
	out.push_back('"');
}

export auto serialize_into(std::string& out, Model const& model, Style style, std::mt19937_64& rng) -> void {
	put_space(out, style, rng);
	if (std::holds_alternative<std::nullptr_t>(model.value)) {
		out += "null";
	} else if (std::holds_alternative<bool>(model.value)) {
		out += std::get<bool>(model.value) ? "true" : "false";
	} else if (std::holds_alternative<std::string>(model.value)) {
		put_string(out, std::get<std::string>(model.value), style.spelling);
	} else if (std::holds_alternative<Elements>(model.value)) {
		out.push_back('[');
		auto first = true;
		for (auto const& element : std::get<Elements>(model.value)) {
			if (!first) {
				out.push_back(',');
			}
			first = false;
			serialize_into(out, element, style, rng);
		}
		put_space(out, style, rng);
		out.push_back(']');
	} else if (std::holds_alternative<Members>(model.value)) {
		out.push_back('{');
		auto first = true;
		for (auto const& [name, member] : std::get<Members>(model.value)) {
			if (!first) {
				out.push_back(',');
			}
			first = false;
			put_space(out, style, rng);
			put_string(out, name, style.spelling);
			put_space(out, style, rng);
			out.push_back(':');
			serialize_into(out, member, style, rng);
		}
		put_space(out, style, rng);
		out.push_back('}');
	} else {
		out += number_text(model);
	}
	put_space(out, style, rng);
}

export [[nodiscard]] auto serialize(Model const& model, Style style, std::mt19937_64& rng) -> std::string {
	auto out = std::string{};
	serialize_into(out, model, style, rng);
	return out;
}

/** Two models are equal when their kinds, texts and integers agree and their doubles have the same bits. */
export [[nodiscard]] auto same(Model const& left, Model const& right) -> bool {
	if (left.value.index() != right.value.index()) {
		return false;
	}
	if (std::holds_alternative<double>(left.value)) {
		return std::bit_cast<std::uint64_t>(std::get<double>(left.value)) == std::bit_cast<std::uint64_t>(std::get<double>(right.value));
	}
	if (std::holds_alternative<Elements>(left.value)) {
		return std::ranges::equal(std::get<Elements>(left.value), std::get<Elements>(right.value), same);
	}
	if (std::holds_alternative<Members>(left.value)) {
		return std::ranges::equal(std::get<Members>(left.value), std::get<Members>(right.value), [](auto const& one, auto const& other) {
			return one.first == other.first && same(one.second, other.second);
		});
	}
	if (std::holds_alternative<bool>(left.value)) {
		return std::get<bool>(left.value) == std::get<bool>(right.value);
	}
	if (std::holds_alternative<std::int64_t>(left.value)) {
		return std::get<std::int64_t>(left.value) == std::get<std::int64_t>(right.value);
	}
	if (std::holds_alternative<std::string>(left.value)) {
		return std::get<std::string>(left.value) == std::get<std::string>(right.value);
	}
	return true;
}

/** Reads a whole value through the engine's typed access, checking every part of it, into a model. */
export template<json::Bounds B>
    requires json::Enforceable<B>
[[nodiscard]] auto model_of(json::Value<B> value) -> engine::core::Result<Model, json::ReadError> {
	auto const kind = value.kind();
	if (!kind) {
		return std::unexpected(kind.error());
	}
	auto const wrap = [](auto read) -> engine::core::Result<Model, json::ReadError> {
		if (!read) {
			return std::unexpected(read.error());
		}
		return Model{.value = *read};
	};
	switch (*kind) {
	case json::Kind::Null: {
		auto const null = value.as_null();
		if (!null) {
			return std::unexpected(null.error());
		}
		return Model{.value = nullptr};
	}
	case json::Kind::Boolean:
		return wrap(value.as_boolean());
	case json::Kind::Integer:
		return wrap(value.as_integer());
	case json::Kind::Real:
		return wrap(value.as_real());
	case json::Kind::String: {
		auto const text = value.as_string();
		if (!text) {
			return std::unexpected(text.error());
		}
		return Model{.value = std::string{*text}};
	}
	case json::Kind::Array: {
		auto array = value.as_array();
		if (!array) {
			return std::unexpected(array.error());
		}
		auto elements = Elements{};
		while (true) {
			auto element = array->next();
			if (!element) {
				return std::unexpected(element.error());
			}
			if (!*element) {
				return Model{.value = std::move(elements)};
			}
			auto inner = model_of(std::move(**element));
			if (!inner) {
				return inner;
			}
			elements.push_back(std::move(*inner));
		}
	}
	case json::Kind::Object: {
		auto object = value.as_object();
		if (!object) {
			return std::unexpected(object.error());
		}
		auto members = Members{};
		while (true) {
			auto member = object->next();
			if (!member) {
				return std::unexpected(member.error());
			}
			if (!*member) {
				return Model{.value = std::move(members)};
			}
			auto name = std::string{(*member)->name};
			auto inner = model_of(std::move((*member)->value));
			if (!inner) {
				return inner;
			}
			members.emplace_back(std::move(name), std::move(*inner));
		}
	}
	}
	std::unreachable();
}

/** Reads a whole text strictly into a model: every part checked, nothing after the root. */
export template<json::Bounds B>
    requires json::Enforceable<B>
[[nodiscard]] auto read_model(json::Reader<B>& reader, std::string_view text) -> engine::core::Result<Model, json::ReadError> {
	auto root = reader.read_copy(text);
	if (!root) {
		return std::unexpected(root.error());
	}
	auto model = model_of(std::move(*root));
	if (!model) {
		return model;
	}
	if (auto const finished = reader.finish(); !finished) {
		return std::unexpected(finished.error());
	}
	return model;
}

/** The parse fault of a failed read, or nothing when it failed for another reason or did not fail. */
export [[nodiscard]] auto fault_of(json::ReadError const& error) -> std::optional<json::ParseFault> {
	if (std::holds_alternative<json::ParseError>(error.cause)) {
		return std::get<json::ParseError>(error.cause).fault;
	}
	return std::nullopt;
}

/** Code points chosen to sit on every boundary the escapes and UTF-8 have, plus the JSON specials. */
constexpr auto edge_points = std::array<char32_t, 24>{
    U'"',     U'\\',    U'/',     U'\0',     U'\x01',    U'\b',      U'\f',     U'\n',
    U'\r',    U'\t',    U'\x1F',  U'\x7F',   U'\u0080',  U'߿',  U'ࠀ', U'퟿',
    U'', U'�', U'￿', U' ', U'\U00010000', U'\U0001F600', U'\U0010FFFF', U' ',
};

export [[nodiscard]] auto random_scalar(std::mt19937_64& rng) -> char32_t {
	switch (rng() % 4) {
	case 0:
		return edge_points[rng() % edge_points.size()];
	case 1:
		return char32_t{static_cast<std::uint32_t>(0x20 + rng() % 0x5F)};
	case 2: {
		auto const value = static_cast<std::uint32_t>(rng() % 0xF800);
		return char32_t{value < 0xD800 ? value : value + 0x800};
	}
	default:
		return char32_t{static_cast<std::uint32_t>(0x10000 + rng() % 0x100000)};
	}
}

export [[nodiscard]] auto random_text(std::mt19937_64& rng, std::size_t max_scalars) -> std::string {
	auto text = std::string{};
	for (auto count = rng() % (max_scalars + 1); count > 0; --count) {
		text += utf8(random_scalar(rng));
	}
	return text;
}

constexpr auto edge_integers = std::array<std::int64_t, 8>{
    0, -1, 1, std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max(),
    std::numeric_limits<std::int64_t>::min() + 1, std::numeric_limits<std::int64_t>::max() - 1, 9007199254740993,
};

constexpr auto edge_doubles = std::array<double, 9>{
    0.0, -0.0, 0.1, 1.0, -1.5, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 9007199254740993.0,
};

export [[nodiscard]] auto random_finite(std::mt19937_64& rng) -> double {
	while (true) {
		auto const value = std::bit_cast<double>(rng());
		if (std::isfinite(value)) {
			return value;
		}
	}
}

/** The generator's own settings, drawn per document so runs explore different corners. */
export struct Shape {
	std::size_t depth;
	std::size_t width;
	std::size_t text;
};

export [[nodiscard]] auto random_shape(std::mt19937_64& rng) -> Shape {
	return Shape{.depth = 1 + rng() % 6, .width = 1 + rng() % 8, .text = rng() % 12};
}

export [[nodiscard]] auto generate(std::mt19937_64& rng, Shape shape) -> Model;

auto generate_scalar(std::mt19937_64& rng, Shape shape) -> Model {
	switch (rng() % 9) {
	case 0:
		return Model{.value = nullptr};
	case 1:
		return Model{.value = (rng() % 2) == 0};
	case 2:
		return Model{.value = edge_integers[rng() % edge_integers.size()]};
	case 3:
		return Model{.value = std::bit_cast<std::int64_t>(rng()) / static_cast<std::int64_t>(std::uint64_t{1} << (rng() % 63))};
	case 4:
		return Model{.value = edge_doubles[rng() % edge_doubles.size()]};
	case 5:
		return Model{.value = random_finite(rng)};
	default:
		return Model{.value = random_text(rng, shape.text)};
	}
}

auto generate(std::mt19937_64& rng, Shape shape) -> Model {
	if (shape.depth == 0) {
		return generate_scalar(rng, shape);
	}
	auto const inner = Shape{.depth = shape.depth - 1, .width = shape.width, .text = shape.text};
	switch (rng() % 3) {
	case 0: {
		auto elements = Elements{};
		for (auto count = rng() % (shape.width + 1); count > 0; --count) {
			elements.push_back(generate(rng, inner));
		}
		return Model{.value = std::move(elements)};
	}
	case 1: {
		auto members = Members{};
		for (auto count = rng() % (shape.width + 1); count > 0; --count) {
			auto name = random_text(rng, 4);
			if (std::ranges::none_of(members, [&name](auto const& member) { return member.first == name; })) {
				members.emplace_back(std::move(name), generate(rng, inner));
			}
		}
		return Model{.value = std::move(members)};
	}
	default:
		return generate_scalar(rng, shape);
	}
}

/** A container root, so that every proper prefix of its text is incomplete. */
export [[nodiscard]] auto generate_container(std::mt19937_64& rng, Shape shape) -> Model {
	while (true) {
		auto model = generate(rng, Shape{.depth = std::max<std::size_t>(shape.depth, 1), .width = shape.width, .text = shape.text});
		if (std::holds_alternative<Elements>(model.value) || std::holds_alternative<Members>(model.value)) {
			return model;
		}
	}
}

/** What each bound measures, for one model as one serializer writes it. */
export struct Measured {
	std::size_t depth;
	std::size_t string_bytes;
	std::size_t members;
	std::size_t number_chars;
};

export [[nodiscard]] auto measure(Model const& model) -> Measured {
	auto measured = Measured{.depth = 0, .string_bytes = 0, .members = 0, .number_chars = 0};
	auto const merge = [&measured](Measured const& inner) {
		measured.depth = std::max(measured.depth, inner.depth + 1);
		measured.string_bytes = std::max(measured.string_bytes, inner.string_bytes);
		measured.members = std::max(measured.members, inner.members);
		measured.number_chars = std::max(measured.number_chars, inner.number_chars);
	};
	if (std::holds_alternative<std::string>(model.value)) {
		measured.string_bytes = std::get<std::string>(model.value).size();
	} else if (std::holds_alternative<std::int64_t>(model.value) || std::holds_alternative<double>(model.value)) {
		measured.number_chars = number_text(model).size();
	} else if (std::holds_alternative<Elements>(model.value)) {
		auto const& elements = std::get<Elements>(model.value);
		measured.depth = 1;
		measured.members = elements.size();
		for (auto const& element : elements) {
			merge(measure(element));
		}
	} else if (std::holds_alternative<Members>(model.value)) {
		auto const& members = std::get<Members>(model.value);
		measured.depth = 1;
		measured.members = members.size();
		for (auto const& [name, member] : members) {
			merge(measure(member));
			measured.string_bytes = std::max(measured.string_bytes, name.size());
		}
	}
	return measured;
}

}
