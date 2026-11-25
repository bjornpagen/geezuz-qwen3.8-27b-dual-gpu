// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.json;
import engine.json_model;
import engine.testing;

namespace {

namespace json = engine::json;
namespace model = engine::json_model;

using Reader = json::Reader<json::request_bounds>;

[[nodiscard]] auto make_reader() -> std::unique_ptr<Reader> {
	auto reader = Reader::make();
	engine::core::invariant(reader.has_value());
	return std::make_unique<Reader>(std::move(*reader));
}

template<json::Sink O>
auto emit_element(json::Elements<O>& elements, model::Model const& value) -> void;

template<json::Sink O>
auto emit_member(json::Members<O>& members, std::string const& name, model::Model const& value) -> void {
	auto const& held = value.value;
	if (std::holds_alternative<std::nullptr_t>(held)) {
		members.null(name);
	} else if (std::holds_alternative<bool>(held)) {
		members.boolean(name, std::get<bool>(held));
	} else if (std::holds_alternative<std::int64_t>(held)) {
		members.integer(name, std::get<std::int64_t>(held));
	} else if (std::holds_alternative<double>(held)) {
		members.real(name, std::get<double>(held));
	} else if (std::holds_alternative<std::string>(held)) {
		members.string(name, std::get<std::string>(held));
	} else if (std::holds_alternative<model::Elements>(held)) {
		members.array(name, [&held](json::Elements<O>& inner) {
			for (auto const& element : std::get<model::Elements>(held)) {
				emit_element(inner, element);
			}
		});
	} else {
		members.object(name, [&held](json::Members<O>& inner) {
			for (auto const& [key, member] : std::get<model::Members>(held)) {
				emit_member(inner, key, member);
			}
		});
	}
}

template<json::Sink O>
auto emit_element(json::Elements<O>& elements, model::Model const& value) -> void {
	auto const& held = value.value;
	if (std::holds_alternative<std::nullptr_t>(held)) {
		elements.null();
	} else if (std::holds_alternative<bool>(held)) {
		elements.boolean(std::get<bool>(held));
	} else if (std::holds_alternative<std::int64_t>(held)) {
		elements.integer(std::get<std::int64_t>(held));
	} else if (std::holds_alternative<double>(held)) {
		elements.real(std::get<double>(held));
	} else if (std::holds_alternative<std::string>(held)) {
		elements.string(std::get<std::string>(held));
	} else if (std::holds_alternative<model::Elements>(held)) {
		elements.array([&held](json::Elements<O>& inner) {
			for (auto const& element : std::get<model::Elements>(held)) {
				emit_element(inner, element);
			}
		});
	} else {
		elements.object([&held](json::Members<O>& inner) {
			for (auto const& [key, member] : std::get<model::Members>(held)) {
				emit_member(inner, key, member);
			}
		});
	}
}

/** The engine's writer, writing a model wrapped in a one-element array (the writer writes containers). */
[[nodiscard]] auto written(model::Model const& value) -> std::string {
	return json::encode([&value](auto& out) {
		using Out = std::remove_cvref_t<decltype(out)>;
		json::write_array(out, [&value](json::Elements<Out>& elements) { emit_element(elements, value); });
	});
}

/** Every generated value, written by the engine's writer, reads back as exactly itself. */
[[nodiscard]] auto writes_read_back(std::uint64_t seed, std::size_t documents) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto count = std::size_t{0}; count < documents; ++count) {
		auto const value = model::generate(rng, model::random_shape(rng));
		auto const text = written(value);
		auto const read = model::read_model(*reader, text);
		auto const expected = model::Model{.value = model::Elements{value}};
		if (!read || !model::same(expected, *read)) {
			std::println("written value did not read back: {:?}", text);
			return false;
		}
	}
	return true;
}

/**
 * Every Unicode scalar value, alone in a string, is written so that it reads back as itself, and is escaped
 * exactly when JSON requires it: the quote, the backslash and the controls below U+0020, nothing else.
 */
[[nodiscard]] auto every_scalar_round_trips() -> bool {
	auto reader = make_reader();
	auto constexpr batch = std::uint32_t{4096};
	for (auto first = std::uint32_t{0}; first <= 0x10FFFF; first += batch) {
		auto scalars = std::vector<char32_t>{};
		for (auto value = first; value < first + batch && value <= 0x10FFFF; ++value) {
			if (value < 0xD800 || value > 0xDFFF) {
				scalars.push_back(char32_t{value});
			}
		}
		auto texts = model::Elements{};
		for (auto const scalar : scalars) {
			texts.push_back(model::Model{.value = model::utf8(scalar)});
		}
		auto const text = written(model::Model{.value = texts});
		auto const read = model::read_model(*reader, text);
		if (!read || !model::same(model::Model{.value = model::Elements{model::Model{.value = texts}}}, *read)) {
			return false;
		}
		for (auto const scalar : scalars) {
			auto const alone = json::encode([scalar](auto& out) {
				using Out = std::remove_cvref_t<decltype(out)>;
				json::write_array(out, [scalar](json::Elements<Out>& elements) { elements.string(model::utf8(scalar)); });
			});
			auto const value = std::uint32_t{scalar};
			auto const must_escape = value < 0x20 || value == '"' || value == '\\';
			auto const literal = "[\"" + model::utf8(scalar) + "\"]";
			if ((alone == literal) == must_escape) {
				return false;
			}
		}
	}
	return true;
}

/** Every finite double is written as text that reads back as a real with exactly its bits. */
[[nodiscard]] auto doubles_read_back_exactly(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	auto reader = make_reader();
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const value = model::random_finite(rng);
		auto const read = model::read_model(*reader, written(model::Model{.value = value}));
		if (!read || !model::same(model::Model{.value = model::Elements{model::Model{.value = value}}}, *read)) {
			std::println("double {} did not read back", value);
			return false;
		}
	}
	return true;
}

/**
 * An append fits exactly when the bytes it writes fit what remains; one that does not fit writes nothing and says
 * how much it needed; one that fits writes what `encode` writes.
 */
[[nodiscard]] auto appends_whole_or_not_at_all(std::uint64_t seed, std::size_t count) -> bool {
	auto rng = engine::testing::generator(seed);
	for (auto index = std::size_t{0}; index < count; ++index) {
		auto const value = model::generate(rng, model::random_shape(rng));
		auto const whole = written(value);
		auto const prefix = model::random_text(rng, 4);
		auto buffer = json::Buffer{prefix.size() + rng() % (2 * whole.size() + 1)};
		buffer.put(prefix);
		auto const appended = buffer.append([&value](auto& out) {
			using Out = std::remove_cvref_t<decltype(out)>;
			json::write_array(out, [&value](json::Elements<Out>& elements) { emit_element(elements, value); });
		});
		auto const fits = whole.size() <= buffer.available() + (appended ? whole.size() : 0);
		if (appended.has_value() != fits) {
			return false;
		}
		if (appended && buffer.text() != prefix + whole) {
			return false;
		}
		if (!appended && (buffer.text() != prefix || appended.error() != json::Overflow{.needed = whole.size(), .available = buffer.available()})) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "written values read back exactly (seed 1)", .holds = writes_read_back(1, 5000)},
	    engine::testing::Case{.name = "every scalar value round-trips and only the required ones are escaped", .holds = every_scalar_round_trips()},
	    engine::testing::Case{.name = "every written double reads back with its bits (seed 2)", .holds = doubles_read_back_exactly(2, 200000)},
	    engine::testing::Case{.name = "an append writes all of its bytes or none (seed 3)", .holds = appends_whole_or_not_at_all(3, 5000)},
	};
	return engine::testing::report(cases);
}
