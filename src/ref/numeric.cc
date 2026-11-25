// SPDX-License-Identifier: AGPL-3.0-only
export module engine.ref:numeric;

import std;
import engine.core;
import engine.prims;

namespace engine::ref {

namespace prims = engine::device::prims;

/** A bf16 by its bit pattern, as the checkpoint stores it and the tensor cores read it. */
export using prims::Bf16;

/** bf16 widens to fp32 exactly. */
export using prims::widen;

/** fp32 to bf16, nearest even: the engine's one activation rounding. */
export [[nodiscard]] constexpr auto round(float value) -> Bf16 {
	return prims::round_to_bf16(value);
}

/** Every finite e4m3 code's exact bf16; the two nan codes never reach it (the checkpoint is scanned at open). */
inline constexpr auto e4m3_table = [] {
	auto table = std::array<Bf16, 256>{};
	for (auto code = 0U; code < 256U; ++code) {
		table.at(code) = prims::to_bf16(prims::E4m3{.bits = static_cast<std::uint8_t>(code)});
	}
	return table;
}();

export [[nodiscard]] constexpr auto decode_e4m3(std::byte code) -> Bf16 {
	return e4m3_table.at(std::to_integer<std::size_t>(code));
}

/** Whether `code` is one of the two e4m3 nan codes. */
export [[nodiscard]] constexpr auto e4m3_nan(std::byte code) -> bool {
	return (std::to_integer<unsigned>(code) & 0x7fU) == 0x7fU;
}

/** The little-endian bf16 at element `index` of `bytes`. */
export [[nodiscard]] constexpr auto bf16_at(std::span<std::byte const> bytes, std::size_t index) -> Bf16 {
	auto const low = std::to_integer<unsigned>(bytes[2 * index]);
	auto const high = std::to_integer<unsigned>(bytes[(2 * index) + 1]);
	return {.bits = static_cast<std::uint16_t>(low | (high << 8U))};
}

/** Rows of equal width, row-major: one row per token. */
export template<class T>
struct Rows {
	std::size_t width;
	std::vector<T> values;

	[[nodiscard]] static auto zeros(std::size_t count, std::size_t width) -> Rows {
		return {.width = width, .values = std::vector<T>(count * width)};
	}

	[[nodiscard]] auto count() const -> std::size_t {
		return width == 0 ? 0 : values.size() / width;
	}

	[[nodiscard]] auto row(std::size_t index) -> std::span<T> {
		return std::span{values}.subspan(index * width, width);
	}

	[[nodiscard]] auto row(std::size_t index) const -> std::span<T const> {
		return std::span{values}.subspan(index * width, width);
	}
};

}
