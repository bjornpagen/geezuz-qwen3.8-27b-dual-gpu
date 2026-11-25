// SPDX-License-Identifier: AGPL-3.0-only
//
// The device primitives that have a meaning on the host, re-exported unchanged so host code (the typed launch
// functions, the baker, the cpu reference, the microbenchmark drivers and the tests) computes with the very
// definitions the kernels compile. Device code cannot import modules, so each definition is a device header;
// including it inside the purview, under one export, makes its declarations this module's surface without restating
// a name. Below them, the host-only boundary pieces: the finite e4m3 code as a parsed type and the baker's tiles.
export module engine.prims;

import std;

export extern "C++" {
#include "prims/budget.h"
#include "prims/fragment.h"
#include "prims/numeric.h"
#include "prims/tile.h"
}

namespace engine::prims {

namespace device = engine::device::prims;

/** An e4m3 code that is a number: the only kind the baker writes, so the device never meets a nan. */
export class FiniteE4m3 {
public:
	/** The code, or nothing for the two nan codes (0x7f, 0xff). */
	[[nodiscard]] static constexpr auto parse(std::uint8_t bits) -> std::optional<FiniteE4m3> {
		auto const code = device::E4m3{.bits = bits};
		if (!device::finite(code)) {
			return std::nullopt;
		}
		return FiniteE4m3{code};
	}

	[[nodiscard]] constexpr auto code() const -> device::E4m3 {
		return code_;
	}

private:
	constexpr explicit FiniteE4m3(device::E4m3 code) : code_{code} {}

	device::E4m3 code_;
};

/**
 * The baked word for four finite codes: `top_first` and `top_second` decode into the row-g register,
 * `bottom_first` and `bottom_second` into the row-(g + 8) register (`fragment::weight` says which elements those
 * are).
 */
export [[nodiscard]] constexpr auto bake_word(FiniteE4m3 top_first, FiniteE4m3 top_second, FiniteE4m3 bottom_first, FiniteE4m3 bottom_second)
    -> std::uint32_t {
	return device::decode::encode(top_first.code(), top_second.code(), bottom_first.code(), bottom_second.code());
}

/** A baked tile: 4096 little-endian words, word w at byte 4w. */
export using TileWords = std::array<std::uint32_t, device::tile::words>;

/**
 * The baked e4m3 tile for 128 x 128 codes given row-major (code (r, k) at 128r + k): each word holds the four codes
 * `tile::e4m3_element` names, re-encoded for the decode. Nothing when any code is a nan, which the baker refuses.
 */
export [[nodiscard]] constexpr auto bake_e4m3_tile(std::span<std::uint8_t const, device::tile::bytes> row_major) -> std::optional<TileWords> {
	auto words = TileWords{};
	for (auto word = 0; word < device::tile::words; ++word) {
		auto codes = std::array<device::E4m3, 4>{};
		for (auto place = 0; place < 4; ++place) {
			auto const at = device::tile::e4m3_element({.word = word, .place = place});
			auto const index = static_cast<std::size_t>(at.row) * device::tile::e4m3_columns + static_cast<std::size_t>(at.column);
			codes.at(static_cast<std::size_t>(place)) = {.bits = row_major[index]};
		}
		if (!std::ranges::all_of(codes, device::finite)) {
			return std::nullopt;
		}
		words.at(static_cast<std::size_t>(word)) = device::decode::encode(codes[0], codes[1], codes[2], codes[3]);
	}
	return words;
}

/** The bf16 tile for 128 x 64 values given row-major (value (r, k) at 64r + k), as bf16 bit patterns. */
export [[nodiscard]] constexpr auto bake_bf16_tile(std::span<std::uint16_t const, device::tile::bytes / 2> row_major) -> TileWords {
	auto words = TileWords{};
	for (auto word = 0; word < device::tile::words; ++word) {
		auto packed = std::uint32_t{0};
		for (auto half = 0; half < 2; ++half) {
			auto const at = device::tile::bf16_element({.word = word, .place = half});
			auto const index = static_cast<std::size_t>(at.row) * device::tile::bf16_columns + static_cast<std::size_t>(at.column);
			auto const bits = std::uint32_t{row_major[index]};
			packed |= bits << (16U * static_cast<std::uint32_t>(half));
		}
		words.at(static_cast<std::size_t>(word)) = packed;
	}
	return words;
}

}
