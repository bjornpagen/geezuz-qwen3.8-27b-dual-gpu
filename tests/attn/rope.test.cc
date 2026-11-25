// SPDX-License-Identifier: AGPL-3.0-only
//
// The norms and the rope against double precision: a prepared query or key is the exact normalized, rotated row (by
// the same fp32 angle) to within its bf16 rounding; and the rope table's layout gives every (position, pair) its own
// eight bytes, a lane's eight pairs one contiguous 64-byte run. The rotations themselves are the model's, held to libm
// by its own test.
import std;
import engine.attn;
import engine.model;
import engine.prims;
import engine.testing;

namespace {

namespace attn = engine::device::attn;
namespace prims = engine::device::prims;
namespace rope = engine::model::rope;

/** A prepared row against the exact norm and rotation in double (by the same fp32 angle), within bf16 rounding. */
[[nodiscard]] auto prepared_rows_are_exact_to_bf16(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 2000; ++trial) {
		auto const scale = std::ldexp(1.0F, std::uniform_int_distribution<int>{-12, 12}(random));
		auto projection = std::array<float, 256>{};
		auto weight = std::array<prims::Bf16, 256>{};
		for (auto index = std::size_t{0}; index < 256; ++index) {
			projection.at(index) = std::normal_distribution<float>{0.0F, scale}(random);
			weight.at(index) = prims::round_to_bf16(std::normal_distribution<float>{0.0F, 0.3F}(random));
		}
		auto const position =
		    std::uniform_int_distribution<std::uint32_t>{0, static_cast<std::uint32_t>(engine::model::max_positions) - 1}(random);
		auto const row = engine::attn::prepare_rotated(projection, weight, position);
		auto squares = 0.0;
		for (auto const value : projection) {
			squares += static_cast<double>(value) * static_cast<double>(value);
		}
		auto const inverse = 1.0 / std::sqrt(squares / 256.0 + static_cast<double>(engine::model::rms_norm_eps));
		auto normal = std::array<double, 256>{};
		for (auto index = std::size_t{0}; index < 256; ++index) {
			normal.at(index) =
			    static_cast<double>(projection.at(index)) * inverse * (1.0 + static_cast<double>(prims::widen(weight.at(index))));
		}
		auto exact = normal;
		for (auto pair = std::size_t{0}; pair < 32; ++pair) {
			auto const angle = static_cast<double>(rope::angle(position, pair));
			exact.at(pair) = normal.at(pair) * std::cos(angle) - normal.at(pair + 32) * std::sin(angle);
			exact.at(pair + 32) = normal.at(pair + 32) * std::cos(angle) + normal.at(pair) * std::sin(angle);
		}
		auto largest = 0.0;
		for (auto const value : exact) {
			largest = std::max(largest, std::abs(value));
		}
		for (auto index = std::size_t{0}; index < 256; ++index) {
			auto const got = static_cast<double>(prims::widen(row.at(index)));
			// Half a bf16 ulp of the value, and the fp32 work's few ulps of the row's scale.
			auto const allowed = std::abs(exact.at(index)) * 0x1p-8 + largest * 0x1p-20;
			sound = sound && std::abs(got - exact.at(index)) <= allowed;
		}
	}
	return sound;
}

/** Every table entry is its own eight bytes, in [position][pair] order, filling the table exactly. */
[[nodiscard]] auto table_entries_tile_the_table() -> bool {
	auto sound = attn::rope_table_bytes == engine::model::max_positions * rope::pairs * 8;
	for (auto const position : {std::size_t{0}, std::size_t{1}, std::size_t{4095}, engine::model::max_positions - 1}) {
		for (auto pair = std::size_t{0}; pair < rope::pairs; ++pair) {
			sound = sound && attn::rope_offset(position, pair) == (position * rope::pairs + pair) * 8;
		}
		// A lane reads pairs 8 r .. 8 r + 7 as four chunks from the first pair's offset, 16-byte aligned.
		for (auto lane = std::size_t{0}; lane < 4; ++lane) {
			sound = sound && attn::rope_offset(position, lane * 8) % 16 == 0;
		}
	}
	return sound && attn::rope_offset(engine::model::max_positions - 1, rope::pairs - 1) + 8 == attn::rope_table_bytes;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "prepared queries and keys are the exact rows to within bf16 rounding (seed 1)",
		                      .holds = prepared_rows_are_exact_to_bf16(1)},
	    engine::testing::Case{.name = "the rope table gives every (position, pair) its own eight bytes",
		                      .holds = table_entries_tile_the_table()},
	};
	return engine::testing::report(cases);
}
