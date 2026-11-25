// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention's kernels against the cpu reference, bit for bit, across the change from 256-position to 2048-position
// groups at position 32768: a context of 32,769 positions whose every key and value both sides prepare, and the rows
// that see past the change. It costs as much as every short case together, so it has its own deadline.
import std;
import engine.numerics_attention;
import engine.testing;

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "decode, two rows across the change to long groups at position 32768 (seed 5)",
		                      .holds = engine::numerics_attention::decode_agrees(5, 32767, 2, 1)},
	};
	return engine::testing::report(cases);
}
