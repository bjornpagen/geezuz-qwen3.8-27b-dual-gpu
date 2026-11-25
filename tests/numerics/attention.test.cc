// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention's kernels against the cpu reference, bit for bit, at the real head shape, near the start of a context:
// decode is a verify cycle's rows (one to five, each seeing the rows before it), prefill a prompt chunk's rows with its
// keys appended first, across the 256-position groups. The change to 2048-position groups at position 32768 has its own
// test (`numerics.attention_long`), under its own deadline.
import std;
import engine.attn;
import engine.numerics_attention;
import engine.testing;

namespace {

namespace attn = engine::device::attn;

using engine::numerics_attention::decode_agrees;
using engine::numerics_attention::prefill_agrees;

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "decode, one row at position 0, every query head (seed 1)",
		                      .holds = decode_agrees(1, 0, 1, attn::query_heads)},
	    engine::testing::Case{.name = "decode, five rows at position 37, every query head (seed 2)",
		                      .holds = decode_agrees(2, 37, 5, attn::query_heads)},
	    engine::testing::Case{.name = "decode, five rows across the first group boundary at 256 (seed 3)",
		                      .holds = decode_agrees(3, 253, 5, attn::query_heads)},
	    engine::testing::Case{.name = "decode, three rows across several groups at position 1021 (seed 4)",
		                      .holds = decode_agrees(4, 1021, 3, 4)},
	    engine::testing::Case{.name = "prefill, a 13-row chunk across the group boundary at 256 (seed 6)",
		                      .holds = prefill_agrees(6, 248, 13)},
	};
	return engine::testing::report(cases);
}
