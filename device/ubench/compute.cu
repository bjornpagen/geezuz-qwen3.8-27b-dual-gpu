// SPDX-License-Identifier: AGPL-3.0-only
//
// The arithmetic microbenchmarks: the tensor core alone and beside the weight decode, the decode and the widening
// alone, the folded-scale decode the engine refuses, the shared transcendentals, `ldmatrix` from swizzled and
// unswizzled tiles, register double buffering, and the two warp reductions. Each rate kernel runs a fixed amount of
// work per block between two block barriers and reports the cycles it took and a value folded from its results, so
// nothing it computes is dead.
#include "prims/budget.h"
#include "prims/fragment.h"
#include "prims/memory.h"
#include "prims/mma.h"
#include "prims/numeric.h"
#include "prims/order.h"
#include "prims/reduce.h"
#include "prims/selfcheck.h"
#include "prims/shared.h"
#include "prims/sync.h"
#include "prims/thread.h"
#include "prims/transcendental.h"
#include "ubench/device.h"
#include "ubench/shapes.h"

namespace engine::device::ubench::compute {

/** A lane's distinct bf16x2 operand bits: normal, finite values whose products stay normal. */
[[nodiscard]] __attribute__((device)) inline auto operand(unsigned salt) -> unsigned {
	unsigned const mantissas = (prims::lane() * 0x9E3779B9U + salt) & 0x007F007FU;
	return mantissas | 0x3F803F80U;
}

__attribute__((device)) inline void record(RateParams params, unsigned long long start, unsigned long long sink) {
	using namespace engine::device::prims;
	sync_block();
	if (thread_in_block() == 0) {
		at<Rate>(params.rates)[block()] = {.cycles = cycles() - start, .sink = sink};
	}
}

/** Folds an accumulator into a sink: its bits, so the `mma`s that made them stay live. */
[[nodiscard]] __attribute__((device)) inline auto folded(prims::Accumulator const& accumulator) -> unsigned long long {
	unsigned long long sink = 0;
	for (float const value : accumulator.reg) {
		sink += __builtin_bit_cast(unsigned, value);
	}
	return sink;
}

template<Mix mix>
__attribute__((device)) inline void tensor(RateParams params) {
	using namespace engine::device::prims;
	AFragment const a = {.reg = {operand(1), operand(2), operand(3), operand(4)}};
	BFragment const b = {.reg = {operand(5), operand(6)}};
	Accumulator accumulators[4] = {};
	Chunk chunks[2] = {
	    {.word = {operand(7), operand(8), operand(9), operand(10)}},
	    {.word = {operand(11), operand(12), operand(13), operand(14)}},
	};
	unsigned widened[8] = {operand(15), operand(16), operand(17), operand(18), operand(19), operand(20), operand(21), operand(22)};
	sync_block();
	unsigned long long const start = cycles();
	for (unsigned iteration = 0; iteration < static_cast<unsigned>(iterations); ++iteration) {
		if constexpr (mix == Mix::Mma) {
			for (auto& accumulator : accumulators) {
				mma(accumulator, a, b);
			}
		} else if constexpr (mix == Mix::MmaDecode) {
			chunks[0].word[0] += iteration;
			AFragments2 const decoded = decode_chunk(chunks[0]);
			for (auto const& step : decoded.step) {
				for (auto& accumulator : accumulators) {
					mma(accumulator, step, b);
				}
			}
		} else if constexpr (mix == Mix::Decode) {
			// Two independent chains: each decode feeds the next chunk, so none of it can be hoisted or dropped.
			for (auto& chunk : chunks) {
				AFragments2 const decoded = decode_chunk(chunk);
				chunk = {.word = {decoded.step[0].reg[0] ^ decoded.step[0].reg[1], decoded.step[0].reg[2] ^ decoded.step[0].reg[3],
				                  decoded.step[1].reg[0] ^ decoded.step[1].reg[1], decoded.step[1].reg[2] ^ decoded.step[1].reg[3]}};
			}
		} else {
			for (unsigned& value : widened) {
				value = __builtin_bit_cast(unsigned, widen_first(Bf16x2{.bits = value}));
			}
		}
	}
	unsigned long long sink = 0;
	for (auto const& accumulator : accumulators) {
		sink += folded(accumulator);
	}
	for (auto const& chunk : chunks) {
		sink += chunk.word[0] + chunk.word[3];
	}
	for (unsigned const value : widened) {
		sink += value;
	}
	record(params, start, sink);
}

/** Code `code`, the two nan codes (which the baker refuses) replaced by zero. */
[[nodiscard]] constexpr auto admitted(unsigned code) -> prims::E4m3 {
	auto const candidate = prims::E4m3{.bits = static_cast<unsigned char>(code & 0xffU)};
	return prims::finite(candidate) ? candidate : prims::E4m3{.bits = 0};
}

[[nodiscard]] constexpr auto power_of_two_bf16(int exponent) -> prims::Bf16 {
	return {.bits = static_cast<unsigned short>(static_cast<unsigned>(exponent + 127) << 7U)};
}

/**
 * The decode the engine refuses: the a-fragments carry the codes' payloads (2^-120 times their values) without the
 * `fma` by 2^120, which is folded into the result instead. All 256 codes as a 16 x 16 a-fragment through `mma.sync`
 * against activations 2^j, one warp per exponent; a result that differs from the code's value times 2^j is a bit the
 * fold lost.
 */
__attribute__((device)) inline void decode_folded(FoldedParams params) {
	using namespace engine::device::prims;
	FoldedReading* const reading = at<FoldedReading>(params.reading);
	unsigned const lane_index = lane();
	unsigned const warp = warp_in_block();
	int const exponent = activation_exponents[warp];
	AFragment a{};
	for (int reg = 0; reg < 4; ++reg) {
		unsigned halves[2] = {};
		for (int half = 0; half < 2; ++half) {
			fragment::Coordinate const at_element = fragment::a(static_cast<int>(lane_index), reg, half);
			halves[half] = decode::payload(admitted(static_cast<unsigned>(at_element.row * 16 + at_element.column)));
		}
		a.reg[reg] = halves[0] | (halves[1] << 16U);
	}
	unsigned mismatched = 0;
	unsigned checked = 0;
	for (int shift = 0; shift < 2; ++shift) {
		BFragment b{};
		for (int reg = 0; reg < 2; ++reg) {
			Bf16 halves[2] = {};
			for (int half = 0; half < 2; ++half) {
				fragment::Coordinate const at_element = fragment::b(static_cast<int>(lane_index), reg, half);
				halves[half] = at_element.row == at_element.column + 8 * shift ? power_of_two_bf16(exponent) : Bf16{.bits = 0};
			}
			b.reg[reg] = pack(halves[0], halves[1]).bits;
		}
		Accumulator d = {.reg = {0.0F, 0.0F, 0.0F, 0.0F}};
		mma(d, a, b);
		for (int reg = 0; reg < 4; ++reg) {
			fragment::Coordinate const at_element = fragment::c(static_cast<int>(lane_index), reg);
			unsigned const code = static_cast<unsigned>(at_element.row * 16 + at_element.column + 8 * shift);
			float const unfolded = d.reg[reg] * math::power_of_two(60) * math::power_of_two(60);
			float const expected = value(admitted(code)) * math::power_of_two(exponent / 2) * math::power_of_two(exponent - exponent / 2);
			mismatched += __builtin_bit_cast(unsigned, unfolded) == __builtin_bit_cast(unsigned, expected) ? 0U : 1U;
			++checked;
		}
	}
	unsigned const warp_checked = warp_sum(checked);
	unsigned const warp_mismatched = warp_sum(mismatched);
	if (lane_index == 0) {
		reading->checked[warp] = warp_checked;
		reading->mismatched[warp] = warp_mismatched;
	}
}

/** A starting input and a step that keep `function` inside its interesting range for every iteration. */
struct Domain {
	float start;
	float step;
};

template<prims::selfcheck::Function function>
[[nodiscard]] constexpr auto domain() -> Domain {
	using prims::selfcheck::Function;
	if constexpr (function == Function::Exp) {
		return {.start = -20.0F, .step = 0x1p-10F};
	} else if constexpr (function == Function::Log) {
		return {.start = 1.0F, .step = 0x1p-6F};
	} else if constexpr (function == Function::Rsqrt) {
		return {.start = 1.0F, .step = 0x1p-4F};
	} else {
		return {.start = -8.0F, .step = 0x1p-12F};
	}
}

template<prims::selfcheck::Function function>
[[nodiscard]] __attribute__((device)) inline auto evaluate(float input) -> float {
	using prims::selfcheck::Function;
	namespace math = prims::math;
	if constexpr (function == Function::Exp) {
		return math::exp(input);
	} else if constexpr (function == Function::Log) {
		return math::log(input);
	} else if constexpr (function == Function::Rsqrt) {
		return math::rsqrt(input);
	} else if constexpr (function == Function::Sigmoid) {
		return math::sigmoid(input);
	} else if constexpr (function == Function::Silu) {
		return math::silu(input);
	} else {
		return math::softplus(input);
	}
}

inline constexpr int transcendental_lanes = 4;

/** Four independent evaluations per iteration per thread, their results folded into a sink. */
template<prims::selfcheck::Function function>
__attribute__((device)) inline void transcendental(RateParams params) {
	using namespace engine::device::prims;
	constexpr Domain range = domain<function>();
	float inputs[transcendental_lanes] = {};
	for (int index = 0; index < transcendental_lanes; ++index) {
		inputs[index] = range.start + static_cast<float>(thread_in_block() * transcendental_lanes + static_cast<unsigned>(index)) * range.step;
	}
	unsigned sink = 0;
	sync_block();
	unsigned long long const start = cycles();
	for (int iteration = 0; iteration < iterations; ++iteration) {
		for (float& input : inputs) {
			sink += __builtin_bit_cast(unsigned, evaluate<function>(input));
			input = input + range.step;
		}
	}
	record(params, start, sink);
}

/** The plain row-major tile the swizzle exists to avoid: eight rows at one chunk share a bank group. */
template<int row_bytes_, int rows_>
struct RowMajor {
	static constexpr int row_bytes = row_bytes_;
	static constexpr int rows = rows_;
	static constexpr int chunks_per_row = row_bytes / 16;
	static constexpr int bytes = row_bytes * rows;

	[[nodiscard]] static constexpr auto offset(int row, int chunk) -> int {
		return row * row_bytes + chunk * 16;
	}
};

/** The tile type a `Swizzle` names, in the enum's order. */
template<Swizzle swizzle>
using ActivationTile = __type_pack_element<static_cast<unsigned>(swizzle), prims::fragment::XorSwizzle<fragment_row_bytes, fragment_rows>,
                                           RowMajor<fragment_row_bytes, fragment_rows>>;

static_assert(prims::fragment::conflict_free<ActivationTile<Swizzle::Xor>>());
static_assert(!prims::fragment::conflict_free<ActivationTile<Swizzle::None>>(), "the comparison tile has the conflicts the swizzle removes");

struct TileLayout {
	alignas(128) unsigned char tile[fragment_row_bytes * fragment_rows];
};

/** Fills the block's tile with words that depend on their place, then waits for every warp's stores. */
template<class Tile>
__attribute__((device)) inline auto fill_tile() -> prims::SharedAddress {
	using namespace engine::device::prims;
	auto& layout = shared_layout<ldmatrix_budget, TileLayout>();
	SharedAddress const tile = shared_address(layout.tile);
	for (unsigned index = thread_in_block(); index < static_cast<unsigned>(Tile::rows * Tile::chunks_per_row); index += ldmatrix_budget.threads) {
		unsigned const row = index / Tile::chunks_per_row;
		unsigned const chunk = index % Tile::chunks_per_row;
		store_shared(tile.plus(static_cast<unsigned>(Tile::offset(static_cast<int>(row), static_cast<int>(chunk)))),
		             {.word = {operand(index), operand(index + 1), operand(index + 2), operand(index + 3)}});
	}
	sync_block();
	return tile;
}

/** The address this lane hands `ldmatrix.x4` for the four 8 x 8 matrices at logical chunks `chunk` and `chunk + 1`. */
template<class Tile>
[[nodiscard]] __attribute__((device)) inline auto matrix_row(prims::SharedAddress tile, int chunk) -> prims::SharedAddress {
	int const matrix = static_cast<int>(prims::lane()) / 8;
	int const row = 8 * (matrix % 2) + static_cast<int>(prims::lane()) % 8;
	return tile.plus(static_cast<unsigned>(Tile::offset(row, chunk + matrix / 2)));
}

template<Swizzle swizzle>
__attribute__((device)) inline void ldmatrix_rate(RateParams params) {
	using namespace engine::device::prims;
	using Tile = ActivationTile<swizzle>;
	SharedAddress const tile = fill_tile<Tile>();
	unsigned sink = 0;
	unsigned long long const start = cycles();
	for (int iteration = 0; iteration < iterations; ++iteration) {
		Matrices4 const loaded = ldmatrix_x4<fragment::Transpose::Plain>(matrix_row<Tile>(tile, (iteration * 2) % Tile::chunks_per_row));
		sink += loaded.reg[0] ^ loaded.reg[1] ^ loaded.reg[2] ^ loaded.reg[3];
	}
	record(params, start, sink);
}

/** `pipeline_steps` k-steps of two `mma`s each, their b-fragments by `ldmatrix.x4.trans`, with or without overlap. */
template<Pipelining pipelining>
__attribute__((device)) inline void pipelined(RateParams params) {
	using namespace engine::device::prims;
	using Tile = ActivationTile<Swizzle::Xor>;
	SharedAddress const tile = fill_tile<Tile>();
	AFragment const a = {.reg = {operand(1), operand(2), operand(3), operand(4)}};
	Accumulator accumulators[2] = {};
	auto const load = [&](int step) {
		return ldmatrix_x4<fragment::Transpose::Transposed>(matrix_row<Tile>(tile, (step * 2) % Tile::chunks_per_row));
	};
	auto const compute = [&](int, Matrices4 const& b) {
		mma(accumulators[0], a, BFragment{.reg = {b.reg[0], b.reg[1]}});
		mma(accumulators[1], a, BFragment{.reg = {b.reg[2], b.reg[3]}});
	};
	unsigned long long const start = cycles();
	for (int round = 0; round < iterations / pipeline_steps; ++round) {
		if constexpr (pipelining == Pipelining::Double) {
			double_buffered<pipeline_steps>(load, compute);
		} else {
			for (int step = 0; step < pipeline_steps; ++step) {
				compute(step, load(step));
			}
		}
	}
	record(params, start, folded(accumulators[0]) + folded(accumulators[1]));
}

inline constexpr int reduction_lanes = 4;

template<Reduction reduction>
__attribute__((device)) inline void reduction_rate(RateParams params) {
	using namespace engine::device::prims;
	unsigned long long sink = 0;
	sync_block();
	unsigned long long const start = cycles();
	if constexpr (reduction == Reduction::FloatButterfly) {
		float values[reduction_lanes] = {};
		for (int index = 0; index < reduction_lanes; ++index) {
			values[index] = static_cast<float>(lane() + 1U) * static_cast<float>(index + 1);
		}
		for (int iteration = 0; iteration < iterations; ++iteration) {
			for (float& value : values) {
				// A sum of 32 values scaled back by 1/32 stays in range however long the chain runs.
				value = warp_sum(value) * 0.03125F;
			}
		}
		for (float const value : values) {
			sink += __builtin_bit_cast(unsigned, value);
		}
	} else {
		unsigned values[reduction_lanes] = {};
		for (int index = 0; index < reduction_lanes; ++index) {
			values[index] = lane() * static_cast<unsigned>(index + 1);
		}
		for (int iteration = 0; iteration < iterations; ++iteration) {
			for (unsigned& value : values) {
				value = warp_sum(value) + static_cast<unsigned>(iteration);
			}
		}
		for (unsigned const value : values) {
			sink += value;
		}
	}
	record(params, start, sink);
}

}

namespace ub = engine::device::ubench;
namespace compute = engine::device::ubench::compute;
using engine::device::prims::selfcheck::Function;

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_tensor_mma = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_tensor_mma(compute::RateParams params) {
	compute::tensor<compute::Mix::Mma>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_tensor_mma_decode = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_tensor_mma_decode(compute::RateParams params) {
	compute::tensor<compute::Mix::MmaDecode>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_tensor_decode = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_tensor_decode(compute::RateParams params) {
	compute::tensor<compute::Mix::Decode>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_tensor_widen = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_tensor_widen(compute::RateParams params) {
	compute::tensor<compute::Mix::Widen>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_decode_folded = compute::folded_budget;
extern "C" __attribute__((global, launch_bounds(compute::folded_budget.threads, compute::folded_budget.blocks_per_sm))) void ubench_decode_folded(compute::FoldedParams params) {
	compute::decode_folded(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_exp = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_exp(compute::RateParams params) {
	compute::transcendental<Function::Exp>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_log = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_log(compute::RateParams params) {
	compute::transcendental<Function::Log>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_rsqrt = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_rsqrt(compute::RateParams params) {
	compute::transcendental<Function::Rsqrt>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_sigmoid = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_sigmoid(compute::RateParams params) {
	compute::transcendental<Function::Sigmoid>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_silu = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_silu(compute::RateParams params) {
	compute::transcendental<Function::Silu>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_softplus = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_softplus(compute::RateParams params) {
	compute::transcendental<Function::Softplus>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_ldmatrix_swizzled = compute::ldmatrix_budget;
extern "C" __attribute__((global, launch_bounds(compute::ldmatrix_budget.threads, compute::ldmatrix_budget.blocks_per_sm))) void ubench_ldmatrix_swizzled(compute::RateParams params) {
	compute::ldmatrix_rate<compute::Swizzle::Xor>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_ldmatrix_row_major = compute::ldmatrix_budget;
extern "C" __attribute__((global, launch_bounds(compute::ldmatrix_budget.threads, compute::ldmatrix_budget.blocks_per_sm))) void ubench_ldmatrix_row_major(compute::RateParams params) {
	compute::ldmatrix_rate<compute::Swizzle::None>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_mma_double_buffered = compute::ldmatrix_budget;
extern "C" __attribute__((global, launch_bounds(compute::ldmatrix_budget.threads, compute::ldmatrix_budget.blocks_per_sm))) void ubench_mma_double_buffered(compute::RateParams params) {
	compute::pipelined<compute::Pipelining::Double>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_mma_sequential = compute::ldmatrix_budget;
extern "C" __attribute__((global, launch_bounds(compute::ldmatrix_budget.threads, compute::ldmatrix_budget.blocks_per_sm))) void ubench_mma_sequential(compute::RateParams params) {
	compute::pipelined<compute::Pipelining::Sequential>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_reduce_float = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_reduce_float(compute::RateParams params) {
	compute::reduction_rate<compute::Reduction::FloatButterfly>(params);
}

extern "C" __attribute__((device, used)) constexpr ub::Budget engine_budget_ubench_reduce_integer = compute::rate_budget;
extern "C" __attribute__((global, launch_bounds(compute::rate_budget.threads, compute::rate_budget.blocks_per_sm))) void ubench_reduce_integer(compute::RateParams params) {
	compute::reduction_rate<compute::Reduction::IntegerRedux>(params);
}
