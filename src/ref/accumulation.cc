// SPDX-License-Identifier: AGPL-3.0-only
export module engine.ref:accumulation;

import std;
import engine.mma_model;
import :numeric;

namespace engine::ref {

/** The operands of one k16 step of one output element: sixteen bf16 of the weight row and of the activation. */
export using Step = std::span<Bf16 const, 16>;

/**
 * How one tensor-core step accumulates: d = c + sum of the 16 products. The reference is a template over it, so the
 * engine's own accumulation and the ieee one that measures the noise floor are two types, never a switch.
 */
export template<class A>
concept Accumulation = requires(float c, Step a, Step b) {
	{ A::step(c, a, b) } -> std::same_as<float>;
	{ A::name } -> std::convertible_to<std::string_view>;
};

/**
 * ieee fp32: each product fused into the running sum in ascending k, every operation rounded to nearest. Not what any
 * card computes; the reference under it, against the reference under the tensor core's accumulation, is the noise
 * the accumulation alone causes.
 */
export struct Ieee {
	static constexpr std::string_view name = "ieee";

	[[nodiscard]] static auto step(float c, Step a, Step b) -> float {
		auto sum = c;
		for (auto k = std::size_t{0}; k < 16; ++k) {
			sum = std::fma(widen(a[k]), widen(b[k]), sum);
		}
		return sum;
	}
};

/** The sm_86 tensor core: each k16 step two blocks of eight, aligned to the largest term and truncated. */
export struct Tensor {
	static constexpr std::string_view name = "sm86";

	[[nodiscard]] static auto step(float c, Step a, Step b) -> float {
		return mma_model::step(c, a, b);
	}
};

/** One step's operands from contiguous runs: `a` and `b` each start at their step's first element. */
export [[nodiscard]] constexpr auto step_of(std::span<Bf16 const> run, std::size_t first) -> Step {
	return run.subspan(first).first<16>();
}

}
