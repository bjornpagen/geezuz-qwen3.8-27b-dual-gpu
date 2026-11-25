// SPDX-License-Identifier: AGPL-3.0-only
//
// The arithmetic kernels as the host launches them. Every rate kernel runs one block per SM and reports one `Rate`
// per block.
export module engine.ubench_device:compute;

import std;
import engine.driver;
import engine.prims;
import engine.ptx;
import :shapes;
import :images;
import :names;
import :stream;

namespace engine::ubench_device {

namespace ub = engine::device::ubench;
namespace compute = engine::device::ubench::compute;
using Function = engine::device::prims::selfcheck::Function;

/** A rate kernel of the compute image: one block per SM, one `Rate` each. */
template<std::string_view const& entry_, device::prims::Budget budget_>
struct RateKernel {
	static constexpr ptx::Image image = compute_image;
	static constexpr std::string_view entry = entry_;
	static constexpr driver::Dim grid = every_sm;
	static constexpr device::prims::Budget budget = budget_;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = compute::RateParams;

	[[nodiscard]] static auto params(BlockRates const& rates) -> Params {
		return {.rates = rates.address()};
	}
};

inline constexpr std::string_view tensor_mma = "ubench_tensor_mma";
inline constexpr std::string_view tensor_mma_decode = "ubench_tensor_mma_decode";
inline constexpr std::string_view tensor_decode = "ubench_tensor_decode";
inline constexpr std::string_view tensor_widen = "ubench_tensor_widen";

/** The tensor-rate kernel of each mix, in the enum's order. */
export template<compute::Mix mix>
using TensorKernel = std::tuple_element_t<static_cast<std::size_t>(mix),
                                          std::tuple<RateKernel<tensor_mma, compute::rate_budget>, RateKernel<tensor_mma_decode, compute::rate_budget>,
                                                     RateKernel<tensor_decode, compute::rate_budget>, RateKernel<tensor_widen, compute::rate_budget>>>;

inline constexpr std::string_view exp_entry = "ubench_exp";
inline constexpr std::string_view log_entry = "ubench_log";
inline constexpr std::string_view rsqrt_entry = "ubench_rsqrt";
inline constexpr std::string_view sigmoid_entry = "ubench_sigmoid";
inline constexpr std::string_view silu_entry = "ubench_silu";
inline constexpr std::string_view softplus_entry = "ubench_softplus";

/** The throughput kernel of each transcendental, in the digest's function order. */
export template<Function function>
using TranscendentalKernel = std::tuple_element_t<
    static_cast<std::size_t>(function),
    std::tuple<RateKernel<exp_entry, compute::rate_budget>, RateKernel<log_entry, compute::rate_budget>, RateKernel<rsqrt_entry, compute::rate_budget>,
               RateKernel<sigmoid_entry, compute::rate_budget>, RateKernel<silu_entry, compute::rate_budget>,
               RateKernel<softplus_entry, compute::rate_budget>>>;

inline constexpr std::string_view ldmatrix_swizzled = "ubench_ldmatrix_swizzled";
inline constexpr std::string_view ldmatrix_row_major = "ubench_ldmatrix_row_major";

export template<compute::Swizzle swizzle>
using LdmatrixKernel = std::tuple_element_t<static_cast<std::size_t>(swizzle), std::tuple<RateKernel<ldmatrix_swizzled, compute::ldmatrix_budget>,
                                                                                        RateKernel<ldmatrix_row_major, compute::ldmatrix_budget>>>;

inline constexpr std::string_view mma_double_buffered = "ubench_mma_double_buffered";
inline constexpr std::string_view mma_sequential = "ubench_mma_sequential";

export template<compute::Pipelining pipelining>
using PipelineKernel = std::tuple_element_t<static_cast<std::size_t>(pipelining), std::tuple<RateKernel<mma_double_buffered, compute::ldmatrix_budget>,
                                                                                           RateKernel<mma_sequential, compute::ldmatrix_budget>>>;

inline constexpr std::string_view reduce_float = "ubench_reduce_float";
inline constexpr std::string_view reduce_integer = "ubench_reduce_integer";

export template<compute::Reduction reduction>
using ReductionKernel = std::tuple_element_t<static_cast<std::size_t>(reduction),
                                             std::tuple<RateKernel<reduce_float, compute::rate_budget>, RateKernel<reduce_integer, compute::rate_budget>>>;

export using FoldedReading = driver::DeviceArray<compute::FoldedReading, 1>;

export struct DecodeFoldedKernel {
	static constexpr ptx::Image image = compute_image;
	static constexpr std::string_view entry = "ubench_decode_folded";
	static constexpr driver::Dim grid = one_block;
	static constexpr device::prims::Budget budget = compute::folded_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = compute::FoldedParams;

	[[nodiscard]] static auto params(FoldedReading const& reading) -> Params {
		return {.reading = reading.address()};
	}
};

}
