// SPDX-License-Identifier: AGPL-3.0-only
//
// The integrity kernels as the host launches them: each declaration builds its parameter struct from the typed owners
// it reads and writes, so a launch cannot pass a buffer of the wrong kind or size.
export module engine.soak:kernels;

import std;
import engine.core;
import engine.driver;
import engine.prims;
import engine.ptx;
import :layout;
import :images;

namespace engine::soak {

/** One card's digest state, in host memory the card maps: the host starts a pass in it and reads the digest from it. */
export using StateCell = driver::MappedArray<HashState, 1>;

/** A card's weight image as it lies in the card's memory: the payload of its baked image, byte for byte. */
export template<std::size_t Bytes>
using Weights = driver::DeviceArray<std::byte, Bytes>;

/** The memory the pattern covers on one card. */
export using Region = driver::DeviceArray<Chunk, region_chunks>;

/** One tally per warp of the check, in host memory the card maps. */
export using Tallies = driver::MappedArray<Tally, tallies>;

inline constexpr driver::Dim one_block = {.x = 1, .y = 1, .z = 1};
inline constexpr driver::Dim every_sm = {.x = static_cast<std::uint32_t>(pattern_blocks), .y = 1, .z = 1};

/** The device address `bytes` past the start of `weights`. */
template<std::size_t Bytes>
[[nodiscard]] auto inside(Weights<Bytes> const& weights, std::uint64_t bytes) -> Address {
	core::invariant(bytes <= Bytes);
	return weights.address() + bytes;
}

export struct AbsorbKernel {
	static constexpr ptx::Image image = soak::image;
	static constexpr std::string_view entry = "soak_absorb";
	static constexpr driver::Dim grid = one_block;
	static constexpr engine::device::prims::Budget budget = hash_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = AbsorbParams;

	template<std::size_t Bytes>
	[[nodiscard]] static auto params(StateCell const& state, Weights<Bytes> const& weights, Slice const& slice) -> Params {
		core::invariant((slice.first + slice.count) * block_bytes <= Bytes);
		return {.state = state.device_address(), .source = inside(weights, slice.first * block_bytes), .blocks = slice.count, .unused = 0};
	}
};

export struct FinishKernel {
	static constexpr ptx::Image image = soak::image;
	static constexpr std::string_view entry = "soak_finish";
	static constexpr driver::Dim grid = one_block;
	static constexpr engine::device::prims::Budget budget = hash_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = FinishParams;

	template<std::size_t Bytes>
	[[nodiscard]] static auto params(StateCell const& state, Weights<Bytes> const& weights) -> Params {
		constexpr auto pass = pass_of(Bytes);
		return {
		    .state = state.device_address(),
		    .tail = inside(weights, pass.blocks * block_bytes),
		    .tail_words = pass.tail_words,
		    .unused = 0,
		    .total_bytes = Bytes,
		};
	}
};

export struct WriteKernel {
	static constexpr ptx::Image image = soak::image;
	static constexpr std::string_view entry = "soak_write";
	static constexpr driver::Dim grid = every_sm;
	static constexpr engine::device::prims::Budget budget = pattern_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = WriteParams;

	[[nodiscard]] static auto params(Region const& region, std::uint32_t round) -> Params {
		return {.region = region.address(), .round = round, .unused = 0};
	}
};

export struct CheckKernel {
	static constexpr ptx::Image image = soak::image;
	static constexpr std::string_view entry = "soak_check";
	static constexpr driver::Dim grid = every_sm;
	static constexpr engine::device::prims::Budget budget = pattern_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = CheckParams;

	[[nodiscard]] static auto params(Region const& region, Tallies const& counts, std::uint32_t round) -> Params {
		return {.region = region.address(), .tallies = counts.device_address(), .round = round, .unused = 0};
	}
};

/** Every integrity kernel, loaded together from the one image. */
export using Program = driver::Program<AbsorbKernel, FinishKernel, WriteKernel, CheckKernel>;

}
