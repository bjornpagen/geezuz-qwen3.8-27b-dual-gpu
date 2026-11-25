// SPDX-License-Identifier: AGPL-3.0-only
//
// The deltanet image and its kernels as the host launches them: each kernel's entry, grid, budget and parameter block
// are the device's own definitions, so a launch cannot disagree with the kernel it starts.
export module engine.deltanet:kernels;

import std;
import engine.driver;
import engine.prims;
import engine.ptx;
import :arithmetic;

namespace engine::deltanet {

constexpr char text[] = {
#embed "deltanet.ptx" suffix(, '\0')
};

/** A device allocation's address as the kernels' parameters carry it. */
export template<class T>
[[nodiscard]] auto address(std::uint64_t device_address) -> Address<T> {
	/* SAFETY: the pointer only travels to the card inside launch parameters; the host never dereferences it. */
	return {.value = std::bit_cast<T*>(device_address)};
}

/** The deltanet kernels' ptx image. */
export inline constexpr ptx::Image image = ptx::image(text);

static_assert(image.header == ptx::abi.header, "every image shares the engine's isa and target");

/**
 * One cycle's core at one layer: block b is live slot b / key_heads's key head group b % key_heads. The decode lane
 * runs the same roles inside its persistent kernel; this launch is what the microbenchmark times and the host checks.
 */
export struct DecodeCore {
	static constexpr ptx::Image image = deltanet::image;
	static constexpr std::string_view entry = "deltanet_decode_core";
	static constexpr driver::Dim grid = {.x = decode_items, .y = 1, .z = 1};
	static constexpr engine::device::prims::Budget budget = decode_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = DecodeParams;
};

/** One prompt chunk per live slot at one layer: block b is live slot b / value_heads's value head b % value_heads. */
export struct Prefill {
	static constexpr ptx::Image image = deltanet::image;
	static constexpr std::string_view entry = "deltanet_prefill";
	static constexpr driver::Dim grid = {.x = prefill_items, .y = 1, .z = 1};
	static constexpr engine::device::prims::Budget budget = prefill_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = PrefillParams;
};

/** The windows moved to the chunks' ends; launched after `Prefill` on the same stream, with the same parameters. */
export struct PrefillWindow {
	static constexpr ptx::Image image = deltanet::image;
	static constexpr std::string_view entry = "deltanet_prefill_window";
	static constexpr driver::Dim grid = {.x = window_items, .y = 1, .z = 1};
	static constexpr engine::device::prims::Budget budget = window_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = PrefillParams;
};

/** The chain probe under the engine's tree, eight lanes per column. */
export struct ProbeChain8 {
	static constexpr ptx::Image image = deltanet::image;
	static constexpr std::string_view entry = "deltanet_probe_chain_8";
	static constexpr driver::Dim grid = {.x = probe_blocks<Column>, .y = 1, .z = 1};
	static constexpr engine::device::prims::Budget budget = probe_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = ProbeParams;
};

/** The chain probe with a whole warp per column. */
export struct ProbeChain32 {
	static constexpr ptx::Image image = deltanet::image;
	static constexpr std::string_view entry = "deltanet_probe_chain_32";
	static constexpr driver::Dim grid = {.x = probe_blocks<Tree<32>>, .y = 1, .z = 1};
	static constexpr engine::device::prims::Budget budget = probe_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = ProbeParams;
};

/** The butterfly self-check; its report starts zeroed and must come back with no mismatches. */
export struct CheckTrees {
	static constexpr ptx::Image image = deltanet::image;
	static constexpr std::string_view entry = "deltanet_check_trees";
	static constexpr driver::Dim grid = {.x = check_blocks, .y = 1, .z = 1};
	static constexpr engine::device::prims::Budget budget = check_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = CheckParams;
};

static_assert(driver::Kernel<DecodeCore> && driver::Kernel<Prefill> && driver::Kernel<PrefillWindow>);
static_assert(driver::Kernel<ProbeChain8> && driver::Kernel<ProbeChain32> && driver::Kernel<CheckTrees>);

/** Every deltanet kernel, as one program over the image. */
export using Kernels = driver::Program<DecodeCore, Prefill, PrefillWindow, ProbeChain8, ProbeChain32, CheckTrees>;

}
