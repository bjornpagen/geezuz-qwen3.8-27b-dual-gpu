// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention's ptx image and the declaration of every kernel in it: its entry, grid, budget, residency and the one
// parameter struct it takes. A declaration's budget is the very value the kernel `static_assert`s, read from the
// shared layout header, so the launch and the kernel cannot disagree about threads or shared memory.
module;

#include "attn/layout.h"
#include "attn/selfcheck.h"

export module engine.attn:image;

import std;
import engine.driver;
import engine.ptx;

namespace engine::attn {

namespace device = engine::device::attn;
namespace prims = engine::device::prims;

constexpr char text[] = {
#embed "attn.ptx" suffix(, '\0')
};

export inline constexpr ptx::Image image = ptx::image(text);

static_assert(image.header == ptx::abi.header, "every image shares the engine's isa and target");

/** A kernel of the image with one block per SM. */
inline constexpr driver::Dim per_sm = {.x = device::core_blocks, .y = 1, .z = 1};

/** The prompt chunk's keys and values into the slot's blocks; precedes `Prefill`. */
export struct Append {
	static constexpr ptx::Image image = attn::image;
	static constexpr std::string_view entry = "attn_append";
	static constexpr driver::Dim grid = {.x = device::append_blocks, .y = 1, .z = 1};
	static constexpr prims::Budget budget = device::append_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = device::PromptParams;
};

/** The prompt chunk's causal attention over the slot's kv: gated bf16 outputs. */
export struct Prefill {
	static constexpr ptx::Image image = attn::image;
	static constexpr std::string_view entry = "attn_prefill";
	static constexpr driver::Dim grid = per_sm;
	static constexpr prims::Budget budget = device::core_budget<device::PromptTile>();
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = device::PromptParams;
};

[[nodiscard]] consteval auto decode_core_entry(unsigned rows) -> std::string_view {
	constexpr std::array<std::string_view, 5> entries = {
	    "attn_decode_core_rows1", "attn_decode_core_rows2", "attn_decode_core_rows3", "attn_decode_core_rows4", "attn_decode_core_rows5",
	};
	return entries.at(rows - 1);
}

/** The decode core for slots of at most `rows` rows: every group's record. */
export template<unsigned rows>
struct DecodeCore {
	static constexpr ptx::Image image = attn::image;
	static constexpr std::string_view entry = decode_core_entry(rows);
	static constexpr driver::Dim grid = per_sm;
	static constexpr prims::Budget budget = device::core_budget<device::RowTile<rows>>();
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = device::CoreParams;
};

/** The decode merge: the records folded per (row, query head), gated, as o_proj's bf16 input. */
export struct DecodeMerge {
	static constexpr ptx::Image image = attn::image;
	static constexpr std::string_view entry = "attn_decode_merge";
	static constexpr driver::Dim grid = {.x = device::merge_blocks, .y = 1, .z = 1};
	static constexpr prims::Budget budget = device::merge_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = device::MergeParams;
};

[[nodiscard]] consteval auto one_part_entry(unsigned rows) -> std::string_view {
	return rows == 3 ? "attn_u21_rows3_one_part" : "attn_u21_rows5_one_part";
}

/**
 * U21's comparison core: `DecodeCore<rows>` with P rounded to one bf16 part, which the engine never runs; timing it
 * against `DecodeCore<rows>` prices the exact split. Its records are not attention's.
 */
export template<unsigned rows>
    requires(rows == 3 || rows == 5)
struct U21OnePart {
	static constexpr ptx::Image image = attn::image;
	static constexpr std::string_view entry = one_part_entry(rows);
	static constexpr driver::Dim grid = per_sm;
	static constexpr prims::Budget budget = device::core_budget<device::RowTile<rows>>();
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = device::CoreParams;
};

/** `DecodeCore<5>` over seven blocks: its merged output must equal `DecodeCore<5>`'s bit for bit. */
export struct CheckCoreFewSms {
	static constexpr ptx::Image image = attn::image;
	static constexpr std::string_view entry = "attn_check_core_rows5_sms7";
	static constexpr driver::Dim grid = {.x = device::selfcheck::few_sms, .y = 1, .z = 1};
	static constexpr prims::Budget budget = device::core_budget<device::RowTile<5>>();
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = device::CoreParams;
};

/** The rope table every row reads, filled once at start by evaluating `model::rope::rotation` on the device. */
export struct RopeTable {
	static constexpr ptx::Image image = attn::image;
	static constexpr std::string_view entry = "attn_rope_table";
	static constexpr driver::Dim grid = {.x = device::selfcheck::rope_blocks, .y = 1, .z = 1};
	static constexpr prims::Budget budget = device::selfcheck::rope_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	/** The address of `rope_table_bytes` of device memory. */
	using Params = device::Address;
};

/** The weight split over every fp32 value in [0, 1] on the device, into a zeroed `prims::selfcheck::Report`. */
export struct CheckSplit {
	static constexpr ptx::Image image = attn::image;
	static constexpr std::string_view entry = "attn_check_split";
	static constexpr driver::Dim grid = {.x = device::selfcheck::split_blocks, .y = 1, .z = 1};
	static constexpr prims::Budget budget = device::selfcheck::split_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	/** The address of the report. */
	using Params = device::Address;
};

static_assert(driver::Kernel<Append> && driver::Kernel<Prefill> && driver::Kernel<DecodeMerge>);
static_assert(driver::Kernel<DecodeCore<1>> && driver::Kernel<DecodeCore<2>> && driver::Kernel<DecodeCore<3>>);
static_assert(driver::Kernel<DecodeCore<4>> && driver::Kernel<DecodeCore<5>>);
static_assert(driver::Kernel<U21OnePart<3>> && driver::Kernel<U21OnePart<5>> && driver::Kernel<CheckCoreFewSms>);
static_assert(driver::Kernel<RopeTable> && driver::Kernel<CheckSplit>);

}
