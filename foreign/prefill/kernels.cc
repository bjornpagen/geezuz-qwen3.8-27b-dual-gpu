// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill's ptx image and the declaration of every kernel in it: its entry, grid, budget, residency and the one
// parameter struct it takes. A row-wise kernel's budget is the layout header's; a gemm kernel's is the budget record
// its own image carries (the gemm's shared layout is device code), so the host never restates either.
export module engine.prefill_device:kernels;

import std;
import engine.core;
import engine.driver;
import engine.prims;
import engine.ptx;
import :layout;

namespace engine::prefill_device {

namespace prims = engine::device::prims;

constexpr char text[] = {
#embed "prefill.ptx" suffix(, '\0')
};

/** Prefill's ptx image. */
export inline constexpr ptx::Image image = ptx::image(text);

static_assert(image.header == ptx::abi.header, "every image shares the engine's isa and target");

/** A kernel entry's name in fixed storage, NUL-terminated, so it can be a template argument and reach the loader. */
export struct Entry {
	std::array<char, 40> text{};
	std::size_t size = 0;

	consteval Entry(char const* name) {
		while (name[size] != '\0') {
			core::invariant(size + 1 < text.size());
			text.at(size) = name[size];
			++size;
		}
	}

	[[nodiscard]] constexpr auto view() const -> std::string_view {
		return {text.data(), size};
	}
};

/** A row-wise kernel: four blocks per SM, a warp per (row, 128-feature tile). */
export template<Entry name, class P>
struct RowKernel {
	static constexpr ptx::Image image = prefill_device::image;
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = {.x = rows_blocks, .y = 1, .z = 1};
	static constexpr prims::Budget budget = rows_budget;
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = P;

	static_assert(ptx::recorded_budget(image, entry).threads == budget.threads &&
	                  ptx::recorded_budget(image, entry).blocks_per_sm == budget.blocks_per_sm,
	              "the image's record and the layout's budget agree");
};

/** A gemm kernel: one block per SM of the decode's warp roles, its budget the image's record. */
export template<Entry name, class P>
struct GemmKernel {
	static constexpr ptx::Image image = prefill_device::image;
	static constexpr std::string_view entry = name.view();
	static constexpr driver::Dim grid = {.x = gemm_blocks, .y = 1, .z = 1};
	static constexpr prims::Budget budget = ptx::recorded_budget(image, entry);
	static constexpr driver::Residency residency = driver::Residency::Ordinary;
	using Params = P;
};

export using Zero = RowKernel<"prefill_zero", ZeroParams>;
export using Embed = RowKernel<"prefill_embed", EmbedParams>;
export using Norm = RowKernel<"prefill_norm", NormParams>;
export using Decay = RowKernel<"prefill_decay", DecayParams>;
export using ResidualAdd = RowKernel<"prefill_residual_add", ResidualParams>;
export using ResidualStart = RowKernel<"prefill_residual_start", ResidualParams>;
export using MtpGather = RowKernel<"prefill_mtp_gather", MtpParams>;
export using MtpNorm = RowKernel<"prefill_mtp_norm", MtpParams>;
export using Keep = RowKernel<"prefill_keep", KeepParams>;
export using CheckWords = RowKernel<"prefill_check_words", CheckParams>;

export using DeltanetIn = GemmKernel<"prefill_deltanet_in", GemmParams>;
export using DeltanetZ = GemmKernel<"prefill_deltanet_z", GemmParams>;
export using DeltanetOut = GemmKernel<"prefill_deltanet_out", PushParams>;
export using AttentionQuery = GemmKernel<"prefill_attention_query", GemmParams>;
export using AttentionKey = GemmKernel<"prefill_attention_key", GemmParams>;
export using AttentionValue = GemmKernel<"prefill_attention_value", GemmParams>;
export using AttentionGate = GemmKernel<"prefill_attention_gate", GemmParams>;
export using AttentionOut = GemmKernel<"prefill_attention_out", PushParams>;
export using MlpIn = GemmKernel<"prefill_mlp_in", GemmParams>;
export using MlpOut = GemmKernel<"prefill_mlp_out", PushParams>;
export using MtpFc = GemmKernel<"prefill_mtp_fc", PushParams>;
export using MtpAttentionQuery = GemmKernel<"prefill_mtp_attention_query", GemmParams>;
export using MtpAttentionKey = GemmKernel<"prefill_mtp_attention_key", GemmParams>;
export using MtpAttentionValue = GemmKernel<"prefill_mtp_attention_value", GemmParams>;
export using MtpAttentionGate = GemmKernel<"prefill_mtp_attention_gate", GemmParams>;
export using MtpAttentionOut = GemmKernel<"prefill_mtp_attention_out", PushParams>;
export using MtpMlpIn = GemmKernel<"prefill_mtp_mlp_in", GemmParams>;
export using MtpMlpOut = GemmKernel<"prefill_mtp_mlp_out", PushParams>;

static_assert(driver::Kernel<Zero> && driver::Kernel<Embed> && driver::Kernel<Norm> && driver::Kernel<Decay> &&
              driver::Kernel<ResidualAdd> && driver::Kernel<ResidualStart> && driver::Kernel<MtpGather> && driver::Kernel<MtpNorm> &&
              driver::Kernel<Keep> && driver::Kernel<CheckWords>);
static_assert(DeltanetIn::budget.threads == MlpOut::budget.threads && MtpFc::budget.blocks_per_sm == 1,
              "the gemms run the decode's roles, one block per SM");

/** Every kernel of the image, as one program. */
export using Program =
    driver::Program<Zero, Embed, Norm, Decay, ResidualAdd, ResidualStart, MtpGather, MtpNorm, Keep, CheckWords, DeltanetIn, DeltanetZ,
	                DeltanetOut, AttentionQuery, AttentionKey, AttentionValue, AttentionGate, AttentionOut, MlpIn, MlpOut, MtpFc,
	                MtpAttentionQuery, MtpAttentionKey, MtpAttentionValue, MtpAttentionGate, MtpAttentionOut, MtpMlpIn, MtpMlpOut>;

}
