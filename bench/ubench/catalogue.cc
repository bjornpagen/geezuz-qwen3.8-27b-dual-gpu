// SPDX-License-Identifier: AGPL-3.0-only
//
// The closed set of microbenchmarks, what each answers, and the review table they answer. A benchmark either answers
// one of the adversarial review's open questions, whose expectation and decision the runner reads from the review
// itself, or measures a required primitive the review's table does not cover and states its own. Every question in
// the review has exactly one benchmark; the build checks it.
export module engine.ubench:catalogue;

import std;
import engine.core;
import engine.ubench_device;

namespace engine::ubench {

export enum class Benchmark : std::uint8_t {
	ReadStream,
	TileLayout,
	MemoryClock,
	TensorRate,
	DecodeExact,
	BarrierCost,
	CrossSmHop,
	PeerFlag,
	PeerPush,
	ReleaseCost,
	LinkHealth,
	HostFetch,
	L2Persistence,
	RingStall,
	SustainedClock,
	OffsetStep,
	TranscendentalRate,
	GemvRows,
	ProbeMma,
	DeltanetChain,
	SplitKAttention,
	PrefillGemm,
	StageHandoff,
	L2PolicyReuse,
	LdmatrixSwizzle,
	MmaDoubleBuffer,
	WarpReductions,
	CheckFragments,
	CheckReduce,
	CheckTranscendentals,
	CheckStageRing,
	CheckOrder,
};

/** A benchmark that answers the review's open question `number` (its U-number). */
export struct Question {
	int number;

	[[nodiscard]] constexpr auto operator==(Question const&) const -> bool = default;
};

/** A benchmark of a primitive the review's table does not cover: what it expects and what it decides. */
export struct Primitive {
	std::string_view expected;
	std::string_view settles;
};

/**
 * One benchmark: its command-line name, what it answers, the evidence behind its expectation (the hardware notes'
 * sections, confidence labels and sources), and what the operator must set by hand before running it, if anything.
 */
export struct Entry {
	Benchmark benchmark;
	std::string_view name;
	std::variant<Question, Primitive> answers;
	std::string_view basis;
	std::optional<std::string_view> operator_sets;
};

inline constexpr std::string_view root_links = "root: persistence mode on (`nvidia-smi -pm 1`) and the bridge's links reading NV4 in "
                                               "`nvidia-smi topo -m`; stock driver, no p2p patch";

export inline constexpr auto catalogue = std::to_array<Entry>({
    {.benchmark = Benchmark::ReadStream,
     .name = "read_stream",
     .answers = Question{1},
     .basis = "hw-dram §2: 3090 plateau 890-900 GB/s measured by others (Chips and Cheese); A40 pure read at 97.7% of peak "
              "measured by others (gpu-benches); 93-96% of the memory clock's peak, and p2's 912 GB/s, inferred. The verdict is "
              "the phase-b gate: the best point at 93% or more of the measured clock's peak",
     .operator_sets = "root: persistence mode on (`nvidia-smi -pm 1`); for p0, one of memory_clock's three attempts; any -lgc or "
                      "-lmc lock in force is the operator's to note, the runner reads only the memory clock and p-state"},
    {.benchmark = Benchmark::TileLayout,
     .name = "tile_layout",
     .answers = Question{2},
     .basis = "hw-dram §1: a 256-byte chunk maps to one bank and row, tRC about 45 ns on GDDR6, measured by others (GPUHammer, "
              "A6000); the contiguous-tile rule inferred",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::MemoryClock,
     .name = "memory_clock",
     .answers = Question{3},
     .basis = "hw-power §3: CUDA work runs in p2 at 9501 MHz, measured by others (three reports); 9751 MHz is p0's maximum, "
              "documented",
     .operator_sets = "root, one at a time, rerunning this benchmark after each: `nvidia-smi -lgc F,F -lmc 9751,9751` in one call; "
                      "`nvidia-smi --cuda-clocks=1`; nvidia-modeset `disable_vrr_memclk_switch=1`. Never CUDA_DISABLE_PERF_BOOST"},
    {.benchmark = Benchmark::TensorRate,
     .name = "tensor_rate",
     .answers = Question{4},
     .basis = "hw-sm §2-3: 512 flop/clk/SM datasheet [WP] and inferred, 505 measured by others on GA104 [SUN]; fma.rn.bf16x2 "
              "128 results/clk/SM documented [CG12.6]; lop3 64/clk/SM documented, prmt 64 inferred; the pipe hfma2 issues on is "
              "unknown",
     .operator_sets = "root: lock the SM clock (`nvidia-smi -lgc F,F`) so a per-clock rate holds over the run"},
    {.benchmark = Benchmark::DecodeExact,
     .name = "decode_exact",
     .answers = Question{7},
     .basis = "hw-sm §6: sequence B exact over all 254 finite codes, verified by simulation [SIM] and by the host model's "
              "exhaustive test; fma.bf16x2 keeps subnormals, documented [PTX]. The exact path is the prims self-check, a gate; "
              "the folded variant is measured beside it",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::BarrierCost,
     .name = "barrier_cost",
     .answers = Question{8},
     .basis = "hw-sm §7: mbarrier and named-barrier costs unpublished for sm_86; ~60 ns per async barrier measured by others on "
              "H100 (Hazy); nanosleep sleeps within [0, 2t], documented [PTX]. Times convert cycles at the calibrated SM clock",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::CrossSmHop,
     .name = "cross_sm_hop",
     .answers = Question{9},
     .basis = "hw-dram §3: unloaded l2 latency about 138 ns on the 3090 and 144 ns on an A40, measured by others; several "
              "hundred ns per hop inferred. The verdict checks the protocol: every echoed value must be the one released",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::PeerFlag,
     .name = "peer_flag",
     .answers = Question{10},
     .basis = "hw-link §3: one-way remote store ~0.6-1.0 us and flag handoff ~1-2 us, inferred (±2x) from a GB200 measurement; "
              "upstream measured fence.sc.sys + flag at 28 us per call against 24 us for st.release.sys (hw-link §4). Four "
              "variants: release/acquire with the engine's backoff and spinning, relaxed (the floor) and fence.sc.sys-fenced",
     .operator_sets = root_links},
    {.benchmark = Benchmark::PeerPush,
     .name = "peer_push",
     .answers = Question{11},
     .basis = "hw-link §1: 52.6-52.75 GB/s per direction with the copy engines, measured by others; 4 x 14.0625 = 56.25 GB/s "
              "datasheet [WP]; the SM-store rate inferred",
     .operator_sets = root_links},
    {.benchmark = Benchmark::ReleaseCost,
     .name = "release_cost",
     .answers = Question{12},
     .basis = "hw-link §4: fence interference documented (CUDA guide); sm_86 has one memory-sync domain; the cost itself unknown",
     .operator_sets = root_links},
    {.benchmark = Benchmark::LinkHealth,
     .name = "link_health",
     .answers = Question{13},
     .basis = "hw-link §1-2: 4 links x 14.0625 GB/s per direction datasheet [WP], NV4 measured by others; native peer atomics "
              "not established. The engine's start refuses a machine whose eight links do not all join the peer; this reads "
              "each link's version (raw, its NVML numbering inferred) and speed, and the peer path's rank and atomics",
     .operator_sets = "the open kernel module first, then the proprietary module of the same version; iommu=pt; persistence mode "
                      "on; confirm the raw link version against `nvidia-smi nvlink -s` and `topo -m` once"},
    {.benchmark = Benchmark::HostFetch,
     .name = "host_fetch",
     .answers = Question{14},
     .basis = "hw-link §5: a mapped load around 1+ us and a host-to-device flag ~1-2 us, measured by others; ~24 GB/s effective "
              "measured by others, 31.5 GB/s raw inferred",
     .operator_sets = "run once under `numactl --cpunodebind=0 --membind=0` and once with `--membind=1`: the mapped table and the "
                      "mailbox ring live where the process's memory policy puts them"},
    {.benchmark = Benchmark::L2Persistence,
     .name = "l2_persistence",
     .answers = Question{15},
     .basis = "hw-sm §4: persisting l2 about 4.5 MB (75% of 6 MiB) inferred; 6144 KB of l2 datasheet [WP]. The driver's "
              "persisting and window limits are read; survival is each 128 KiB segment's read-back time against the same "
              "segment hot and a cold one",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::RingStall,
     .name = "ring_stall",
     .answers = Question{16},
     .basis = "hw-sm implication 5: 64-80 KiB per SM is 5.7-7.0 us of run-ahead at 936 GB/s, inferred",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::SustainedClock,
     .name = "sustained_clock",
     .answers = Question{18},
     .basis = "hw-power §1-2: ~1800-1900 MHz under water at 400 W inferred; 1710 MHz at 350 W on air measured by others. A "
              "10-minute soak of both cards: the SM clock measured on the card launch by launch, NVML's clock and power "
              "sampled, and the software power cap's counter across the soak",
     .operator_sets = "root: `nvidia-smi -pm 1`, `nvidia-smi -i N -pl 400`; run unlocked first, then with `nvidia-smi -lgc F,F` "
                      "one step below the sustained clock, accepting the lock only if the power cap's counter stays flat"},
    {.benchmark = Benchmark::OffsetStep,
     .name = "offset_step",
     .answers = Question{19},
     .basis = "hw-dram §5: mining-stable offsets reach 21.5-22.6 Gbps per pin, measured by others (low quality); the in-rating "
              "ceiling about 10501 MHz inferred. One run is one step: the knee is where efficiency against the clock's peak "
              "falls across steps",
     .operator_sets = "root, one step per run at operating temperature: +250 Linux units (+125 MHz) through "
                      "nvmlDeviceSetMemClkVfOffset or `nvidia-settings -a '[gpu:N]/GPUMemoryTransferRateOffset[<top level>]=<offset>'` "
                      "(an offset breaks a -lmc lock); junction temperature with the gddr6 tool only under iomem=relaxed"},
    {.benchmark = Benchmark::TranscendentalRate,
     .name = "transcendental_rate",
     .answers = Question{20},
     .basis = "hw-sm §2: MUFU 16/clk/SM and fp32 fma 128/clk/SM documented [CG12.6 table 4]; the shared routines' 10-30 fp32 "
              "operations from their specification. The mufu side cannot be measured: the ptx gate refuses every .approx "
              "instruction, so no engine image holds one; the documented rate stands in",
     .operator_sets = "root: lock the SM clock (`nvidia-smi -lgc F,F`)"},
    {.benchmark = Benchmark::GemvRows,
     .name = "gemv_rows",
     .answers = Question{5},
     .basis = "hw-sm §6: 128 of ~152-173 tensor clocks per weight fragment at 32 rows, 160 at 40, inferred; the gemv component's "
              "own streaming kernel over down_proj's whole placement at 8 to 40 rows",
     .operator_sets = "root: lock the SM clock (`nvidia-smi -lgc F,F`): the crossover moves with the core clock"},
    {.benchmark = Benchmark::ProbeMma,
     .name = "probe_mma",
     .answers = Question{6},
     .basis = "hw-sm §3: 8-product blocks, truncation, a 25-bit window and at least 3 carry bits, measured by others [GM §IV] "
              "on an RTX 3060 and identical on an A100 [FASI]; 256 seeded random probes, every lane's D compared bit for bit "
              "with engine.mma_model",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::DeltanetChain,
     .name = "deltanet_chain",
     .answers = Question{17},
     .basis = "prior-art §7: upstream's core ~80 us per layer at 8 slots, its state floor 27 us at 936 GB/s; ~300 clocks per "
              "token per chain from the review's arithmetic. The deltanet component's chain probes, one-slot prefill and "
              "8-slot decode core",
     .operator_sets = "root: lock the SM clock (`nvidia-smi -lgc F,F`)"},
    {.benchmark = Benchmark::SplitKAttention,
     .name = "split_k_attention",
     .answers = Question{21},
     .basis = "the review's arithmetic only (no hardware source): exact P's two extra PV mmas ride under the kv stream at 3 "
              "rows per slot and approach the ridge at 5. The attention component's decode core against its one-part twin, "
              "4096 positions per slot",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::PrefillGemm,
     .name = "prefill_gemm",
     .answers = Question{22},
     .basis = "prior-art §0: Marlin ~63 TFLOPS at 250 W and FlashAttention-2 54-57 TFLOPS, measured by others; 71 TFLOPS "
              "ceiling at 1695 MHz datasheet [WP]. The gemv component's prefill gemm over 2048 rows",
     .operator_sets = "root: `nvidia-smi -pl 400` and the SM clock lock"},
    {.benchmark = Benchmark::StageHandoff,
     .name = "stage_handoff",
     .answers =
         Primitive{
             .expected = "unpublished for sm_86; well under one tile's stream time per SM (16 KiB at ~11 GB/s per SM, about 1.5 us)",
             .settles = "warp specialization: whether handing a stage from the producer warp to two consumer warpgroups costs "
                        "anything beside the tile's transfer",
         },
     .basis = "inferred from hw-dram §2 (each SM's share of the stream)",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::L2PolicyReuse,
     .name = "l2_policy_reuse",
     .answers =
         Primitive{
             .expected = "activation loads near l2 latency (about 138 ns) under the engine's split; slower under uniform "
                         "priorities, where the stream evicts them",
             .settles = "l2 control: evict_first on every single-use stream and evict_last on what is reused within microseconds",
         },
     .basis = "hw-sm §4 and hw-dram implication 2, inferred; the l2 latency measured by others (hw-dram §3)",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::LdmatrixSwizzle,
     .name = "ldmatrix_swizzle",
     .answers =
         Primitive{
             .expected = "swizzled: 0.25 ldmatrix.x4 per clock per SM (512 bytes at 128 B/clk/SM); row-major: eight-way bank "
                         "conflicts, about 1/32",
             .settles = "fragments: the xor swizzle on activation tiles",
         },
     .basis = "hw-sm §1: shared memory 128 B/clk/SM in 32 banks, documented [CG12.6]; the conflict count inferred",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::MmaDoubleBuffer,
     .name = "mma_double_buffer",
     .answers =
         Primitive{
             .expected = "double-buffered near the mma rate (512 flop/clk/SM); sequential below it by the exposed ldmatrix latency",
             .settles = "fragments: register double buffering across k-steps",
         },
     .basis = "hw-sm §3: 512 flop/clk/SM datasheet [WP]; peak needs two warps or two independent chains per scheduler, "
              "measured by others [SUN]",
     .operator_sets = "root: lock the SM clock (`nvidia-smi -lgc F,F`)"},
    {.benchmark = Benchmark::WarpReductions,
     .name = "warp_reductions",
     .answers =
         Primitive{
             .expected = "unpublished; the float butterfly at most 0.2 warp reductions per clock per SM (five shuffles, at one warp "
                         "shuffle per clock per SM)",
             .settles = "reductions: the fixed-order butterfly and redux.sync cost too little to matter beside the stream",
         },
     .basis = "inferred from the shuffle rate (32 results/clk/SM) the CUDA guide documents for compute capability 8.x [CG12.6]",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::CheckFragments,
     .name = "check_fragments",
     .answers =
         Primitive{
             .expected = "zero mismatches",
             .settles = "the fragment maps the baker and the kernels share hold for ldmatrix and .trans on the hardware",
         },
     .basis = "the fragment maps' host proofs; the ldmatrix layout documented [PTX]",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::CheckReduce,
     .name = "check_reduce",
     .answers =
         Primitive{
             .expected = "zero mismatches",
             .settles = "the warp reductions are exact and order-fixed on the hardware",
         },
     .basis = "the reductions' fixed butterfly order, recomputed per lane on the device",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::CheckTranscendentals,
     .name = "check_transcendentals",
     .answers =
         Primitive{
             .expected = "the device's digest equals the host's for every function and block (the host side takes minutes)",
             .settles = "the kernels compute exactly the reference's transcendental bits over every fp32 input",
         },
     .basis = "the transcendentals header, proven on the host over every fp32 input",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::CheckStageRing,
     .name = "check_stage_ring",
     .answers =
         Primitive{
             .expected = "zero mismatches; each checked tile well under a microsecond",
             .settles = "the cp.async ring with mbarrier completion delivers every word in order on the hardware",
         },
     .basis = "the stage ring's phase arithmetic, proven on the host; mbarrier and cp.async semantics documented [PTX]",
     .operator_sets = std::nullopt},
    {.benchmark = Benchmark::CheckOrder,
     .name = "check_order",
     .answers =
         Primitive{
             .expected = "zero mismatches",
             .settles = "release and acquire across SMs, the epoch flags, the peer protocol, the fenced relaxed flag and the "
                        "remaining primitives hold on the hardware",
         },
     .basis = "the memory model's release/acquire and fence.sc semantics, documented [PTX]",
     .operator_sets = std::nullopt},
});

static_assert(std::ranges::all_of(std::views::iota(std::size_t{0}, catalogue.size()),
                                  [](std::size_t index) {
	                                  return std::to_underlying(catalogue.at(index).benchmark) == index;
                                  }),
              "the catalogue lists every benchmark once, in the enum's order");
static_assert(catalogue.size() == std::to_underlying(Benchmark::CheckOrder) + 1U);

export [[nodiscard]] constexpr auto entry(Benchmark benchmark) -> Entry const& {
	return catalogue.at(std::to_underlying(benchmark));
}

/** The benchmark a command-line name names; nothing for a name that is not one. */
export [[nodiscard]] constexpr auto parse(std::string_view name) -> std::optional<Benchmark> {
	for (auto const& candidate : catalogue) {
		if (candidate.name == name) {
			return candidate.benchmark;
		}
	}
	return std::nullopt;
}

/** One row of the review's microbenchmark table. */
export struct Row {
	int question;
	std::string_view measures;
	std::string_view expected;
	std::string_view settles;
};

/** The review's microbenchmark table, in its order. */
export struct Table {
	std::array<Row, 64> rows;
	std::size_t count;

	/** The rows, borrowed from this table; a temporary table has no view to give, so none can dangle. */
	[[nodiscard]] constexpr auto view() const& -> std::span<Row const> {
		return std::span{rows}.first(count);
	}

	auto view() const&& -> std::span<Row const> = delete;
};

[[nodiscard]] constexpr auto trimmed(std::string_view text) -> std::string_view {
	while (!text.empty() && text.front() == ' ') {
		text.remove_prefix(1);
	}
	while (!text.empty() && text.back() == ' ') {
		text.remove_suffix(1);
	}
	return text;
}

/**
 * The table under the review's "microbenchmarks for lane `ubench`" heading: every line `| U<n> | measures | expected
 * (source) | settles |` up to the next heading. A review without that section, or a row of another shape, is a
 * malformed document.
 */
export [[nodiscard]] constexpr auto review_table(std::string_view document) -> Table {
	constexpr auto heading = std::string_view{"## microbenchmarks for lane `ubench`"};
	auto const at = document.find(heading);
	core::invariant(at != std::string_view::npos);
	auto section = document.substr(at + heading.size());
	section = section.substr(0, section.find("\n## "));
	auto table = Table{.rows = {}, .count = 0};
	while (!section.empty()) {
		auto const end = section.find('\n');
		auto const line = section.substr(0, end);
		section = end == std::string_view::npos ? std::string_view{} : section.substr(end + 1);
		if (!line.starts_with("| U")) {
			continue;
		}
		auto cells = std::array<std::string_view, 4>{};
		auto rest = line.substr(1);
		for (auto& cell : cells) {
			auto const bar = rest.find('|');
			core::invariant(bar != std::string_view::npos);
			cell = trimmed(rest.substr(0, bar));
			rest = rest.substr(bar + 1);
		}
		core::invariant(trimmed(rest).empty() && cells[0].size() >= 2);
		auto number = 0;
		for (auto const digit : cells[0].substr(1)) {
			core::invariant(digit >= '0' && digit <= '9');
			number = number * 10 + (digit - '0');
		}
		core::invariant(table.count < table.rows.size());
		table.rows.at(table.count++) = {.question = number, .measures = cells[1], .expected = cells[2], .settles = cells[3]};
	}
	return table;
}

/** The review this build embeds, parsed. */
export [[nodiscard]] constexpr auto review() -> Table {
	return review_table(ubench_device::adversarial);
}

/** How many benchmarks answer question `number`. */
export [[nodiscard]] constexpr auto answers(int number) -> std::size_t {
	return static_cast<std::size_t>(std::ranges::count_if(catalogue, [&](Entry const& candidate) {
		auto const* const question = std::get_if<Question>(&candidate.answers);
		return question != nullptr && question->number == number;
	}));
}

/**
 * Whether `table` and the catalogue agree: every question in the table has exactly one benchmark, and no benchmark
 * answers a question the table does not ask.
 */
export [[nodiscard]] constexpr auto covers(Table const& table) -> bool {
	auto const rows = table.view();
	auto const each_once = std::ranges::all_of(rows, [](Row const& row) {
		return answers(row.question) == 1;
	});
	auto const no_strays = std::ranges::all_of(catalogue, [&](Entry const& candidate) {
		auto const* const question = std::get_if<Question>(&candidate.answers);
		return question == nullptr || std::ranges::count(rows, question->number, &Row::question) == 1;
	});
	return each_once && no_strays;
}

static_assert(covers(review()), "every question in the review has exactly one benchmark");

/** The review's row for question `number`, which the table holds exactly once. */
export [[nodiscard]] constexpr auto row(Table const& table, int number) -> Row {
	auto const rows = table.view();
	auto const found = std::ranges::find(rows, number, &Row::question);
	core::invariant(found != rows.end());
	return *found;
}

}
