// SPDX-License-Identifier: AGPL-3.0-only
//
// The benchmarks over the engine's own building blocks, whose kernels the gemv, deltanet and attention components
// build: the streaming gemv across row counts, the prefill gemm, the deltanet chains and cores, and decode attention
// with the exact P split against one bf16 part. Every buffer is sized by its component's geometry; the contents are
// whatever the allocation held, because these time the kernels and do not check what they compute (each component's
// self-checks do that).
export module engine.ubench:lanes;

import std;
import engine.attn;
import engine.core;
import engine.deltanet;
import engine.device;
import engine.gemv;
import engine.model;
import engine.prims;
import engine.ubench_device;
import :outcome;
import :protocol;

namespace engine::ubench {

namespace kernels = engine::ubench_device;
namespace deltanet = engine::deltanet;
namespace attn_device = engine::device::attn;

using model::Card;

/** One row sweep point of the streaming gemv on one card: its best time and its slowest block's cycles. */
export struct GemvPoint {
	Card card;
	int rows;
	double milliseconds;
	double slowest_block_cycles;
};

/** down_proj's placement on one card, the bytes every gemv point streams. */
inline constexpr auto down_bytes = static_cast<double>(kernels::placement_chunks<engine::gemv::down>) * 16.0;

export [[nodiscard]] auto evaluate_gemv_rows(std::span<GemvPoint const> points, Calibration calibration) -> Result<Outcome> {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& point : points) {
		if (!(point.slowest_block_cycles > 0.0)) {
			return no_reading("gemv_stream_r1", "a block reported no elapsed cycles");
		}
		auto const label = std::format("{}, {} rows", on(point.card), point.rows);
		outcome.readings.push_back({.label = label + ", down_proj streamed", .value = gigabytes_per_second(down_bytes, point.milliseconds), .unit = Unit::GigabytesPerSecond});
		outcome.readings.push_back({.label = label + ", slowest block", .value = calibration.nanoseconds(point.card, point.slowest_block_cycles) / 1000.0, .unit = Unit::Microseconds});
	}
	outcome.notes.emplace_back("the row budget is the most rows at which the stream rate holds flat");
	return outcome;
}

/** The slowest block's 32-bit elapsed cycles. */
[[nodiscard]] auto slowest(std::span<unsigned const> blocks) -> double {
	return static_cast<double>(std::ranges::max(blocks));
}

export [[nodiscard]] auto gemv_rows(BenchView const& bench) -> Result<Outcome> {
	auto points = std::vector<GemvPoint>{};
	auto const ran = for_each_card([&](Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = std::apply(
		    [&](auto... sweep) {
			    return lift(view.context.load<decltype(sweep)...>());
		    },
		    kernels::GemvSweep{});
		auto const tiles = lift(view.context.allocate<engine::device::prims::Chunk, kernels::placement_chunks<engine::gemv::down>>());
		auto const activations = scratch<engine::device::prims::Chunk, kernels::StreamActivations::bytes / 16>(view.context);
		auto const scales = scratch<unsigned short, kernels::StreamScales::bytes / 2>(view.context);
		auto const fixups = scratch<float, kernels::StreamFixups::bytes / 4>(view.context);
		auto const out = scratch<float, kernels::StreamOut::bytes / 4>(view.context);
		auto flags = Readback<unsigned long long, engine::gemv::selfcheck::blocks>::create(view.context);
		auto elapsed = Readback<unsigned, engine::gemv::selfcheck::blocks>::create(view.context);
		if (auto const failed = first_failure(program, tiles, activations, scales, fixups, out, flags, elapsed)) {
			return std::unexpected(*failed);
		}
		return for_each_shape<kernels::GemvSweep>([&](auto kernel) -> Result<void> {
			using K = decltype(kernel);
			// The fixup flags count epochs from zero, so every launch starts them over.
			auto const clear = [&] {
				return flags->clear(view);
			};
			auto const took = best_time<K>(view, *program, clear, *tiles, *activations, *scales, *fixups, flags->device(), *out, elapsed->device());
			auto const blocks = took.and_then([&](double) {
				return elapsed->fetch(view);
			});
			return blocks.transform([&](std::span<unsigned const, engine::gemv::selfcheck::blocks> cycles) {
				points.push_back({.card = card, .rows = 8 * K::row_tiles, .milliseconds = *took, .slowest_block_cycles = slowest(cycles)});
			});
		});
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	return evaluate_gemv_rows(points, bench.calibration);
}

/** One prefill gemm's best time on one card. */
export struct GemmPoint {
	Card card;
	kernels::PrefillMatrix which;
	double milliseconds;
};

/** The flop of `rows` rows through `matrix`'s whole placement: every output of every tensor, over its k extent. */
[[nodiscard]] constexpr auto gemm_flop(engine::gemv::Matrix const& matrix, double rows) -> double {
	auto const outputs = static_cast<double>(matrix.output_tiles * matrix.tensors * engine::gemv::tile_rows);
	return 2.0 * rows * outputs * static_cast<double>(engine::gemv::k_extent(matrix));
}

export [[nodiscard]] auto evaluate_prefill_gemm(std::span<GemmPoint const> points) -> Outcome {
	constexpr auto rows = static_cast<double>(engine::gemv::selfcheck::prefill_rows);
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& point : points) {
		auto const& matrix = point.which == kernels::PrefillMatrix::Down ? engine::gemv::down : engine::gemv::gate_up;
		outcome.readings.push_back({.label = std::format("{}, {} rows through {}", on(point.card), engine::gemv::selfcheck::prefill_rows,
		                                                 point.which == kernels::PrefillMatrix::Down ? "down_proj" : "gate_proj and up_proj"),
		                            .value = gemm_flop(matrix, rows) / (point.milliseconds * 1.0e9),
		                            .unit = Unit::Teraflops});
	}
	outcome.notes.emplace_back("the planning number is 1,600 prompt tokens per second at ~50 TFLOPS in situ, 2,000 at ~60");
	return outcome;
}

template<kernels::PrefillMatrix which>
[[nodiscard]] auto time_gemm(CardView view) -> Result<double> {
	using K = kernels::GemmPrefillKernel<which>;
	using Buffers = K::Buffers;
	auto const program = lift(view.context.load<K>());
	auto const tiles = lift(view.context.allocate<engine::device::prims::Chunk, Buffers::Tiles::bytes / 16>());
	auto const activations = scratch<engine::device::prims::Chunk, Buffers::Activations::bytes / 16>(view.context);
	auto const gate_scales = scratch<unsigned short, Buffers::Scales::bytes / 2>(view.context);
	auto const up_scales = scratch<unsigned short, Buffers::Scales::bytes / 2>(view.context);
	auto const out = scratch<unsigned, Buffers::Out::bytes / 4>(view.context);
	auto const elapsed = scratch<unsigned, engine::gemv::selfcheck::blocks>(view.context);
	if (auto const failed = first_failure(program, tiles, activations, gate_scales, up_scales, out, elapsed)) {
		return std::unexpected(*failed);
	}
	return best_time<K>(view, *program, nothing, *tiles, *activations, *gate_scales, *up_scales, *out, *elapsed);
}

export [[nodiscard]] auto prefill_gemm(BenchView const& bench) -> Result<Outcome> {
	auto points = std::vector<GemmPoint>{};
	auto const ran = for_each_card([&](Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const down = time_gemm<kernels::PrefillMatrix::Down>(view);
		auto const gate_up = down.and_then([&](double) {
			return time_gemm<kernels::PrefillMatrix::GateUp>(view);
		});
		return gate_up.transform([&](double milliseconds) {
			points.push_back({.card = card, .which = kernels::PrefillMatrix::Down, .milliseconds = *down});
			points.push_back({.card = card, .which = kernels::PrefillMatrix::GateUp, .milliseconds = milliseconds});
		});
	});
	return ran.transform([&] {
		return evaluate_prefill_gemm(points);
	});
}

/** Tokens the chain probe runs every chain through. */
inline constexpr unsigned probe_tokens = 1024;
inline constexpr auto probe_floats = static_cast<std::size_t>(deltanet::probe_heads) * probe_tokens * deltanet::dim;
inline constexpr auto probe_cycle_words = 2 * static_cast<std::size_t>(std::max(deltanet::probe_blocks<deltanet::Column>, deltanet::probe_blocks<deltanet::Tree<32>>));

/** One card's deltanet readings: each probe's median cycles per token, the all-chains time, prefill and decode core. */
export struct DeltanetReading {
	Card card;
	double eight_lane_cycles_per_token;
	double warp_cycles_per_token;
	double eight_lane_milliseconds;
	double prefill_milliseconds;
	double decode_core_milliseconds;
};

/** The median block's cycles per token from a probe's per-block start and end counts. */
export [[nodiscard]] auto cycles_per_token(std::span<unsigned long long const> counts, std::size_t blocks, std::string_view kernel) -> Result<double> {
	auto per_token = std::vector<double>{};
	for (auto block = std::size_t{0}; block < blocks; ++block) {
		auto const started = counts[2 * block];
		auto const finished = counts[2 * block + 1];
		if (finished <= started) {
			return no_reading(kernel, "a block's chain took no cycles");
		}
		per_token.push_back(static_cast<double>(finished - started) / probe_tokens);
	}
	return median(std::move(per_token));
}

/** Tokens of the one-slot prefill chunk. */
inline constexpr unsigned prefill_tokens = 2048;
/** Rows per slot in the decode core: the committed token and two drafts, the 8-slot, 24-row cycle. */
inline constexpr unsigned decode_rows_per_slot = 3;

export [[nodiscard]] auto evaluate_deltanet_chain(std::span<DeltanetReading const> readings, Calibration calibration) -> Outcome {
	constexpr auto chains = static_cast<double>(deltanet::probe_heads) * deltanet::dim;
	constexpr auto state_moved = static_cast<double>(model::slots) * deltanet::value_heads * static_cast<double>(deltanet::head_state_bytes) * 2.0;
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		auto const card = reading.card;
		outcome.readings.push_back({.label = std::format("{}, one chain's token, eight lanes per column", on(card)), .value = reading.eight_lane_cycles_per_token, .unit = Unit::Cycles});
		outcome.readings.push_back({.label = std::format("{}, one chain's token, a warp per column", on(card)), .value = reading.warp_cycles_per_token, .unit = Unit::Cycles});
		outcome.readings.push_back({.label = std::format("{}, one chain's token, eight lanes per column", on(card)),
		                            .value = calibration.nanoseconds(card, reading.eight_lane_cycles_per_token),
		                            .unit = Unit::Nanoseconds});
		outcome.readings.push_back({.label = std::format("{}, all {:.0f} chains of a layer, chain-tokens", on(card), chains),
		                            .value = chains * probe_tokens / (reading.eight_lane_milliseconds * 1000.0),
		                            .unit = Unit::PerMicrosecond});
		outcome.readings.push_back({.label = std::format("{}, prefill of one slot, {} tokens, one layer", on(card), prefill_tokens),
		                            .value = reading.prefill_milliseconds,
		                            .unit = Unit::Milliseconds});
		outcome.readings.push_back({.label = std::format("{}, decode core, 8 slots x {} rows, one layer", on(card), decode_rows_per_slot),
		                            .value = reading.decode_core_milliseconds * 1000.0,
		                            .unit = Unit::Microseconds});
		outcome.readings.push_back({.label = std::format("{}, decode core's state stream (read and write)", on(card)),
		                            .value = gigabytes_per_second(state_moved, reading.decode_core_milliseconds),
		                            .unit = Unit::GigabytesPerSecond});
	}
	return outcome;
}

using ProbeEight = deltanet::ProbeChain8;
using ProbeWarp = deltanet::ProbeChain32;

/** A deltanet parameter field of type `Field` (an `Address<T>`) at a device address, whatever its element type is. */
template<class Field>
[[nodiscard]] auto field_at(std::uint64_t device_address) -> Field {
	return deltanet::address<std::remove_pointer_t<decltype(Field::value)>>(device_address);
}

/** One layer's side data and activation rows for `rows` rows, as byte buffers. */
template<std::size_t rows>
struct DeltanetRows {
	device::DeviceArray<unsigned char, static_cast<std::size_t>(deltanet::channels) * 4 * 2> conv;
	device::DeviceArray<unsigned char, static_cast<std::size_t>(deltanet::value_heads) * 2> a_log;
	device::DeviceArray<unsigned char, static_cast<std::size_t>(deltanet::value_heads) * 2> dt_bias;
	device::DeviceArray<unsigned char, static_cast<std::size_t>(deltanet::dim) * 2> norm;
	device::DeviceArray<unsigned char, rows * static_cast<std::size_t>(deltanet::qkv_row_bytes)> qkv;
	device::DeviceArray<unsigned char, rows * static_cast<std::size_t>(deltanet::gate_row_bytes)> gates;
	device::DeviceArray<unsigned char, rows * static_cast<std::size_t>(deltanet::output_row_bytes)> outputs;

	[[nodiscard]] static auto create(device::Context const& context) -> Result<DeltanetRows> {
		auto conv = scratch<unsigned char, static_cast<std::size_t>(deltanet::channels) * 4 * 2>(context);
		auto a_log = scratch<unsigned char, static_cast<std::size_t>(deltanet::value_heads) * 2>(context);
		auto dt_bias = scratch<unsigned char, static_cast<std::size_t>(deltanet::value_heads) * 2>(context);
		auto norm = scratch<unsigned char, static_cast<std::size_t>(deltanet::dim) * 2>(context);
		auto qkv = scratch<unsigned char, rows * static_cast<std::size_t>(deltanet::qkv_row_bytes)>(context);
		auto gates = scratch<unsigned char, rows * static_cast<std::size_t>(deltanet::gate_row_bytes)>(context);
		auto outputs = scratch<unsigned char, rows * static_cast<std::size_t>(deltanet::output_row_bytes)>(context);
		if (auto const failed = first_failure(conv, a_log, dt_bias, norm, qkv, gates, outputs)) {
			return std::unexpected(*failed);
		}
		return DeltanetRows{std::move(*conv), std::move(*a_log), std::move(*dt_bias), std::move(*norm), std::move(*qkv), std::move(*gates), std::move(*outputs)};
	}

	[[nodiscard]] auto weights() const -> deltanet::Weights {
		using Weights = deltanet::Weights;
		return {
		    .conv = field_at<decltype(Weights::conv)>(conv.address()),
		    .a_log = field_at<decltype(Weights::a_log)>(a_log.address()),
		    .dt_bias = field_at<decltype(Weights::dt_bias)>(dt_bias.address()),
		    .norm = field_at<decltype(Weights::norm)>(norm.address()),
		};
	}

	[[nodiscard]] auto rows_view() const -> deltanet::Rows {
		using Rows = deltanet::Rows;
		return {
		    .qkv = field_at<decltype(Rows::qkv)>(qkv.address()),
		    .gates = field_at<decltype(Rows::gates)>(gates.address()),
		    .outputs = field_at<decltype(Rows::outputs)>(outputs.address()),
		};
	}
};

/* PIN(clang-tidy-optional-merged-record): device params are built from plain addresses where no result is held */

/** The deltanet benchmark's buffers by device address, and the side data and rows built from them. */
struct DeltanetBuffers {
	std::uint64_t keys;
	std::uint64_t queries;
	std::uint64_t values;
	std::uint64_t gates;
	std::uint64_t state;
	std::uint64_t outputs;
	std::uint64_t cycles;
	std::uint64_t images;
	std::uint64_t rings;
	deltanet::Weights prompt_weights;
	deltanet::Rows prompt_rows;
	deltanet::Weights cycle_weights;
	deltanet::Rows cycle_rows;
};

/** The chain probes' parameters: every chain of a layer through `probe_tokens` tokens. */
[[nodiscard]] auto probe_params(DeltanetBuffers const& buffers) -> deltanet::ProbeParams {
	using Probe = deltanet::ProbeParams;
	return {
	    .keys = field_at<decltype(Probe::keys)>(buffers.keys),
	    .queries = field_at<decltype(Probe::queries)>(buffers.queries),
	    .values = field_at<decltype(Probe::values)>(buffers.values),
	    .gates = field_at<decltype(Probe::gates)>(buffers.gates),
	    .state = field_at<decltype(Probe::state)>(buffers.state),
	    .outputs = field_at<decltype(Probe::outputs)>(buffers.outputs),
	    .cycles = field_at<decltype(Probe::cycles)>(buffers.cycles),
	    .tokens = probe_tokens,
	};
}

/** One slot's prompt chunk of `prefill_tokens` tokens at layer 0. */
[[nodiscard]] auto prefill_params(DeltanetBuffers const& buffers) -> deltanet::PrefillParams {
	auto params = deltanet::PrefillParams{
	    .images = field_at<decltype(deltanet::PrefillParams::images)>(buffers.images),
	    .weights = buffers.prompt_weights,
	    .rows = buffers.prompt_rows,
	    .layer = 0,
	    .live = 1,
	    .slots = {},
	};
	params.slots[0] = {.slot = 0, .tokens = prefill_tokens, .first_row = 0};
	return params;
}

/** Every slot live with `decode_rows_per_slot` rows at layer 0. */
[[nodiscard]] auto decode_params(DeltanetBuffers const& buffers) -> deltanet::DecodeParams {
	auto params = deltanet::DecodeParams{
	    .images = field_at<decltype(deltanet::DecodeParams::images)>(buffers.images),
	    .rings = field_at<decltype(deltanet::DecodeParams::rings)>(buffers.rings),
	    .weights = buffers.cycle_weights,
	    .rows = buffers.cycle_rows,
	    .layer = 0,
	    .live = static_cast<unsigned>(model::slots),
	    .slots = {},
	};
	for (auto slot = 0U; slot < static_cast<unsigned>(model::slots); ++slot) {
		params.slots[slot] = {.slot = slot, .replays = 0, .rows = decode_rows_per_slot, .first_row = slot * decode_rows_per_slot};
	}
	return params;
}

/** The probes, the one-slot prefill and the 8-slot decode core on one card. */
[[nodiscard]] auto deltanet_on(CardView view) -> Result<DeltanetReading> {
	auto const program = lift(view.context.load<deltanet::DecodeCore, deltanet::Prefill, deltanet::PrefillWindow, ProbeEight, ProbeWarp>());
	auto const keys = scratch<float, probe_floats>(view.context);
	auto const queries = scratch<float, probe_floats>(view.context);
	auto const values = scratch<float, probe_floats>(view.context);
	auto const gates = scratch<deltanet::Gates, static_cast<std::size_t>(deltanet::probe_heads) * probe_tokens>(view.context);
	auto const state = scratch<float, static_cast<std::size_t>(deltanet::probe_heads) * deltanet::head_floats>(view.context);
	auto const outputs = scratch<float, probe_floats>(view.context);
	auto counts = Readback<unsigned long long, probe_cycle_words>::create(view.context);
	auto const images = scratch<unsigned char, static_cast<std::size_t>(model::slots) * deltanet::slot_image_bytes>(view.context);
	auto const rings = scratch<unsigned char, static_cast<std::size_t>(model::slots) * deltanet::ring_bytes_per_slot>(view.context);
	auto prompt = DeltanetRows<prefill_tokens>::create(view.context);
	auto cycle = DeltanetRows<static_cast<std::size_t>(model::slots) * decode_rows_per_slot>::create(view.context);
	if (auto const failed = first_failure(program, keys, queries, values, gates, state, outputs, counts, images, rings, prompt, cycle)) {
		return std::unexpected(*failed);
	}
	auto const buffers = DeltanetBuffers{
	    .keys = keys->address(),
	    .queries = queries->address(),
	    .values = values->address(),
	    .gates = gates->address(),
	    .state = state->address(),
	    .outputs = outputs->address(),
	    .cycles = counts->device().address(),
	    .images = images->address(),
	    .rings = rings->address(),
	    .prompt_weights = prompt->weights(),
	    .prompt_rows = prompt->rows_view(),
	    .cycle_weights = cycle->weights(),
	    .cycle_rows = cycle->rows_view(),
	};
	auto const probe = probe_params(buffers);
	auto const prefill = prefill_params(buffers);
	auto const decode = decode_params(buffers);
	auto reading = DeltanetReading{.card = view.card, .eight_lane_cycles_per_token = 0, .warp_cycles_per_token = 0, .eight_lane_milliseconds = 0,
	                               .prefill_milliseconds = 0, .decode_core_milliseconds = 0};
	auto const timed = [&]<class K>(K::Params const& params) -> Result<double> {
		return best_time<K>(view, *program, nothing, params);
	};
	auto const probed = [&]<class K> -> Result<double> {
		return timed.template operator()<K>(probe)
		    .and_then([&](double) {
			    return counts->fetch(view);
		    })
		    .and_then([&](std::span<unsigned long long const, probe_cycle_words> read) {
			    return cycles_per_token(read, K::grid.x, K::entry);
		    });
	};
	auto const ran = probed.template operator()<ProbeEight>()
	                     .and_then([&](double eight) {
		                     reading.eight_lane_cycles_per_token = eight;
		                     return timed.template operator()<ProbeEight>(probe);
	                     })
	                     .and_then([&](double milliseconds) {
		                     reading.eight_lane_milliseconds = milliseconds;
		                     return probed.template operator()<ProbeWarp>();
	                     })
	                     .and_then([&](double warp) {
		                     reading.warp_cycles_per_token = warp;
		                     return best_time<deltanet::Prefill, deltanet::PrefillWindow>(view, *program, nothing, prefill);
	                     })
	                     .and_then([&](double milliseconds) {
		                     reading.prefill_milliseconds = milliseconds;
		                     return timed.template operator()<deltanet::DecodeCore>(decode);
	                     });
	return ran.transform([&](double milliseconds) {
		reading.decode_core_milliseconds = milliseconds;
		return reading;
	});
}

export [[nodiscard]] auto deltanet_chain(BenchView const& bench) -> Result<Outcome> {
	auto readings = std::vector<DeltanetReading>{};
	auto const ran = for_each_card([&](Card card) {
		return deltanet_on(bench.card(card)).transform([&](DeltanetReading reading) {
			readings.push_back(reading);
		});
	});
	return ran.transform([&] {
		return evaluate_deltanet_chain(readings, bench.calibration);
	});
}

/** Positions of context per slot in the attention benchmark. */
inline constexpr unsigned attention_context = 4096;
inline constexpr auto attention_blocks_per_slot = static_cast<std::size_t>(attention_context / model::kv_block_tokens);
inline constexpr auto attention_budget = model::RowBudget::FourTiles;
inline constexpr auto attention_rows = static_cast<std::size_t>(model::rows(attention_budget));
inline constexpr auto record_rows = static_cast<std::size_t>(attn_device::max_record_rows(attention_budget));

/** The live slots a cycle of `rows` rows per slot fits in the row budget. */
[[nodiscard]] constexpr auto attention_live(unsigned rows) -> unsigned {
	return std::min(static_cast<unsigned>(model::slots), static_cast<unsigned>(attention_rows) / rows);
}

/** One rows-per-slot point: the best core time with three PV mmas (the engine's) and with one. */
export struct AttentionPoint {
	Card card;
	unsigned rows;
	unsigned live;
	double exact_milliseconds;
	double one_part_milliseconds;
};

export [[nodiscard]] auto evaluate_split_k_attention(std::span<AttentionPoint const> points) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& point : points) {
		auto const kv = static_cast<double>(point.live) * attention_context * static_cast<double>(attn_device::kv_heads) * 2.0 *
		                static_cast<double>(attn_device::row_bytes);
		auto const label = std::format("{}, {} slots x {} rows at {} positions", on(point.card), point.live, point.rows, attention_context);
		outcome.readings.push_back({.label = label + ", exact P (three PV mmas)", .value = point.exact_milliseconds * 1000.0, .unit = Unit::Microseconds});
		outcome.readings.push_back({.label = label + ", one bf16 part", .value = point.one_part_milliseconds * 1000.0, .unit = Unit::Microseconds});
		outcome.readings.push_back({.label = label + ", kv stream with exact P", .value = gigabytes_per_second(kv, point.exact_milliseconds), .unit = Unit::GigabytesPerSecond});
		outcome.readings.push_back({.label = label + ", exact over one part", .value = point.exact_milliseconds / point.one_part_milliseconds, .unit = Unit::Ratio});
	}
	outcome.notes.emplace_back("a ratio near 1 at both row counts: the exact split rides under the kv stream, as the decision assumed");
	return outcome;
}

/* PIN(clang-tidy-optional-merged-record): device params are built from plain addresses where no result is held */

/** The attention benchmark's buffers by device address. */
struct AttentionBuffers {
	std::uint64_t blocks;
	std::uint64_t tables;
	std::uint64_t query;
	std::uint64_t key;
	std::uint64_t value;
	std::uint64_t gate;
	std::uint64_t query_norm;
	std::uint64_t key_norm;
	std::uint64_t rotations;
	std::uint64_t record_outputs;
	std::uint64_t running;
};

/** The decode core's parameters at layer 0: `attention_live(rows)` slots of `rows` rows, each at `attention_context`. */
template<unsigned rows>
[[nodiscard]] auto core_params(AttentionBuffers const& buffers) -> attn_device::CoreParams {
	auto params = attn_device::CoreParams{
	    .cycle = {.live = attention_live(rows), .work = {}},
	    .kv = {.blocks = buffers.blocks, .tables = buffers.tables, .layer = 0, .unused = 0},
	    .rows = {.query = buffers.query, .key = buffers.key, .value = buffers.value, .gate = buffers.gate},
	    .preparation = {.query = buffers.query_norm, .key = buffers.key_norm, .rotations = buffers.rotations},
	    .records = {.output = buffers.record_outputs, .running = buffers.running},
	};
	for (auto slot = 0U; slot < attention_live(rows); ++slot) {
		params.cycle.work[slot] = {.slot = slot, .position = attention_context - rows, .rows = rows, .first_row = slot * rows};
	}
	return params;
}

/** Every slot's block table: slot s holds blocks [s * blocks per slot, (s + 1) * blocks per slot) in order. */
[[nodiscard]] auto block_tables() -> std::vector<std::uint32_t> {
	auto table = std::vector<std::uint32_t>(static_cast<std::size_t>(model::slots) * model::kv_table_blocks);
	for (auto slot = std::size_t{0}; slot < model::slots; ++slot) {
		for (auto block = std::size_t{0}; block < attention_blocks_per_slot; ++block) {
			table.at(slot * model::kv_table_blocks + block) = static_cast<std::uint32_t>(slot * attention_blocks_per_slot + block);
		}
	}
	return table;
}

template<unsigned rows>
[[nodiscard]] auto attention_on(CardView view) -> Result<AttentionPoint> {
	using Exact = engine::attn::DecodeCore<rows>;
	using OnePart = engine::attn::U21OnePart<rows>;
	constexpr auto live = attention_live(rows);
	auto const program = lift(view.context.load<Exact, OnePart>());
	auto const blocks = scratch<unsigned char, static_cast<std::size_t>(model::slots) * attention_blocks_per_slot * model::kv_block_bytes>(view.context);
	auto tables = Readback<std::uint32_t, static_cast<std::size_t>(model::slots) * model::kv_table_blocks>::create(view.context);
	auto const query = scratch<float, attention_rows * attn_device::query_row_floats>(view.context);
	auto const key = scratch<float, attention_rows * attn_device::kv_row_floats>(view.context);
	auto const value = scratch<float, attention_rows * attn_device::kv_row_floats>(view.context);
	auto const gate = scratch<float, attention_rows * attn_device::gate_row_floats>(view.context);
	auto const query_norm = scratch<unsigned short, static_cast<std::size_t>(attn_device::head_dim)>(view.context);
	auto const key_norm = scratch<unsigned short, static_cast<std::size_t>(attn_device::head_dim)>(view.context);
	auto const rotations = scratch<unsigned char, static_cast<std::size_t>(attn_device::rope_table_bytes)>(view.context);
	auto const record_outputs = scratch<float, record_rows * attn_device::head_dim>(view.context);
	auto const running = scratch<attn_device::Running, record_rows>(view.context);
	if (auto const failed = first_failure(program, blocks, tables, query, key, value, gate, query_norm, key_norm, rotations, record_outputs, running)) {
		return std::unexpected(*failed);
	}
	auto const params = core_params<rows>({
	    .blocks = blocks->address(),
	    .tables = tables->device().address(),
	    .query = query->address(),
	    .key = key->address(),
	    .value = value->address(),
	    .gate = gate->address(),
	    .query_norm = query_norm->address(),
	    .key_norm = key_norm->address(),
	    .rotations = rotations->address(),
	    .record_outputs = record_outputs->address(),
	    .running = running->address(),
	});
	auto const table = block_tables();
	auto const time = [&]<class K> -> Result<double> {
		return best_time<K>(view, *program, nothing, params);
	};
	auto const exact = tables->put(view, std::span<std::uint32_t const>{table}).and_then([&] {
		return time.template operator()<Exact>();
	});
	auto const one_part = exact.and_then([&](double) {
		return time.template operator()<OnePart>();
	});
	return one_part.transform([&](double milliseconds) {
		return AttentionPoint{.card = view.card, .rows = rows, .live = live, .exact_milliseconds = *exact, .one_part_milliseconds = milliseconds};
	});
}

export [[nodiscard]] auto split_k_attention(BenchView const& bench) -> Result<Outcome> {
	auto points = std::vector<AttentionPoint>{};
	auto const ran = for_each_card([&](Card card) {
		auto const view = bench.card(card);
		return attention_on<3>(view)
		    .and_then([&](AttentionPoint three) {
			    points.push_back(three);
			    return attention_on<5>(view);
		    })
		    .transform([&](AttentionPoint five) {
			    points.push_back(five);
		    });
	});
	return ran.transform([&] {
		return evaluate_split_k_attention(points);
	});
}

}
