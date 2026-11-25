// SPDX-License-Identifier: AGPL-3.0-only
//
// The tiling and the merge against the straightforward reference, in one rounding. The decode path is simulated as the
// kernels divide it (each SM's assignment, each group's record at its record row, the merge's fold of a row's records)
// and must equal the reference's left fold of the row's groups bit for bit at every SM count: where a group is
// computed never changes how. A row's output must not depend on what the kv holds past its own position (the later
// rows of its cycle, stale drafts or zeros), so a verify row equals the same row decoded alone; nothing is ever a nan;
// and the result is softmax attention, to within the roundings the contract names.
//
// The accumulation here is `SequentialFp32`, a test double that sums each k16 slice by fp32 fmas in k order. Every
// property above holds for any fixed accumulation; with the model of the sm_86 mma in its place the reference is the
// bit-exact expectation of the kernels.
import std;
import engine.attn;
import engine.prims;
import engine.testing;

namespace {

namespace attn = engine::device::attn;
namespace prims = engine::device::prims;

using attn::Count;
using attn::Cycle;
using engine::attn::KvView;
using engine::attn::Row;
using engine::attn::State;

/** Test double for the tensor core's sum: fp32 fmas in k order. Not a model of the mma. */
struct SequentialFp32 {
	[[nodiscard]] static auto step(float accumulator, std::span<prims::Bf16 const, 16> a, std::span<prims::Bf16 const, 16> b) -> float {
		for (auto k = std::size_t{0}; k < 16; ++k) {
			accumulator = std::fma(prims::widen(a[k]), prims::widen(b[k]), accumulator);
		}
		return accumulator;
	}
};

static_assert(engine::attn::Accumulation<SequentialFp32>);

[[nodiscard]] auto random_row(std::mt19937_64& random, float scale) -> Row {
	auto row = Row{};
	for (auto& value : row) {
		value = prims::round_to_bf16(std::normal_distribution<float>{0.0F, scale}(random));
	}
	return row;
}

/** One slot's data: its kv through the end of its last chunk, and each row's twelve queries and gates. */
struct SlotData {
	std::vector<Row> keys;
	std::vector<Row> values;
	std::vector<std::array<Row, attn::query_heads>> queries;
	std::vector<std::array<std::array<float, attn::head_dim>, attn::query_heads>> gates;
};

/** A cycle of up to four slots with contexts up to about 1500 positions (several groups), and its data per kv head. */
struct Case {
	Cycle cycle;
	std::vector<std::array<SlotData, attn::kv_heads>> slots;
};

[[nodiscard]] auto random_case(std::mt19937_64& random) -> Case {
	auto result = Case{.cycle = {.live = 0, .work = {}}, .slots = {}};
	auto first_row = 0U;
	for (auto slot = 0U; slot < 4; ++slot) {
		auto const rows = std::uniform_int_distribution<unsigned>{1, static_cast<unsigned>(attn::max_rows)}(random);
		auto const position = std::uniform_int_distribution<unsigned>{0, 1500}(random);
		result.cycle.work[result.cycle.live] = {.slot = slot, .position = position, .rows = rows, .first_row = first_row};
		++result.cycle.live;
		first_row += rows;
		auto const positions = attn::chunks(result.cycle.work[slot]) * attn::chunk_positions;
		auto data = std::array<SlotData, attn::kv_heads>{};
		for (auto& head : data) {
			// Scores spread over several units, so the softmax neither saturates nor flattens.
			auto const key_scale = std::uniform_real_distribution<float>{0.5F, 3.0F}(random);
			for (auto index = Count{0}; index < positions; ++index) {
				head.keys.push_back(random_row(random, key_scale));
				head.values.push_back(random_row(random, 1.0F));
			}
			for (auto token = 0U; token < rows; ++token) {
				auto queries = std::array<Row, attn::query_heads>{};
				auto gates = std::array<std::array<float, attn::head_dim>, attn::query_heads>{};
				for (auto query = Count{0}; query < attn::query_heads; ++query) {
					queries.at(query) = random_row(random, 1.0F);
					for (auto& gate : gates.at(query)) {
						gate = std::normal_distribution<float>{0.0F, 2.0F}(random);
					}
				}
				head.queries.push_back(queries);
				head.gates.push_back(gates);
			}
		}
		result.slots.push_back(std::move(data));
	}
	return result;
}

[[nodiscard]] auto view(SlotData const& data) -> KvView {
	return {.keys = data.keys, .values = data.values};
}

/** Query row `row` (token-major, six heads per token) of the kv head: its query and its gate. */
[[nodiscard]] auto query_of(SlotData const& data, Count head, Count row) -> Row const& {
	return data.queries.at(row / attn::group_heads).at(head * attn::group_heads + row % attn::group_heads);
}

[[nodiscard]] auto same_bits(Row const& left, Row const& right) -> bool {
	return std::ranges::equal(left, right, [](prims::Bf16 a, prims::Bf16 b) { return a.bits == b.bits; });
}

/** The reference output of every (work, token, query head), in merge-unit order. */
[[nodiscard]] auto reference_outputs(Case const& input) -> std::vector<Row> {
	auto outputs = std::vector<Row>{};
	for (auto work = Count{0}; work < input.cycle.live; ++work) {
		auto const& slot = input.cycle.work[work];
		for (auto token = Count{0}; token < slot.rows; ++token) {
			for (auto query = Count{0}; query < attn::query_heads; ++query) {
				auto const head = query / attn::group_heads;
				auto const& data = input.slots.at(work).at(head);
				auto const row = token * attn::group_heads + query % attn::group_heads;
				auto const state =
				    engine::attn::attend<SequentialFp32>(query_of(data, head, row), view(data), slot.position + token, attn::PvParts::Three);
				outputs.push_back(engine::attn::finish(state, data.gates.at(token).at(query)));
			}
		}
	}
	return outputs;
}

/**
 * The decode path as the kernels divide it: each SM's items write each group's state at its record row (a row's
 * records past its own position stay nan, as nothing may read them); the merge folds each unit's records in order.
 */
[[nodiscard]] auto decode_outputs(Case const& input, Count sms) -> std::vector<Row> {
	auto const& cycle = input.cycle;
	auto records = std::vector<State>(attn::record_rows(cycle),
	                                  State{.output = {}, .maximum = std::numeric_limits<float>::quiet_NaN(), .total = std::numeric_limits<float>::quiet_NaN()});
	for (auto& record : records) {
		record.output.fill(std::numeric_limits<float>::quiet_NaN());
	}
	for (auto sm = Count{0}; sm < sms; ++sm) {
		auto const mine = attn::assignment(cycle, sm, sms);
		for (auto const& item : std::span{mine.item}.first(mine.count)) {
			auto const& slot = cycle.work[item.work];
			auto const& data = input.slots.at(item.work).at(item.head);
			for (auto group = Count{item.first_group}; group < Count{item.first_group} + item.groups; ++group) {
				for (auto row = Count{0}; row < attn::query_rows(slot); ++row) {
					auto const limit = Count{slot.position} + row / attn::group_heads;
					if (attn::group_begin(group) * attn::chunk_positions <= limit) {
						records.at(attn::record_row(cycle, item.work, item.head, group, row)) =
						    engine::attn::group_state<SequentialFp32>(query_of(data, item.head, row), view(data), limit, group, attn::PvParts::Three);
					}
				}
			}
		}
	}
	auto outputs = std::vector<Row>{};
	for (auto unit = Count{0}; unit < attn::merge_units(cycle); ++unit) {
		auto const place = attn::merge_unit(cycle, unit);
		auto const& slot = cycle.work[place.work];
		auto const head = place.query_head / attn::group_heads;
		auto const row = Count{place.token} * attn::group_heads + place.query_head % attn::group_heads;
		auto state = records.at(attn::record_row(cycle, place.work, head, 0, row));
		for (auto group = Count{1}; group < attn::row_groups(slot, place.token); ++group) {
			state = engine::attn::fold(state, records.at(attn::record_row(cycle, place.work, head, group, row)));
		}
		outputs.push_back(engine::attn::finish(state, input.slots.at(place.work).at(head).gates.at(place.token).at(place.query_head)));
	}
	return outputs;
}

/** At every SM count the decode path equals the reference bit for bit. */
[[nodiscard]] auto decode_path_equals_reference(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 2; ++trial) {
		auto const input = random_case(random);
		auto const expected = reference_outputs(input);
		for (auto const sms : std::array<Count, 4>{1, 7, 82, 200}) {
			auto const got = decode_outputs(input, sms);
			sound = sound && std::ranges::equal(got, expected, same_bits);
		}
	}
	return sound;
}

/**
 * A row's output does not change with what the kv holds past its position: the later rows of a verify cycle, stale
 * rejected drafts, or the zeros of a fresh block. So a verify row equals the same row decoded alone.
 */
[[nodiscard]] auto rows_ignore_kv_past_their_position(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 40; ++trial) {
		auto const position = std::uniform_int_distribution<Count>{0, 1200}(random);
		auto const extent = (position / attn::chunk_positions + 2) * attn::chunk_positions;
		auto keys = std::vector<Row>{};
		auto values = std::vector<Row>{};
		for (auto index = Count{0}; index < extent; ++index) {
			keys.push_back(random_row(random, 2.0F));
			values.push_back(random_row(random, 1.0F));
		}
		auto const query = random_row(random, 1.0F);
		auto gate = std::array<float, attn::head_dim>{};
		std::ranges::generate(gate, [&] { return std::normal_distribution<float>{0.0F, 2.0F}(random); });
		auto const output = [&](std::vector<Row> const& k, std::vector<Row> const& v) {
			return engine::attn::finish(engine::attn::attend<SequentialFp32>(query, {.keys = k, .values = v}, position, attn::PvParts::Three), gate);
		};
		auto const as_verify = output(keys, values);
		auto zero_keys = keys;
		auto zero_values = values;
		auto huge_keys = keys;
		auto huge_values = values;
		for (auto index = position + 1; index < extent; ++index) {
			zero_keys.at(index).fill(prims::Bf16{.bits = 0});
			zero_values.at(index).fill(prims::Bf16{.bits = 0});
			huge_keys.at(index).fill(prims::round_to_bf16(3.0e38F));
			huge_values.at(index).fill(prims::round_to_bf16(-3.0e38F));
		}
		auto const finite = std::ranges::all_of(as_verify, [](prims::Bf16 value) { return std::isfinite(prims::widen(value)); });
		sound = sound && finite && same_bits(as_verify, output(zero_keys, zero_values)) && same_bits(as_verify, output(huge_keys, huge_values));
	}
	return sound;
}

/** The result is softmax attention: the normalized output within a few fp32 roundings of the exact value in double. */
[[nodiscard]] auto result_is_softmax_attention(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto sound = true;
	for (auto trial = 0; trial < 30; ++trial) {
		auto const position = std::uniform_int_distribution<Count>{0, 1500}(random);
		auto const extent = (position / attn::chunk_positions + 1) * attn::chunk_positions;
		auto keys = std::vector<Row>{};
		auto values = std::vector<Row>{};
		for (auto index = Count{0}; index < extent; ++index) {
			keys.push_back(random_row(random, 3.0F));
			values.push_back(random_row(random, 1.0F));
		}
		auto const query = random_row(random, 1.0F);
		auto const state = engine::attn::attend<SequentialFp32>(query, {.keys = keys, .values = values}, position, attn::PvParts::Three);
		auto scores = std::vector<double>{};
		for (auto index = Count{0}; index <= position; ++index) {
			auto dot = 0.0;
			for (auto feature = Count{0}; feature < attn::head_dim; ++feature) {
				dot += static_cast<double>(prims::widen(keys.at(index).at(feature))) * static_cast<double>(prims::widen(query.at(feature)));
			}
			scores.push_back(dot / 16.0);
		}
		auto const maximum = std::ranges::max(scores);
		auto total = 0.0;
		for (auto& score : scores) {
			score = std::exp(score - maximum);
			total += score;
		}
		for (auto feature = Count{0}; feature < attn::head_dim; ++feature) {
			auto exact = 0.0;
			for (auto index = Count{0}; index <= position; ++index) {
				exact += scores.at(index) * static_cast<double>(prims::widen(values.at(index).at(feature)));
			}
			exact /= total;
			auto const got = static_cast<double>(state.output.at(feature)) / static_cast<double>(state.total);
			sound = sound && std::isfinite(got) && std::abs(got - exact) <= 1e-4;
		}
	}
	return sound;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the decode path equals the reference bit for bit at 1, 7, 82 and 200 SMs (seed 1)",
	                          .holds = decode_path_equals_reference(1)},
	    engine::testing::Case{.name = "a row ignores the kv past its position: zeros, later rows, huge values (seed 2)",
	                          .holds = rows_ignore_kv_past_their_position(2)},
	    engine::testing::Case{.name = "the result is softmax attention to within fp32 roundings (seed 3)", .holds = result_is_softmax_attention(3)},
	};
	return engine::testing::report(cases);
}
