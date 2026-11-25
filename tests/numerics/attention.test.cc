// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention's kernels against the cpu reference, bit for bit, at the real head shape: the host emulation of the
// kernels (the warp's lanes for the norms, the rope table's rotations, every score and PV mma through the model of the
// sm_86 tensor core, the online softmax over each group's chunks, the merge of the groups) must equal the reference's
// own route for the same projections: its keys, values and queries after norm and rope, and every row's gated output.
// Decode is a verify cycle's rows (one to five, each seeing the rows before it); prefill a prompt chunk's rows, its
// keys appended first. Positions straddle the 256-position groups near the start and the change to 2048-position
// groups at position 32768.
import std;
import engine.attn;
import engine.model;
import engine.prims;
import engine.ref;
import engine.testing;

namespace {

namespace attn = engine::device::attn;
namespace model = engine::model;
namespace prims = engine::device::prims;
namespace ref = engine::ref;

using model::Count;

/** One card's attention: its query heads reading its kv heads. */
struct Card {
	static constexpr Count query_heads = attn::query_heads;
	static constexpr Count kv_heads = attn::kv_heads;
	static constexpr Count head_dim = model::head_dim;
	static constexpr Count rotary_dim = model::rotary_dim;
};

constexpr auto dim = model::head_dim;

using Projection = std::array<float, dim>;

[[nodiscard]] auto projection(std::mt19937_64& random, float scale) -> Projection {
	auto out = Projection{};
	std::ranges::generate(out, [&] {
		return std::normal_distribution<float>{0.0F, scale}(random);
	});
	return out;
}

/** The fp32 query and gate projections of a slot's rows from position `first`, and the layer's q_norm and k_norm weights. */
struct Slot {
	Count first;
	std::vector<std::array<Projection, attn::query_heads>> queries;
	std::vector<std::array<Projection, attn::query_heads>> gates;
	std::array<prims::Bf16, dim> query_norm;
	std::array<prims::Bf16, dim> key_norm;
};

/** The slot's kv as the emulation keeps it (one view per kv head, through the end of the last chunk) and as the reference does. */
struct Caches {
	std::array<std::vector<engine::attn::Row>, attn::kv_heads> keys;
	std::array<std::vector<engine::attn::Row>, attn::kv_heads> values;
	ref::KvCache<Card> reference;
	bool same;
};

struct Drawn {
	Slot slot;
	Caches caches;
};

/**
 * Projections drawn so the scores spread over several units (the softmax neither saturates nor flattens), and every
 * position's key and value appended by the emulation's preparation and by the reference's, compared bit for bit. The
 * key and value projections are drawn as they are appended, so a long context costs no fp32 rows.
 */
[[nodiscard]] auto draw(std::uint64_t seed, Count first, Count positions) -> Drawn {
	auto random = engine::testing::generator(seed);
	auto drawn = Drawn{.slot = {.first = first, .queries = {}, .gates = {}, .query_norm = {}, .key_norm = {}},
	                   .caches = {.keys = {}, .values = {}, .reference = {}, .same = true}};
	auto& slot = drawn.slot;
	auto& caches = drawn.caches;
	for (auto& weight : slot.query_norm) {
		weight = prims::round_to_bf16(std::normal_distribution<float>{0.0F, 0.3F}(random));
	}
	for (auto& weight : slot.key_norm) {
		weight = prims::round_to_bf16(std::normal_distribution<float>{0.0F, 0.3F}(random));
	}
	for (auto position = Count{0}; position < positions; ++position) {
		for (auto head = Count{0}; head < attn::kv_heads; ++head) {
			auto const raw = projection(random, 4.0F);
			auto const key = engine::attn::prepare_rotated(raw, slot.key_norm, static_cast<std::uint32_t>(position));
			auto const reference = ref::positioned<Card>(raw, slot.key_norm, position);
			caches.same = caches.same && std::ranges::equal(key, reference, [](prims::Bf16 a, prims::Bf16 b) {
				              return a.bits == b.bits;
			              });
			caches.keys.at(head).push_back(key);
			caches.reference.keys.insert(caches.reference.keys.end(), key.begin(), key.end());
		}
		for (auto head = Count{0}; head < attn::kv_heads; ++head) {
			auto const value = engine::attn::prepare_value(projection(random, 1.0F));
			caches.values.at(head).push_back(value);
			caches.reference.values.insert(caches.reference.values.end(), value.begin(), value.end());
		}
	}
	// A chunk is read whole; the positions past the last are zeros, which no row sees.
	auto const padded = model::divide_up(positions, attn::chunk_positions) * attn::chunk_positions;
	for (auto head = Count{0}; head < attn::kv_heads; ++head) {
		caches.keys.at(head).resize(padded);
		caches.values.at(head).resize(padded);
	}
	for (auto position = first; position < positions; ++position) {
		auto queries = std::array<Projection, attn::query_heads>{};
		auto gates = std::array<Projection, attn::query_heads>{};
		for (auto head = Count{0}; head < attn::query_heads; ++head) {
			queries.at(head) = projection(random, 4.0F);
			gates.at(head) = projection(random, 2.0F);
		}
		slot.queries.push_back(queries);
		slot.gates.push_back(gates);
	}
	return drawn;
}

/** Row `position`'s output for query head `head`, by the emulation and by the reference: whether they agree bit for bit. */
[[nodiscard]] auto row_agrees(Slot const& slot, Caches const& caches, Count position, Count head) -> bool {
	auto const kv_head = head / attn::group_heads;
	auto const query = engine::attn::prepare_rotated(slot.queries.at(position - slot.first).at(head), slot.query_norm,
	                                                 static_cast<std::uint32_t>(position));
	auto const reference_query = ref::positioned<Card>(slot.queries.at(position - slot.first).at(head), slot.query_norm, position);
	auto const kv = engine::attn::KvView{.keys = caches.keys.at(kv_head), .values = caches.values.at(kv_head)};
	auto const emulated = engine::attn::finish(engine::attn::attend(query, kv, position, attn::PvParts::Three),
	                                           slot.gates.at(position - slot.first).at(head));
	auto const referenced =
	    ref::attend<ref::Tensor, Card>(caches.reference, reference_query, head, position, slot.gates.at(position - slot.first).at(head));
	auto const same = [](prims::Bf16 a, prims::Bf16 b) {
		return a.bits == b.bits;
	};
	return std::ranges::equal(query, reference_query, same) && std::ranges::equal(emulated, referenced, same);
}

/**
 * A verify cycle at `position` with `rows` rows: every row of every query head of the card, row t seeing positions up
 * to position + t, the cycle's own kv appended first as the decode core splices it.
 */
[[nodiscard]] auto decode_agrees(std::uint64_t seed, Count position, Count rows, Count heads) -> bool {
	auto const [slot, caches] = draw(seed, position, position + rows);
	auto holds = caches.same;
	for (auto row = Count{0}; row < rows; ++row) {
		for (auto head = Count{0}; head < heads; ++head) {
			holds = holds && row_agrees(slot, caches, position + row, head * (attn::query_heads / heads));
		}
	}
	return holds;
}

/**
 * A prompt chunk of `count` rows from `start` over a cached prefix: the append writes every row's kv, then each prefill
 * job's rows (tiles of five, by kv head, as `PromptJobs` deals them) attend up to their own positions.
 */
[[nodiscard]] auto prefill_agrees(std::uint64_t seed, Count start, Count count) -> bool {
	auto const [slot, caches] = draw(seed, start, start + count);
	auto holds = caches.same;
	auto seen = std::vector<int>(count * attn::kv_heads);
	for (auto block = 0U; block < attn::core_blocks; ++block) {
		auto const jobs = attn::PromptJobs{.slot = 0,
		                                   .start = static_cast<unsigned>(start),
		                                   .rows = static_cast<unsigned>(count),
		                                   .first = block,
		                                   .blocks = attn::core_blocks};
		for (auto index = 0U; index < jobs.count(); ++index) {
			auto const job = jobs.at(index);
			for (auto token = 0U; token < job.work.rows; ++token) {
				auto const position = Count{job.work.position} + token;
				++seen.at((position - start) * attn::kv_heads + job.head);
				for (auto within = Count{0}; within < attn::group_heads; ++within) {
					holds = holds && row_agrees(slot, caches, position, job.head * attn::group_heads + within);
				}
			}
		}
	}
	return holds && std::ranges::all_of(seen, [](int times) {
		       return times == 1;
	       });
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "decode, one row at position 0, every query head (seed 1)",
		                      .holds = decode_agrees(1, 0, 1, attn::query_heads)},
	    engine::testing::Case{.name = "decode, five rows at position 37, every query head (seed 2)",
		                      .holds = decode_agrees(2, 37, 5, attn::query_heads)},
	    engine::testing::Case{.name = "decode, five rows across the first group boundary at 256 (seed 3)",
		                      .holds = decode_agrees(3, 253, 5, attn::query_heads)},
	    engine::testing::Case{.name = "decode, three rows across several groups at position 1021 (seed 4)",
		                      .holds = decode_agrees(4, 1021, 3, 4)},
	    engine::testing::Case{.name = "decode, two rows across the change to long groups at position 32768 (seed 5)",
		                      .holds = decode_agrees(5, 32767, 2, 1)},
	    engine::testing::Case{.name = "prefill, a 23-row chunk across the group boundary at 256 (seed 6)",
		                      .holds = prefill_agrees(6, 240, 23)},
	};
	return engine::testing::report(cases);
}
