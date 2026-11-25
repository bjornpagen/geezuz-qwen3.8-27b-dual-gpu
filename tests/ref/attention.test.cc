// SPDX-License-Identifier: AGPL-3.0-only
//
// Gated attention on a small head shape against float64 softmax attention over the same bf16 keys, values and query,
// over contexts of many chunks; rope against float64 rotation; and a row's output unchanged by kv appended after it.
import std;
import engine.model;
import engine.ref;
import engine.testing;

namespace {

namespace ref = engine::ref;
namespace model = engine::model;

using model::Count;

struct Small {
	static constexpr Count query_heads = 4;
	static constexpr Count kv_heads = 2;
	static constexpr Count head_dim = 64;
	static constexpr Count rotary_dim = 16;
};

constexpr auto dim = Small::head_dim;

[[nodiscard]] auto bf16_draws(std::mt19937_64& generator, Count count, float scale) -> std::vector<ref::Bf16> {
	auto value = std::uniform_real_distribution<float>{-scale, scale};
	auto out = std::vector<ref::Bf16>(count);
	for (auto& each : out) {
		each = ref::round(value(generator));
	}
	return out;
}

[[nodiscard]] auto cache_of(std::mt19937_64& generator, Count positions) -> ref::KvCache<Small> {
	return {.keys = bf16_draws(generator, positions * Small::kv_heads * dim, 2.0F), .values = bf16_draws(generator, positions * Small::kv_heads * dim, 2.0F)};
}

/** Softmax attention for `head` at `position` in float64, gated and not rounded. */
[[nodiscard]] auto float64_attention(ref::KvCache<Small> const& cache, std::span<ref::Bf16 const> query, Count head, Count position,
                                     std::span<float const> gate) -> std::vector<double> {
	auto const kv_head = head / (Small::query_heads / Small::kv_heads);
	auto scores = std::vector<double>{};
	for (auto at = Count{0}; at <= position; ++at) {
		auto const key = cache.key(at, kv_head);
		auto score = 0.0;
		for (auto feature = Count{0}; feature < dim; ++feature) {
			score += static_cast<double>(ref::widen(key[feature])) * static_cast<double>(ref::widen(query[feature]));
		}
		scores.push_back(score / 16.0);
	}
	auto const most = std::ranges::max(scores);
	auto total = 0.0;
	for (auto& score : scores) {
		score = std::exp(score - most);
		total += score;
	}
	auto out = std::vector<double>(dim);
	for (auto feature = Count{0}; feature < dim; ++feature) {
		auto sum = 0.0;
		for (auto at = Count{0}; at <= position; ++at) {
			sum += scores[at] * static_cast<double>(ref::widen(cache.value(at, kv_head, feature)));
		}
		out[feature] = (sum / total) / (1.0 + std::exp(-static_cast<double>(gate[feature])));
	}
	return out;
}

template<ref::Accumulation A>
[[nodiscard]] auto attention_matches_float64(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto const cache = cache_of(generator, 150);
	auto gate_value = std::uniform_real_distribution<float>{-3.0F, 3.0F};
	for (auto const position : {Count{0}, Count{1}, Count{31}, Count{32}, Count{63}, Count{100}, Count{149}}) {
		for (auto head = Count{0}; head < Small::query_heads; ++head) {
			auto const query = bf16_draws(generator, dim, 2.0F);
			auto gate = std::vector<float>(dim);
			for (auto& each : gate) {
				each = gate_value(generator);
			}
			auto const got = ref::attend<A, Small>(cache, std::span<ref::Bf16 const, dim>{query}, head, position, std::span<float const, dim>{gate});
			auto const want = float64_attention(cache, query, head, position, gate);
			for (auto const [g, w] : std::views::zip(got, want)) {
				if (std::abs(static_cast<double>(ref::widen(g)) - w) > (0x1p-8 * std::abs(w)) + 1e-6) {
					return false;
				}
			}
		}
	}
	return true;
}

/** A row sees only positions up to its own: kv appended after it changes none of its bits. */
[[nodiscard]] auto later_kv_changes_nothing(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto const longer = cache_of(generator, 96);
	auto shorter = longer;
	shorter.keys.resize(41 * Small::kv_heads * dim);
	shorter.values.resize(41 * Small::kv_heads * dim);
	auto const query = bf16_draws(generator, dim, 2.0F);
	auto const gate = std::vector<float>(dim, 0.5F);
	for (auto head = Count{0}; head < Small::query_heads; ++head) {
		auto const a = ref::attend<ref::Ieee, Small>(shorter, std::span<ref::Bf16 const, dim>{query}, head, 40, std::span<float const, dim>{gate});
		auto const b = ref::attend<ref::Ieee, Small>(longer, std::span<ref::Bf16 const, dim>{query}, head, 40, std::span<float const, dim>{gate});
		if (a != b) {
			return false;
		}
	}
	return true;
}

/** Rope against float64 cos and sin of the same angle, with the head norm's output fed in directly. */
[[nodiscard]] auto rope_matches_float64(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto value = std::uniform_real_distribution<float>{-4.0F, 4.0F};
	auto place = std::uniform_int_distribution<Count>{0, model::max_positions - 1};
	for (auto trial = 0; trial < 300; ++trial) {
		auto head = std::array<float, dim>{};
		for (auto& each : head) {
			each = value(generator);
		}
		auto const position = place(generator);
		auto rotated = head;
		ref::rotate<Small>(rotated, position);
		for (auto pair = Count{0}; pair < Small::rotary_dim / 2; ++pair) {
			auto const angle = static_cast<double>(model::rope::angle(position, pair));
			auto const low = static_cast<double>(head.at(pair));
			auto const high = static_cast<double>(head.at(pair + (Small::rotary_dim / 2)));
			auto const want_low = (low * std::cos(angle)) - (high * std::sin(angle));
			auto const want_high = (high * std::cos(angle)) + (low * std::sin(angle));
			if (std::abs(static_cast<double>(rotated.at(pair)) - want_low) > 1e-5 || std::abs(static_cast<double>(rotated.at(pair + (Small::rotary_dim / 2))) - want_high) > 1e-5) {
				return false;
			}
		}
		for (auto feature = Small::rotary_dim; feature < dim; ++feature) {
			if (rotated.at(feature) != head.at(feature)) {
				return false;
			}
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "chunked attention with the exact weight split, ieee and sm_86, is float64 softmax within bf16 rounding (seed 21)",
	        .holds = attention_matches_float64<ref::Ieee>(21) && attention_matches_float64<ref::Tensor>(21),
	    },
	    engine::testing::Case{.name = "kv appended after a row changes none of its output bits (seed 22)", .holds = later_kv_changes_nothing(22)},
	    engine::testing::Case{.name = "rope turns the rotary pairs as float64 does and passes the rest through (seed 23)", .holds = rope_matches_float64(23)},
	};
	return engine::testing::report(cases);
}
