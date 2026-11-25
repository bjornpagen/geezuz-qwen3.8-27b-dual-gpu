// SPDX-License-Identifier: AGPL-3.0-only
//
// The ranking's keys against the one order on candidates: for every pair, the larger key is the candidate
// `ranks_before` puts first, so sorting keys is ranking candidates; a key turns back into its candidate; and no
// candidate's key is the empty key. Logits are drawn to hit the order's edges: equal logits with different ids, both
// zeros, infinities, subnormals, and the extremes of the finite range.
import std;
import engine.decode_device;
import engine.model;
import engine.sampling;
import engine.testing;

namespace {

namespace dd = engine::decode_device;
namespace sampling = engine::sampling;

using sampling::Candidate;

[[nodiscard]] auto logit(std::mt19937_64& random) -> float {
	constexpr auto edges = std::array{0.0F,
	                                  -0.0F,
	                                  std::numeric_limits<float>::infinity(),
	                                  -std::numeric_limits<float>::infinity(),
	                                  std::numeric_limits<float>::denorm_min(),
	                                  -std::numeric_limits<float>::denorm_min(),
	                                  std::numeric_limits<float>::max(),
	                                  std::numeric_limits<float>::lowest(),
	                                  1.0F,
	                                  -1.0F};
	switch (random() % 4) {
	case 0:
		return std::span{edges}[random() % edges.size()];
	case 1:
		// A narrow range, so equal logits come up often.
		return static_cast<float>(static_cast<int>(random() % 7) - 3);
	default: {
		// Any finite pattern.
		auto bits = static_cast<std::uint32_t>(random());
		while ((bits & 0x7f80'0000U) == 0x7f80'0000U) {
			bits = static_cast<std::uint32_t>(random());
		}
		return std::bit_cast<float>(bits);
	}
	}
}

[[nodiscard]] auto candidate(std::mt19937_64& random) -> Candidate {
	auto const id = static_cast<int>(random() % (random() % 2 == 0 ? 8U : engine::model::sampled_vocabulary));
	return {.logit = logit(random), .id = id};
}

[[nodiscard]] auto keys_rank_as_candidates(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 1'000'000; ++trial) {
		auto const a = candidate(random);
		auto const b = candidate(random);
		auto const ka = dd::key_of(a);
		auto const kb = dd::key_of(b);
		if ((ka > kb) != sampling::ranks_before(a, b) || (kb > ka) != sampling::ranks_before(b, a)) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto keys_round_trip(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 1'000'000; ++trial) {
		auto const c = candidate(random);
		auto const back = dd::candidate_of(dd::key_of(c));
		// -0 comes back as +0: an equal logit, which every use of a candidate reads alike.
		auto const logit = c.logit == 0.0F ? 0.0F : c.logit;
		if (back.id != c.id || std::bit_cast<std::uint32_t>(back.logit) != std::bit_cast<std::uint32_t>(logit) || dd::key_of(c) == dd::no_candidate) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto no_candidate_ranks_last() -> bool {
	auto const none = dd::candidate_of(dd::no_candidate);
	auto const lowest = Candidate{.logit = -std::numeric_limits<float>::infinity(), .id = engine::model::sampled_vocabulary - 1};
	return none.id < 0 && dd::key_of(none) == dd::no_candidate && dd::key_of(lowest) > dd::no_candidate;
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const cases = std::array{
	    Case{.name = "keys order as ranks_before (seed 1)", .holds = keys_rank_as_candidates(1)},
	    Case{.name = "keys order as ranks_before (seed 2)", .holds = keys_rank_as_candidates(2)},
	    Case{.name = "a key turns back into its candidate (seed 3)", .holds = keys_round_trip(3)},
	    Case{.name = "the empty key is no candidate and ranks after every real one", .holds = no_candidate_ranks_last()},
	};
	return engine::testing::report(cases);
}
