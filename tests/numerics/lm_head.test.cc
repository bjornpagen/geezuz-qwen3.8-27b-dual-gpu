// SPDX-License-Identifier: AGPL-3.0-only
//
// The lm_head against the cpu reference, bit for bit: bf16 tiles over the whole hidden size with the head phase's own
// run, on both cards (each card's first rows of its permuted vocabulary), through the emulation of every warp of the
// decode gemv; then the top-k candidates the two cards' halves merge to, from the emulation's logits and from the
// reference's. It reads the whole hidden size per row, so it has its own deadline.
import std;
import engine.gemv;
import engine.model;
import engine.numerics_problems;
import engine.sampling;
import engine.sampling_device;
import engine.schedule;
import engine.testing;

namespace {

namespace gemv = engine::gemv;
namespace model = engine::model;
namespace sampling = engine::sampling;

using engine::numerics_problems::make_problem;
using engine::numerics_problems::reference_rows;
using engine::numerics_problems::same_bits;
using gemv::Matrix;
using gemv::Problem;

/** The first output tiles of the lm_head, with the head phase's own run. */
inline constexpr Matrix head_phase = gemv::matrix(gemv::phase_of(engine::schedule::Op::Head), 0);
inline constexpr Matrix head_front = {
    .units = head_phase.units, .first = 0, .output_tiles = 4, .k_tiles = head_phase.k_tiles, .tensors = 1, .format = model::Format::Bf16};

static_assert(head_phase.format == model::Format::Bf16 && gemv::k_extent(head_phase) == model::hidden);

/**
 * The lm_head's logits on both cards (each card's first rows of its permuted vocabulary) and the top-k candidates the
 * two cards' halves merge to: the emulation's against the reference's, logit bits and ids alike.
 */
[[nodiscard]] auto head_candidates_are_the_reference(std::uint64_t seed) -> bool {
	constexpr auto rows = 8;
	auto const width = static_cast<std::size_t>(head_front.output_tiles * gemv::tile_rows);
	auto emulated = std::array<std::vector<float>, model::cards>{};
	auto referenced = std::array<std::vector<float>, model::cards>{};
	auto problems = std::vector<Problem>{};
	for (auto card = std::size_t{0}; card < model::cards; ++card) {
		auto problem = make_problem(head_front, rows, seed + card);
		if (card == 1) {
			problem.activations = problems.front().activations;
		}
		emulated.at(card) = gemv::rows_of(problem, gemv::decode<1>(problem));
		referenced.at(card) = reference_rows(problem);
		problems.push_back(std::move(problem));
	}
	auto holds = same_bits(emulated[0], referenced[0]) && same_bits(emulated[1], referenced[1]);
	for (auto row = std::size_t{0}; row < rows; ++row) {
		// Logits by token id: card c's row r is the token `token(c, r)`, here numbered densely in card-major order.
		auto const logits = [&](std::array<std::vector<float>, model::cards> const& cards) {
			auto out = std::vector<float>{};
			for (auto const& card : cards) {
				auto const row_logits = std::span{card}.subspan(row * width, width);
				out.insert(out.end(), row_logits.begin(), row_logits.end());
			}
			return out;
		};
		auto const from_emulation = sampling::candidates(logits(emulated));
		auto const from_reference = sampling::candidates(logits(referenced));
		holds = holds && std::ranges::equal(from_emulation, from_reference, [](auto const& a, auto const& b) {
			        return a.id == b.id && std::bit_cast<std::uint32_t>(a.logit) == std::bit_cast<std::uint32_t>(b.logit);
		        });
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "lm_head logits over the real hidden and their top-k candidates equal the reference (seed 7)",
		                      .holds = head_candidates_are_the_reference(7)},
	};
	return engine::testing::report(cases);
}
