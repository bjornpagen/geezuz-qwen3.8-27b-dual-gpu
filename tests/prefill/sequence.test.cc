// SPDX-License-Identifier: AGPL-3.0-only
//
// A chunk's whole launch sequence, all 64 layers, for chunks drawn over every shape the scheduler hands out: both
// cards run the same kernels in the same order with the same exchanges; every row-parallel matmul's push is met by its
// residual at the same epoch and epochs run on without gaps; each launch computes whole row blocks covering its rows;
// the kv append precedes the attention that reads it, after the gate it applies; the window move follows the deltanet
// core it must not race; a prompt's first chunk alone clears the slot's state, first; the mtp head writes exactly the
// entries the chunk's tokens complete; and the carried row is the chunk's last.
import std;
import engine.attn;
import engine.deltanet;
import engine.model;
import engine.prefill;
import engine.prefill_device;
import engine.testing;

namespace {

namespace prefill = engine::prefill;
namespace pd = engine::prefill_device;
namespace model = engine::model;

using model::Count;

[[nodiscard]] auto view(model::Card card) -> prefill::CardView {
	auto const index = std::to_underlying(card);
	auto const memories = prefill::CardMemories{.image = 0x1000'0000ULL * (index + 1),
	                                            .pool = 0x2000'0000ULL,
	                                            .tables = 0x3000'0000ULL,
	                                            .states = 0x4000'0000ULL * (index + 1),
	                                            .rotations = 0x5000'0000ULL,
	                                            .prompt = 0x6000'0000ULL};
	return {.card = card,
	        .memories = memories,
	        .workspace = 0x7000'0000ULL * (index + 1),
	        .peer_workspace = 0x7000'0000ULL * (2 - index),
	        .embedding = 0x8000'0000ULL,
	        .tokens = 0x9000'0000ULL};
}

/** What a launch says about the sequence's laws, kernel-independently. */
struct Facts {
	std::size_t image;
	std::size_t kernel;
	/** The exchange it pushes or meets; zero for neither. */
	std::uint64_t push;
	std::uint64_t meet;
	/** Rows it reads and rows it computes, where it names them. */
	std::optional<std::pair<Count, Count>> rows;
};

/** Reads one launch's facts: its exchange and its rows, where it has them. */
struct Reader {
	Facts& facts;

	template<class K>
	auto operator()(prefill::Launch<K> const& launch) const -> void {
		if constexpr (requires {
			              launch.params.epoch;
			              launch.params.peer_workspace;
		              }) {
			facts.push = launch.params.epoch;
		} else if constexpr (requires { launch.params.epoch; }) {
			facts.meet = launch.params.epoch;
		}
		if constexpr (
		    requires { launch.params.rows.computed; } &&
		    (requires { launch.params.weight; } || requires { launch.params.epoch; } || requires { launch.params.hidden_norm; })) {
			facts.rows = std::pair{Count{launch.params.rows.count}, Count{launch.params.rows.computed}};
		}
	}
};

[[nodiscard]] auto facts_of(prefill::Step const& step) -> Facts {
	auto facts = Facts{.image = step.index(), .kernel = 0, .push = 0, .meet = 0, .rows = std::nullopt};
	std::visit(
	    [&](auto const& inner) {
		    facts.kernel = inner.index();
		    std::visit(Reader{.facts = facts}, inner);
	    },
	    step);
	return facts;
}

[[nodiscard]] auto row_blocks_of(prefill::Step const& step) -> std::optional<Count> {
	auto blocks = std::optional<Count>{};
	std::visit(
	    [&](auto const& inner) {
		    std::visit(
		        [&]<class L>(L const& launch) {
			        if constexpr (requires { launch.params.row_blocks; }) {
				        blocks = launch.params.row_blocks;
			        }
		        },
		        inner);
	    },
	    step);
	return blocks;
}

[[nodiscard]] auto same_prompt(engine::device::attn::PromptParams const& left, engine::device::attn::PromptParams const& right) -> bool {
	return left.slot == right.slot && left.start == right.start && left.count == right.count && left.kv.layer == right.kv.layer &&
	       left.kv.blocks == right.kv.blocks && left.kv.tables == right.kv.tables && left.rows.query == right.rows.query &&
	       left.rows.key == right.rows.key && left.rows.value == right.rows.value && left.rows.gate == right.rows.gate &&
	       left.preparation.query == right.preparation.query && left.preparation.key == right.preparation.key &&
	       left.preparation.rotations == right.preparation.rotations && left.output == right.output;
}

[[nodiscard]] auto same_chunk(engine::deltanet::PrefillParams const& left, engine::deltanet::PrefillParams const& right) -> bool {
	return left.images.value == right.images.value && left.layer == right.layer && left.live == right.live &&
	       left.slots[0].slot == right.slots[0].slot && left.slots[0].tokens == right.slots[0].tokens &&
	       left.slots[0].first_row == right.slots[0].first_row && left.rows.outputs.value == right.rows.outputs.value;
}

/** The laws of both cards' sequences of one chunk, started at `epoch`. */
[[nodiscard]] auto lawful(prefill::ChunkFacts const& chunk, std::uint64_t epoch) -> bool {
	auto const zero = prefill::sequence(chunk, view(model::Card::Zero), epoch);
	auto const one = prefill::sequence(chunk, view(model::Card::One), epoch);
	auto const entries = prefill::mtp_entries(chunk);
	auto const exchanges = 2 * model::trunk_layers + (entries.count > 0 ? 3 : 0);
	auto holds = zero.steps.size() == one.steps.size() && zero.next_epoch == epoch + exchanges && one.next_epoch == zero.next_epoch;
	for (auto const* cards : {&zero, &one}) {
		auto const& steps = cards->steps;
		auto expected = epoch;
		auto rows = std::pair<Count, Count>{};
		for (auto index = std::size_t{0}; index < steps.size(); ++index) {
			auto const facts = facts_of(steps[index]);
			if (facts.rows) {
				rows = *facts.rows;
				holds =
				    holds && rows.second % pd::block_rows == 0 && rows.second >= rows.first && rows.second < rows.first + pd::block_rows;
			}
			if (auto const blocks = row_blocks_of(steps[index])) {
				// A gemm computes the rows of the norm (or residual) before it, in whole blocks.
				holds = holds && *blocks * pd::block_rows == rows.second;
			}
			if (facts.push != 0) {
				auto const next = index + 1 < steps.size() ? facts_of(steps[index + 1]) : Facts{};
				holds = holds && facts.push == expected && next.meet == expected;
				++expected;
			}
		}
		holds = holds && expected == epoch + exchanges;
	}
	for (auto index = std::size_t{0}; index < zero.steps.size() && holds; ++index) {
		auto const left = facts_of(zero.steps[index]);
		auto const right = facts_of(one.steps[index]);
		holds = left.image == right.image && left.kernel == right.kernel && left.push == right.push && left.meet == right.meet;
	}
	for (auto const card : {model::Card::Zero, model::Card::One}) {
		auto const& steps = (card == model::Card::Zero ? zero : one).steps;
		auto const* first = std::get_if<prefill::PrefillLaunch>(&steps.front());
		auto const* cleared = first == nullptr ? nullptr : std::get_if<prefill::Launch<pd::Zero>>(first);
		auto const zeros = std::ranges::count_if(steps, [](prefill::Step const& step) {
			auto const* launch = std::get_if<prefill::PrefillLaunch>(&step);
			return launch != nullptr && std::holds_alternative<prefill::Launch<pd::Zero>>(*launch);
		});
		if (chunk.from == 0) {
			holds = holds && zeros == 1 && cleared != nullptr &&
			        cleared->params.target == view(card).memories.states + Count{chunk.slot} * model::slot_state_bytes &&
			        cleared->params.chunks * 16 == model::slot_state_bytes;
		} else {
			holds = holds && zeros == 0;
		}
		auto const* last = std::get_if<prefill::PrefillLaunch>(&steps.back());
		auto const* keep = last == nullptr ? nullptr : std::get_if<prefill::Launch<pd::Keep>>(last);
		holds = holds && keep != nullptr && keep->params.row == chunk.count - 1 && keep->params.slot == chunk.slot;
		auto appends = Count{0};
		auto mtp_rows = std::optional<std::pair<Count, Count>>{};
		for (auto index = std::size_t{0}; index < steps.size(); ++index) {
			if (auto const* attention = std::get_if<prefill::AttentionLaunch>(&steps[index])) {
				if (auto const* append = std::get_if<prefill::Launch<engine::attn::Append>>(attention)) {
					auto const* next = std::get_if<prefill::AttentionLaunch>(&steps.at(index + 1));
					auto const* reader = next == nullptr ? nullptr : std::get_if<prefill::Launch<engine::attn::Prefill>>(next);
					auto const* gate = std::get_if<prefill::PrefillLaunch>(&steps.at(index - 1));
					auto const gated = gate != nullptr && (std::holds_alternative<prefill::Launch<pd::AttentionGate>>(*gate) ||
					                                       std::holds_alternative<prefill::Launch<pd::MtpAttentionGate>>(*gate));
					holds = holds && reader != nullptr && gated && same_prompt(reader->params, append->params);
					++appends;
					if (append->params.kv.layer == prefill::mtp_kv_layer) {
						mtp_rows = std::pair{Count{append->params.start}, Count{append->params.count}};
					} else {
						holds = holds && append->params.start == chunk.from && append->params.count == chunk.count;
					}
				}
			}
			if (auto const* core = std::get_if<prefill::DeltanetLaunch>(&steps[index])) {
				if (auto const* run = std::get_if<prefill::Launch<engine::deltanet::Prefill>>(core)) {
					auto const* next = std::get_if<prefill::DeltanetLaunch>(&steps.at(index + 1));
					auto const* window = next == nullptr ? nullptr : std::get_if<prefill::Launch<engine::deltanet::PrefillWindow>>(next);
					holds = holds && window != nullptr && same_chunk(window->params, run->params) &&
					        run->params.slots[0].tokens == chunk.count && run->params.slots[0].slot == chunk.slot;
				}
			}
		}
		auto const attention_layers = model::trunk_layers_of(model::Mixer::FullAttention);
		holds = holds && appends == attention_layers + (entries.count > 0 ? 1 : 0);
		holds = holds && (entries.count > 0 ? mtp_rows == std::pair{Count{entries.first}, Count{entries.count}} : !mtp_rows.has_value());
	}
	return holds;
}

[[nodiscard]] auto drawn_chunks_are_lawful(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto epoch = std::uint64_t{1};
	for (auto round = 0; round < 24; ++round) {
		auto const count = std::uniform_int_distribution<std::uint32_t>{1, static_cast<std::uint32_t>(pd::max_rows)}(generator);
		auto const from = std::uniform_int_distribution<std::uint32_t>{0, 3}(generator) == 0
		                      ? 0U
		                      : std::uniform_int_distribution<std::uint32_t>{1, 100'000}(generator);
		auto const carry =
		    from > 0 && std::uniform_int_distribution<int>{0, 1}(generator) == 1 ? prefill::Carry::Present : prefill::Carry::Absent;
		auto const slot = std::uniform_int_distribution<std::uint32_t>{0, static_cast<std::uint32_t>(model::slots - 1)}(generator);
		auto const chunk = prefill::ChunkFacts{.slot = slot, .from = from, .count = count, .carry = carry};
		if (!lawful(chunk, epoch)) {
			return false;
		}
		epoch = prefill::sequence(chunk, view(model::Card::Zero), epoch).next_epoch;
	}
	return true;
}

/** The mtp entries a chunk completes: up to its last position's, from the carried row's or the chunk's first. */
[[nodiscard]] auto entries_are_the_completed_ones() -> bool {
	using prefill::Carry;
	auto const entries = [](std::uint32_t from, std::uint32_t count, Carry carry) {
		return prefill::mtp_entries({.slot = 0, .from = from, .count = count, .carry = carry});
	};
	return entries(0, 1, Carry::Absent) == prefill::MtpEntries{.first = 0, .count = 0} &&
	       entries(0, 5, Carry::Absent) == prefill::MtpEntries{.first = 0, .count = 4} &&
	       entries(5, 1, Carry::Present) == prefill::MtpEntries{.first = 4, .count = 1} &&
	       entries(5, 3, Carry::Present) == prefill::MtpEntries{.first = 4, .count = 3} &&
	       entries(5, 3, Carry::Absent) == prefill::MtpEntries{.first = 5, .count = 2} &&
	       entries(5, 1, Carry::Absent) == prefill::MtpEntries{.first = 5, .count = 0};
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "chunks drawn over every shape keep the sequence's laws", .holds = drawn_chunks_are_lawful(31)},
	    engine::testing::Case{.name = "a prompt's first chunk of one token",
		                      .holds = lawful({.slot = 0, .from = 0, .count = 1, .carry = prefill::Carry::Absent}, 1)},
	    engine::testing::Case{
	        .name = "a whole chunk of the largest size",
	        .holds = lawful({.slot = 7, .from = 0, .count = static_cast<std::uint32_t>(pd::max_rows), .carry = prefill::Carry::Absent}, 9)},
	    engine::testing::Case{.name = "a resumed chunk crossing a row block",
		                      .holds = lawful({.slot = 3, .from = 2048, .count = 33, .carry = prefill::Carry::Present}, 300)},
	    engine::testing::Case{.name = "the mtp entries are the ones the chunk completes", .holds = entries_are_the_completed_ones()},
	};
	return engine::testing::report(cases);
}
