// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill leaves exactly the state decode expects: a prompt prefilled in chunks, through the very launch sequences
// `engine.prefill` builds, leaves both cards' kv blocks (the trunk's and the mtp head's), every deltanet state and
// convolution window, and the final norm of its last row bit for bit as the same tokens decoded one per cycle, however
// the prompt is cut. After a prefix-cache hit (no carried row) exactly one thing differs: the mtp head's kv of the
// entry before the chunk, which stays zero.
import std;
import engine.attn;
import engine.model;
import engine.prefill;
import engine.prefill_emulation;
import engine.prims;
import engine.testing;

namespace {

namespace emulation = engine::prefill_emulation;
namespace prefill = engine::prefill;
namespace model = engine::model;
namespace attn_device = engine::device::attn;

using emulation::Region;
using model::Count;

inline constexpr std::uint32_t slot = 5;
/** Two deltanet layers (ordinals 0 and 3) and two attention layers (kv layers 0 and 1), in trunk order. */
inline constexpr std::array<Count, 4> layers = {0, 3, 4, 7};

static_assert(model::trunk_mixers[0] == model::Mixer::LinearAttention && model::trunk_mixers[3] == model::Mixer::FullAttention);
static_assert(model::trunk_mixers[4] == model::Mixer::LinearAttention && model::trunk_mixers[7] == model::Mixer::FullAttention);

/** One chunk of the prompt, and whether the slot still carries the row before it. */
struct Cut {
	std::uint32_t length;
	prefill::Carry carry;
};

/** The slot's block table on both cards: scattered blocks, the same index on both cards. */
auto fill_tables(emulation::Memory& memory, Count positions) -> void {
	for (auto const card : {model::Card::Zero, model::Card::One}) {
		for (auto entry = Count{0}; entry * model::kv_block_tokens < positions; ++entry) {
			auto const block = static_cast<std::uint32_t>(7 + 3 * entry);
			memory.store_one(emulation::base(Region::Tables, card) + (Count{slot} * model::kv_table_blocks + entry) * 4, block);
		}
	}
}

[[nodiscard]] auto prefilled(emulation::Model const& weights, std::span<std::uint32_t const> tokens, std::span<Cut const> cuts)
    -> emulation::Memory {
	auto memory = emulation::Memory{weights};
	fill_tables(memory, tokens.size() + model::kv_block_tokens);
	// The slot held another request before: its state image is anything but zero until the prompt's first chunk clears it.
	for (auto const card : {model::Card::Zero, model::Card::One}) {
		memory.store<std::uint32_t>(emulation::base(Region::States, card) + Count{slot} * model::slot_state_bytes,
		                            std::vector<std::uint32_t>(4096, 0x3c00'3c00U));
	}
	auto epoch = std::uint64_t{1};
	auto from = std::uint32_t{0};
	for (auto const cut : cuts) {
		memory.store<std::uint32_t>(emulation::base(Region::Tokens, model::Card::Zero), tokens.subspan(from, cut.length));
		auto const facts = prefill::ChunkFacts{.slot = slot, .from = from, .count = cut.length, .carry = cut.carry};
		auto steps = std::array<std::vector<prefill::Step>, model::cards>{};
		auto next = epoch;
		for (auto const card : {model::Card::Zero, model::Card::One}) {
			auto sequence = prefill::Sequence{facts, emulation::view(card), epoch};
			sequence.begin();
			for (auto const layer : layers) {
				sequence.layer(layer);
			}
			sequence.end();
			next = sequence.epoch();
			steps.at(std::to_underlying(card)) = std::move(sequence).steps();
		}
		emulation::execute(weights, memory, steps);
		epoch = next;
		from += cut.length;
	}
	return memory;
}

[[nodiscard]] auto decoded(emulation::Model const& weights, std::span<std::uint32_t const> tokens)
    -> std::pair<emulation::Memory, std::vector<emulation::Bf16>> {
	auto memory = emulation::Memory{weights};
	fill_tables(memory, tokens.size() + model::kv_block_tokens);
	auto decoder = emulation::Decoder{weights, memory, slot, layers};
	for (auto const [position, token] : std::views::enumerate(tokens)) {
		decoder.step(token, static_cast<std::uint32_t>(position));
	}
	auto last = decoder.last();
	return {std::move(memory), std::move(last)};
}

/** The mtp head's kv at `position` on both cards, cleared: what a chunk without the carried row leaves there. */
auto clear_mtp_entry(emulation::Memory& memory, Count position) -> void {
	auto const zero = attn_device::Bf16{.bits = 0};
	for (auto const card : {model::Card::Zero, model::Card::One}) {
		for (auto head = Count{0}; head < attn_device::kv_heads; ++head) {
			for (auto const kind : {attn_device::Kind::Key, attn_device::Kind::Value}) {
				for (auto feature = Count{0}; feature < attn_device::head_dim; ++feature) {
					auto const block =
					    memory.value<std::uint32_t>(emulation::base(Region::Tables, card) +
						                            (Count{slot} * model::kv_table_blocks + position / model::kv_block_tokens) * 4);
					memory.store_one(emulation::base(Region::Pool, card) + Count{block} * model::kv_block_bytes +
					                     attn_device::element_offset(prefill::mtp_kv_layer, head, kind, position, feature),
					                 zero);
				}
			}
		}
	}
}

[[nodiscard]] auto same_bits(std::span<emulation::Bf16 const> left, std::span<emulation::Bf16 const> right) -> bool {
	return std::ranges::equal(left, right, [](auto a, auto b) {
		return a.bits == b.bits;
	});
}

/** Prefill of the prompt cut as `cuts`, against decode of the same tokens, on both cards. */
[[nodiscard]] auto prefill_equals_decode(std::uint64_t seed, std::vector<Cut> const& cuts) -> bool {
	auto generator = engine::testing::generator(seed);
	auto const weights = emulation::Model{seed};
	auto const total = std::ranges::fold_left(cuts, std::uint32_t{0}, [](std::uint32_t sum, Cut cut) {
		return sum + cut.length;
	});
	auto tokens = std::vector<std::uint32_t>{};
	for (auto index = std::uint32_t{0}; index < total; ++index) {
		tokens.push_back(
		    std::uniform_int_distribution<std::uint32_t>{0, static_cast<std::uint32_t>(model::sampled_vocabulary - 1)}(generator));
	}
	auto const chunked = prefilled(weights, tokens, cuts);
	auto [expected, last] = decoded(weights, tokens);
	auto from = std::uint32_t{0};
	for (auto const cut : cuts) {
		if (cut.carry == prefill::Carry::Absent && from > 0) {
			clear_mtp_entry(expected, from - 1);
		}
		from += cut.length;
	}
	auto const same = [&](Region region) {
		return chunked.same(expected, region, model::Card::Zero) && chunked.same(expected, region, model::Card::One);
	};
	auto const prompt = [&](model::Card card) {
		return chunked.values<emulation::Bf16>(emulation::base(Region::Prompt, card), model::hidden);
	};
	return same(Region::Pool) && same(Region::States) && same_bits(prompt(model::Card::Zero), last) &&
	       same_bits(prompt(model::Card::One), last);
}

[[nodiscard]] auto first(std::uint32_t length) -> Cut {
	return {.length = length, .carry = prefill::Carry::Absent};
}

[[nodiscard]] auto carried(std::uint32_t length) -> Cut {
	return {.length = length, .carry = prefill::Carry::Present};
}

/** A chunk after a prefix-cache hit: the slot's carried row is not the one before it. */
[[nodiscard]] auto resumed(std::uint32_t length) -> Cut {
	return {.length = length, .carry = prefill::Carry::Absent};
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "one chunk", .holds = prefill_equals_decode(21, {first(5)})},
	    engine::testing::Case{.name = "one token, then the rest", .holds = prefill_equals_decode(22, {first(1), carried(4)})},
	    engine::testing::Case{.name = "a token per chunk",
		                      .holds = prefill_equals_decode(23, {first(1), carried(1), carried(1), carried(1)})},
	    engine::testing::Case{.name = "two chunks", .holds = prefill_equals_decode(24, {first(3), carried(2)})},
	    engine::testing::Case{.name = "a prefix-cache hit loses only the mtp entry before it",
		                      .holds = prefill_equals_decode(25, {first(3), resumed(2)})},
	};
	return engine::testing::report(cases);
}
