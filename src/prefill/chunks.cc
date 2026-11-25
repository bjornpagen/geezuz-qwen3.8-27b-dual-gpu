// SPDX-License-Identifier: AGPL-3.0-only
//
// The chunks' runner: both cards' prefill workspaces and programs, the mapped buffer a chunk's tokens cross in, the
// exchange's epoch both cards share, and each slot's carried final-norm row. A run enqueues one chunk's sequence on
// both cards' streams and hands back a sender that completes once both cards have finished it. The decode side yields
// its resident loops before a run and builds the memories each run borrows.
export module engine.prefill:chunks;

import std;
import engine.attn;
import engine.card_scheduler;
import engine.core;
import engine.deltanet;
import engine.device;
import engine.exec;
import engine.model;
import engine.prefill_device;
import engine.sched;
import :memories;
import :steps;

namespace engine::prefill {

namespace ex = engine::exec;
namespace pd = engine::prefill_device;

static_assert(sched::tuning.chunk_tokens <= pd::max_rows, "the prefill workspace holds the scheduler's largest chunk");

/** One card as a run uses it: its context and the stream its chunks are ordered on, borrowed for the call. */
export struct CardStream {
	device::Context const& context;
	device::Stream const& stream;
};

/** Both cards, in rank order. */
export using CardStreams = std::array<CardStream, model::cards>;

/** The sender a run returns: both cards' schedulers, joined. */
export using Finished =
    decltype(ex::when_all(std::declval<card_scheduler::Card const&>().schedule(), std::declval<card_scheduler::Card const&>().schedule()));

using AttentionProgram = device::Program<attn::Append, attn::Prefill>;
using DeltanetProgram = device::Program<deltanet::Prefill, deltanet::PrefillWindow>;
using WorkspaceArray = device::DeviceArray<std::byte, pd::workspace.bytes>;
using Tokens = device::MappedArray<std::uint32_t, pd::max_rows>;

/** One card's own memory and the three programs a chunk launches from. */
struct Card {
	WorkspaceArray workspace;
	pd::Program prefill;
	AttentionProgram attention;
	DeltanetProgram deltanet;
};

/** Loads every kernel of program type `P` from its image on `context`. */
template<class... Ks>
[[nodiscard]] auto load_program(device::Context const& context, std::type_identity<device::Program<Ks...>> /*program*/)
    -> core::Result<device::Program<Ks...>, device::DeviceError> {
	return context.template load<Ks...>();
}

/** What a slot's carried row holds: the final norm of `request`'s token at `position`. */
struct Carried {
	sched::RequestHandle request;
	std::uint64_t position;

	[[nodiscard]] constexpr auto operator==(Carried const&) const -> bool = default;
};

/** Launches one step on one card from the program holding its kernel. */
struct Enqueue {
	CardStream card;
	Card const& own;

	template<class K>
	[[nodiscard]] auto launch(auto const& program, Launch<K> const& step) const -> core::Result<void, device::DeviceError> {
		return card.context.template launch<K>(program, card.stream, step.params);
	}

	[[nodiscard]] auto operator()(PrefillLaunch const& step) const -> core::Result<void, device::DeviceError> {
		return std::visit(
		    [&](auto const& launched) {
			    return launch(own.prefill, launched);
		    },
		    step);
	}

	[[nodiscard]] auto operator()(AttentionLaunch const& step) const -> core::Result<void, device::DeviceError> {
		return std::visit(
		    [&](auto const& launched) {
			    return launch(own.attention, launched);
		    },
		    step);
	}

	[[nodiscard]] auto operator()(DeltanetLaunch const& step) const -> core::Result<void, device::DeviceError> {
		return std::visit(
		    [&](auto const& launched) {
			    return launch(own.deltanet, launched);
		    },
		    step);
	}
};

export class Chunks {
public:
	/**
	 * Allocates both cards' workspaces, loads the programs, and enqueues each workspace's zeroing on its card's stream,
	 * so every exchange flag reads zero before the first epoch.
	 */
	[[nodiscard]] static auto create(CardStreams const& cards) -> core::Result<Chunks, device::DeviceError> {
		auto tokens = cards[0].context.allocate_mapped<std::uint32_t, pd::max_rows>();
		if (!tokens) {
			return std::unexpected(tokens.error());
		}
		auto zero = load(cards[0]);
		if (!zero) {
			return std::unexpected(zero.error());
		}
		auto one = load(cards[1]);
		if (!one) {
			return std::unexpected(one.error());
		}
		if (auto ordered = after_both_zeroed(cards); !ordered) {
			return std::unexpected(ordered.error());
		}
		return Chunks{std::move(*tokens), std::move(*zero), std::move(*one)};
	}

	/**
	 * Enqueues `chunk` on both cards and returns the sender of its completion. Both cards' streams must already be
	 * ordered after whatever last ran on the cards (the decode loops have yielded). The memories and the streams are
	 * borrowed for this call; the sender borrows the streams until it completes, and the next run waits for it.
	 */
	[[nodiscard]] auto run(Memories const& memories, sched::Chunk const& chunk, CardStreams const& cards)
	    -> core::Result<Finished, device::DeviceError> {
		auto const slot = chunk.slot.index();
		auto const count = core::narrow<std::uint32_t>(chunk.tokens.size());
		auto const from = core::narrow<std::uint32_t>(chunk.from);
		auto const facts = ChunkFacts{
		    .slot = core::narrow<std::uint32_t>(slot),
		    .from = from,
		    .count = count,
		    .carry = carry(slot, chunk),
		};
		std::ranges::copy(chunk.tokens, tokens_.host().begin());
		auto next = epoch_;
		for (auto const index : std::views::iota(std::size_t{0}, model::cards)) {
			auto const card = static_cast<model::Card>(index);
			auto const view = CardView{
			    .card = card,
			    .memories = memories.cards[index],
			    .workspace = cards_[index].workspace.address(),
			    .peer_workspace = cards_[1 - index].workspace.address(),
			    .embedding = memories.embedding,
			    .tokens = tokens_.device_address(),
			};
			auto const sequenced = sequence(facts, view, epoch_);
			next = sequenced.next_epoch;
			auto const enqueue = Enqueue{.card = cards[index], .own = cards_[index]};
			for (auto const& step : sequenced.steps) {
				// A launch the driver refuses leaves the cards mid-chunk, the peer possibly waiting on an exchange:
				// the engine stops serving, so nothing here is rolled back.
				if (auto launched = std::visit(enqueue, step); !launched) {
					return std::unexpected(launched.error());
				}
			}
		}
		epoch_ = next;
		carried_[slot] = Carried{.request = chunk.request, .position = chunk.from + count - 1};
		return ex::when_all(card_scheduler::Card{cards[0].context, cards[0].stream}.schedule(),
		                    card_scheduler::Card{cards[1].context, cards[1].stream}.schedule());
	}

private:
	/**
	 * The carried row is the final norm of one request's token at one position, a function of the sequence alone, so
	 * it serves any later chunk of the same request that starts right after it, whatever ran in between.
	 */
	[[nodiscard]] auto carry(std::size_t slot, sched::Chunk const& chunk) const -> Carry {
		if (chunk.from == 0) {
			return Carry::Absent;
		}
		auto const before = Carried{.request = chunk.request, .position = chunk.from - 1};
		return carried_[slot] == before ? Carry::Present : Carry::Absent;
	}

	/**
	 * Each card pushes into the other's workspace, so neither stream may run a chunk before the other card's zeroing
	 * is done: each stream waits for the other card's.
	 */
	[[nodiscard]] static auto after_both_zeroed(CardStreams const& cards) -> core::Result<void, device::DeviceError> {
		auto zero = cards[0].context.create_event();
		auto one = zero.and_then([&](auto const&) {
			return cards[1].context.create_event();
		});
		if (!one) {
			return std::unexpected(one.error());
		}
		return zero->record(cards[0].stream)
		    .and_then([&] {
			    return one->record(cards[1].stream);
		    })
		    .and_then([&] {
			    return cards[0].stream.wait(*one);
		    })
		    .and_then([&] {
			    return cards[1].stream.wait(*zero);
		    });
	}

	Chunks(Tokens tokens, Card zero, Card one) : tokens_{std::move(tokens)}, cards_{std::move(zero), std::move(one)} {}

	[[nodiscard]] static auto load(CardStream const& card) -> core::Result<Card, device::DeviceError> {
		auto workspace = card.context.allocate<std::byte, pd::workspace.bytes>();
		if (!workspace) {
			return std::unexpected(workspace.error());
		}
		auto prefill = load_program(card.context, std::type_identity<pd::Program>{});
		if (!prefill) {
			return std::unexpected(prefill.error());
		}
		auto attention = load_program(card.context, std::type_identity<AttentionProgram>{});
		if (!attention) {
			return std::unexpected(attention.error());
		}
		auto core = load_program(card.context, std::type_identity<DeltanetProgram>{});
		if (!core) {
			return std::unexpected(core.error());
		}
		auto zeroed =
		    card.context.launch<pd::Zero>(*prefill, card.stream, {.target = workspace->address(), .chunks = pd::workspace.bytes / 16});
		if (!zeroed) {
			return std::unexpected(zeroed.error());
		}
		return Card{.workspace = std::move(*workspace),
		            .prefill = std::move(*prefill),
		            .attention = std::move(*attention),
		            .deltanet = std::move(*core)};
	}

	Tokens tokens_;
	std::array<Card, model::cards> cards_;
	std::uint64_t epoch_ = 1;
	std::array<std::optional<Carried>, model::slots> carried_{};
};

}
