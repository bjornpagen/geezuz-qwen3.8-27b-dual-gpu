// SPDX-License-Identifier: AGPL-3.0-only
//
// The engine's side of the determinism harness: token sequences teacher-forced through the cards one position per
// logits ticket, every row's logits handed to the caller. A walk drives the slots directly, without the api or the
// scheduler (which plans drafted cycles, not logits tickets): sequence i lives in slot i, and the kv pool, the one
// bookkeeping of which block holds what, gives each slot the block its next position writes, its commands riding ahead
// of the ticket.
export module engine.cards:harness;

import std;
import engine.core;
import engine.decode;
import engine.exec;
import engine.kv;
import engine.model;
import engine.sampling;
import engine.sched;
import :adapter;

namespace engine::cards {

namespace ex = engine::exec;

/** Why a walk did not finish: the cards failed, or the pool could not hold the sequences. */
export struct [[nodiscard]] WalkError {
	std::variant<decode::DecodeError, kv::Exhausted> reason;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit(
		    [](auto const& alternative) {
			    return alternative.retry();
		    },
		    reason);
	}

	[[nodiscard]] auto operator==(WalkError const&) const -> bool = default;
};

/**
 * One row a walk computed: sequence `sequence`'s logits after its token at `position`, over the sampled vocabulary in
 * token order. A borrow of the walk's own storage, valid for the call that hands it over.
 */
export struct Forced {
	std::size_t sequence;
	sched::Count position;
	std::span<float const> logits;
};

/**
 * A logits ticket plans no sampling, but its rows carry a rule: greedy, the one that draws nothing. A row may commit
 * only its own token.
 */
[[nodiscard]] auto forced_row(std::size_t sequence, sched::Count position, sched::Token token) -> sched::Row {
	return {
	    .slot = kv::Slot::at(sequence),
	    .request = sched::RequestHandle{.bits = sequence + 1},
	    .position = position,
	    .token = token,
	    .rule = sampling::rule(sampling::Params{sampling::Greedy{}}, 0),
	    .most = 1,
	};
}

/**
 * The cards and the pool a walk drives. The pool is large (a block table per slot), so it is held behind a pointer.
 * Each walk leaves the pool as it found it: every block its slots took is released when it ends.
 */
export template<decode::Catalogs C>
class BasicHarness {
public:
	explicit BasicHarness(BasicCards<C> cards) : cards_{std::move(cards)}, pool_{std::make_unique<kv::Pool>()} {}

	/**
	 * Teacher-forces `sequences` side by side, sequence i in slot i from position 0: each ticket takes the next token of
	 * every sequence that has one left, so all of them are live until the shorter ones end. `each(Forced)` hears every
	 * row, ticket by ticket, in slot order within a ticket.
	 */
	template<std::invocable<Forced const&> Each>
	[[nodiscard]] auto walk(std::span<std::vector<sched::Token> const> sequences, Each each) -> core::Result<void, WalkError> {
		core::invariant(!sequences.empty() && sequences.size() <= model::slots);
		auto const longest = std::ranges::max(sequences | std::views::transform([](std::vector<sched::Token> const& sequence) {
			                                      return sequence.size();
		                                      }));
		auto walked = core::Result<void, WalkError>{};
		for (auto position = sched::Count{0}; position < longest && walked; ++position) {
			walked = step(sequences, position, each);
		}
		for (auto const index : std::views::iota(std::size_t{0}, sequences.size())) {
			pool_->trim(kv::Slot::at(index), 0);
		}
		return walked;
	}

private:
	template<class Each>
	[[nodiscard]] auto step(std::span<std::vector<sched::Token> const> sequences, sched::Count position, Each& each) -> core::Result<void, WalkError> {
		auto batch = sched::Batch{.depth = model::DraftDepth::One, .rows = {}};
		auto extends = std::vector<kv::Extend>{};
		auto walking = std::vector<std::size_t>{};
		for (auto const [index, sequence] : std::views::enumerate(sequences)) {
			auto const at = core::narrow<std::size_t>(index);
			if (position >= sequence.size()) {
				continue;
			}
			batch.rows.push_back(forced_row(at, position, sequence[position]));
			extends.push_back(kv::Extend{.slot = kv::Slot::at(at), .length = position, .until = position + 1});
			walking.push_back(at);
		}
		if (auto room = pool_->ensure(cards_, extends); !room) {
			return std::unexpected(WalkError{.reason = room.error()});
		}
		auto waited = ex::wait<decode::DecodeError>(cards_.logits(std::move(batch)));
		// Nothing asks the harness's wait to stop, and the cards' senders are never stopped.
		core::invariant(waited.has_value());
		if (!*waited) {
			return std::unexpected(WalkError{.reason = waited->error()});
		}
		auto const& [logits] = **waited;
		core::invariant(logits.rows.size() == walking.size());
		for (auto const [row, sequence] : std::views::enumerate(walking)) {
			std::invoke(each, Forced{.sequence = sequence, .position = position, .logits = logits.rows[core::narrow<std::size_t>(row)]});
		}
		return {};
	}

	BasicCards<C> cards_;
	std::unique_ptr<kv::Pool> pool_;
};

/** The harness over the model's images. */
export using Harness = BasicHarness<decode::ModelCatalogs>;

}
