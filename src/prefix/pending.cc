// SPDX-License-Identifier: AGPL-3.0-only
export module engine.prefix:pending;

import std;
import engine.core;
import engine.handle;
import engine.kv;
import engine.model;
import :point;
import :tree;

namespace engine::prefix {

/** A generational reference to one slot's announced prefill; stale once that prefill passes its points or ends. */
export struct PrefillHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator==(PrefillHandle const&) const -> bool = default;
};

/**
 * A slot is prefilling a prompt that shares this one's first `position` tokens and has not reached that point yet; when
 * it does, the cache holds the state there. A prompt that waits for it instead of prefilling the shared part itself
 * is prefilled once.
 */
export struct Wait {
	PrefillHandle prefill;
	Count position;

	[[nodiscard]] constexpr auto operator==(Wait const&) const -> bool = default;
};

/** The points of a prompt a slot is prefilling that it has not reached yet. */
struct Prefill {
	/** The prompt through its last pending point. */
	std::vector<Token> tokens;
	/** Pending points, increasing: positions [next, size) of the array. */
	std::array<Count, point_kinds> positions;
	Count next;
	Count size;

	[[nodiscard]] auto pending() const -> std::span<Count const> {
		return std::span{positions}.subspan(next, size - next);
	}
};

/** The announced prefills of the slots, at most one per slot: what a prompt arriving now could wait for. */
class Prefills {
public:
	/** The slot prefills `prompt` from `from`; its points past `from` are pending until it reaches them or leaves. */
	auto announce(kv::Slot slot, std::span<Token const> prompt, Points const& points, Count from) -> void {
		withdraw(slot);
		auto const ahead = points.after(from);
		if (ahead.empty()) {
			return;
		}
		core::invariant(ahead.back().position < prompt.size());
		auto prefill = Prefill{
		    .tokens = {prompt.begin(), prompt.begin() + core::narrow<std::ptrdiff_t>(ahead.back().position)},
		    .positions = {},
		    .next = 0,
		    .size = ahead.size(),
		};
		std::ranges::transform(ahead, prefill.positions.begin(), &Point::position);
		auto const handle = table_.insert(std::move(prefill));
		core::invariant(handle.has_value());
		slots_[slot.index()] = handle;
	}

	/** The slot's prefill has reached `position`: its points up to there are no longer pending. */
	auto reached(kv::Slot slot, Count position) -> void {
		auto const handle = slots_[slot.index()];
		if (!handle) {
			return;
		}
		auto& prefill = table_.get(*handle);
		while (prefill.next < prefill.size && prefill.positions[prefill.next] <= position) {
			prefill.next += 1;
		}
		if (prefill.next == prefill.size) {
			withdraw(slot);
		}
	}

	auto withdraw(kv::Slot slot) -> void {
		if (auto const handle = std::exchange(slots_[slot.index()], std::nullopt)) {
			table_.release(*handle);
		}
	}

	/** The deepest pending point above `above` and at most `limit` whose prefix `prompt` shares. */
	[[nodiscard]] auto deepest(std::span<Token const> prompt, Count above, Count limit) const -> std::optional<Wait> {
		auto best = std::optional<Wait>{};
		for (auto const handle : table_.handles()) {
			best = table_.try_get(handle)
			           .transform([&](Prefill const& prefill) {
				           return deeper(prefill, handle, prompt, best ? best->position : above, limit).or_else([&] {
					           return best;
				           });
			           })
			           .value_or(best);
		}
		return best;
	}

	[[nodiscard]] auto pending(Wait wait) const -> bool {
		auto const prefill = table_.try_get(wait.prefill);
		return prefill.has_value() && std::ranges::contains(prefill->pending(), wait.position);
	}

private:
	/** The deepest pending point of `prefill` above `above` and at most `limit` whose prefix `prompt` shares. */
	[[nodiscard]] static auto deeper(Prefill const& prefill, PrefillHandle handle, std::span<Token const> prompt, Count above, Count limit)
	    -> std::optional<Wait> {
		auto const deepest_first = prefill.pending() | std::views::reverse;
		auto const found = std::ranges::find_if(deepest_first, [&](Count position) {
			return position > above && position <= limit && std::ranges::equal(prompt.first(position), std::span{prefill.tokens}.first(position));
		});
		return found == deepest_first.end() ? std::nullopt : std::optional{Wait{.prefill = handle, .position = *found}};
	}

	handle::Table<Prefill, PrefillHandle, model::slots, generation_bits> table_;
	std::array<std::optional<PrefillHandle>, model::slots> slots_{};
};

}
