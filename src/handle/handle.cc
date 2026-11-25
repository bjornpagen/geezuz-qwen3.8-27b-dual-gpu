// SPDX-License-Identifier: AGPL-3.0-only
export module engine.handle;

import std;
import engine.core;

namespace engine::handle {

/**
 * A generational reference into one owner's table: `{index, generation}` packed in one integer. Each table kind
 * declares its own handle struct (`struct SlotHandle { std::uint64_t bits; ... }`), so a handle for one table cannot
 * be passed to another; the struct names the table, it is not a phantom tag over a shared representation.
 */
export template<class H>
concept Handle = std::regular<H> && std::is_aggregate_v<H> && sizeof(H) == sizeof(std::uint64_t) && requires(H handle) {
	{ handle.bits } -> std::same_as<std::uint64_t&>;
	H{.bits = std::uint64_t{}};
};

/**
 * A bounded table of values addressed by generational handles. A looked-up reference is valid only for the call that
 * obtained it and is never stored. Releasing a slot advances its generation, so every handle to the old occupant is
 * stale from then on; a slot whose generation is exhausted retires instead of wrapping, so no stale handle can ever
 * become valid again.
 */
export template<class T, Handle H, std::size_t Capacity, std::uint8_t GenerationBits>
class Table {
public:
	static_assert(Capacity > 0);
	static_assert(GenerationBits >= 2);
	static constexpr std::uint64_t index_bits = std::bit_width(Capacity - 1);
	static_assert(index_bits + GenerationBits <= 64, "a handle is one 64-bit integer");
	static constexpr std::uint64_t generation_limit = (std::uint64_t{1} << GenerationBits) - 1;

	Table() {
		for (auto index = std::size_t{0}; index < Capacity; ++index) {
			free_[index] = Capacity - 1 - index;
		}
	}

	/** Stores `value` in a free slot; nullopt when every slot is live or retired. */
	[[nodiscard]] auto insert(T value) -> std::optional<H> {
		if (free_count_ == 0) {
			return std::nullopt;
		}
		--free_count_;
		auto const index = free_[free_count_];
		auto& slot = slots_[index];
		slot.value.emplace(std::move(value));
		return H{.bits = (slot.generation << index_bits) | core::narrow<std::uint64_t>(index)};
	}

	[[nodiscard]] auto try_get(H handle) -> std::optional<T&> {
		auto const index = index_of(handle);
		if (!index) {
			return std::nullopt;
		}
		return *slots_[*index].value;
	}

	[[nodiscard]] auto try_get(H handle) const -> std::optional<T const&> {
		auto const index = index_of(handle);
		if (!index) {
			return std::nullopt;
		}
		return *slots_[*index].value;
	}

	/** The value of a handle the caller knows is live; a stale handle is an invariant violation. */
	[[nodiscard]] auto get(H handle) -> T& {
		auto found = try_get(handle);
		core::invariant(found.has_value());
		return *found;
	}

	/** Destroys the value of a live handle and makes every handle to it stale. */
	auto release(H handle) -> void {
		auto const index = index_of(handle);
		core::invariant(index.has_value());
		auto& slot = slots_[*index];
		slot.value.reset();
		++slot.generation;
		if (slot.generation <= generation_limit) {
			free_[free_count_] = *index;
			++free_count_;
		}
	}

	/**
	 * Every live handle, in slot order: a lazy view over this table, valid while it lives and only while nothing is
	 * inserted or released (its filter reads the table's state, which must hold still while it is iterated).
	 */
	[[nodiscard]] auto handles() const {
		return std::views::iota(std::size_t{0}, Capacity) | std::views::filter([this](std::size_t index) {
			       return slots_[index].value.has_value();
		       }) |
		       std::views::transform([this](std::size_t index) {
			       return H{.bits = (slots_[index].generation << index_bits) | core::narrow<std::uint64_t>(index)};
		       });
	}

	[[nodiscard]] auto live_count() const -> std::size_t {
		return core::narrow<std::size_t>(std::ranges::count_if(slots_, [](Slot const& slot) {
			return slot.value.has_value();
		}));
	}

	[[nodiscard]] auto retired_count() const -> std::size_t {
		return core::narrow<std::size_t>(std::ranges::count_if(slots_, [](Slot const& slot) {
			return slot.generation > generation_limit;
		}));
	}

private:
	struct Slot {
		std::optional<T> value;
		std::uint64_t generation = 1;
	};

	[[nodiscard]] auto index_of(H handle) const -> std::optional<std::size_t> {
		auto const index = core::narrow<std::size_t>(handle.bits & ((std::uint64_t{1} << index_bits) - 1));
		if (index >= Capacity) {
			return std::nullopt;
		}
		auto const& slot = slots_[index];
		if (!slot.value || slot.generation != handle.bits >> index_bits) {
			return std::nullopt;
		}
		return index;
	}

	std::array<Slot, Capacity> slots_{};
	std::array<std::size_t, Capacity> free_{};
	std::size_t free_count_ = Capacity;
};

}
