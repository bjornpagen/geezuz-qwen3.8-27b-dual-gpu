// SPDX-License-Identifier: AGPL-3.0-only
export module engine.kv_fake;

import std;
import engine.kv;
import engine.model;

/**
 * The test double of the cards: a content-checking fake device. It applies the pool's commands to a model of the
 * bytes, in which every position of a block holds either garbage, zero, or the kv of one exact token sequence, and
 * every snapshot block holds one part of one saved state. Writes go through the device's own block tables, as the
 * kernels' do, so a missing or stale table entry lands the kv in the wrong block and shows up when the harness reads
 * a sequence back.
 */
namespace engine::kv_fake {

using kv::BlockIndex;
using kv::Count;
using kv::Geometry;
using kv::Slot;

/** Memory never written, or a snapshot's bytes read as kv: what attention must never meet. */
export struct Garbage {
	[[nodiscard]] constexpr auto operator==(Garbage const&) const -> bool = default;
};

export struct Zero {
	[[nodiscard]] constexpr auto operator==(Zero const&) const -> bool = default;
};

/** The kv of the last position of the sequence whose tokens hash to `chain`. */
export struct Kv {
	std::uint64_t chain;

	[[nodiscard]] constexpr auto operator==(Kv const&) const -> bool = default;
};

export using Cell = std::variant<Garbage, Zero, Kv>;

/** Part `part` of the deltanet state `state`. */
export struct Part {
	std::uint64_t state;
	Count part;

	[[nodiscard]] constexpr auto operator==(Part const&) const -> bool = default;
};

export using Content = std::variant<std::vector<Cell>, Part>;

template<class... F>
struct Overloaded : F... {
	using F::operator()...;
};

export template<Geometry G>
class FakeDevice {
public:
	FakeDevice()
	    : blocks_(G.pool_blocks, Content{std::vector<Cell>(G.block_tokens, Cell{Garbage{}})}),
	      tables_(model::slots, std::vector<std::optional<BlockIndex>>(G.table_blocks)), states_(model::slots, 0) {}

	auto apply(kv::Command<G> const& command) -> void {
		log_.push_back(command);
		std::visit(Overloaded{
		               [&](kv::ZeroBlock const& zero) {
			               block(zero.block) = std::vector<Cell>(G.block_tokens, Cell{Zero{}});
		               },
		               [&](kv::CopyPrefix const& copy) {
			               if (!std::holds_alternative<std::vector<Cell>>(block(copy.source)) || copy.positions > G.block_tokens ||
			                   copy.source == copy.target) {
				               violation(std::format("copy of block {} into {} is not a kv prefix copy", copy.source, copy.target));
				               return;
			               }
			               auto const& source = std::get<std::vector<Cell>>(block(copy.source));
			               auto cells = std::vector<Cell>(G.block_tokens, Cell{Zero{}});
			               std::ranges::copy(std::span{source}.first(copy.positions), cells.begin());
			               block(copy.target) = std::move(cells);
		               },
		               [&](kv::SetTableEntry const& set) {
			               if (set.entry >= G.table_blocks || set.block >= G.pool_blocks) {
				               violation(std::format("table entry {} names block {}", set.entry, set.block));
				               return;
			               }
			               tables_[set.slot.index()][set.entry] = set.block;
		               },
		               [&](kv::SaveSnapshot<G> const& save) {
			               for (auto part = Count{0}; part < G.snapshot_blocks; ++part) {
				               block(save.blocks[part]) = Part{.state = states_[save.slot.index()], .part = part};
			               }
		               },
		               [&](kv::LoadSnapshot<G> const& load) {
			               auto const state = std::holds_alternative<Part>(block(load.blocks[0]))
			                                      ? std::optional{std::get<Part>(block(load.blocks[0])).state}
			                                      : std::nullopt;
			               auto const intact = state.has_value() && std::ranges::all_of(std::views::iota(Count{0}, G.snapshot_blocks), [&](Count part) {
				               return block(load.blocks[part]) == Content{Part{.state = *state, .part = part}};
			               });
			               if (!intact) {
				               violation(std::format("snapshot loaded into slot {} is not intact", load.slot.index()));
				               return;
			               }
			               states_[load.slot.index()] = *state;
		               },
		           },
		           command);
	}

	/** What a kernel writing position `position` of the slot's sequence does: the device table picks the block. */
	auto write(Slot slot, Count position, Cell cell) -> void {
		auto const entry = tables_[slot.index()][position / G.block_tokens];
		if (!entry) {
			violation(std::format("slot {} writes position {} through an unset table entry", slot.index(), position));
			return;
		}
		if (!std::holds_alternative<std::vector<Cell>>(block(*entry))) {
			violation(std::format("slot {} writes kv into snapshot block {}", slot.index(), *entry));
			return;
		}
		std::get<std::vector<Cell>>(block(*entry))[position % G.block_tokens] = cell;
	}

	/** The device table entry the kernels would read; nullopt when it was never set. */
	[[nodiscard]] auto entry(Slot slot, Count entry) const -> std::optional<BlockIndex> {
		return tables_[slot.index()][entry];
	}

	[[nodiscard]] auto content(BlockIndex index) const -> Content const& {
		return blocks_[index];
	}

	[[nodiscard]] auto state(Slot slot) const -> std::uint64_t {
		return states_[slot.index()];
	}

	/** The slot's deltanet state as the decode kernels would leave it; the harness names it by its sequence. */
	auto set_state(Slot slot, std::uint64_t state) -> void {
		states_[slot.index()] = state;
	}

	/** Every command applied since the last call, in order. */
	[[nodiscard]] auto take_log() -> std::vector<kv::Command<G>> {
		return std::exchange(log_, {});
	}

	[[nodiscard]] auto violations() const -> std::vector<std::string> {
		return violations_;
	}

private:
	[[nodiscard]] auto block(BlockIndex index) -> Content& {
		return blocks_.at(index);
	}

	auto violation(std::string what) -> void {
		violations_.push_back(std::move(what));
	}

	std::vector<Content> blocks_;
	std::vector<std::vector<std::optional<BlockIndex>>> tables_;
	std::vector<std::uint64_t> states_;
	std::vector<kv::Command<G>> log_;
	std::vector<std::string> violations_;
};

}
