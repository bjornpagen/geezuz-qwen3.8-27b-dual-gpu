// SPDX-License-Identifier: AGPL-3.0-only
export module engine.kv:pool;

import std;
import engine.core;
import engine.handle;
import engine.model;
import :device;
import :error;
import :geometry;

namespace engine::kv {

/** Who keeps a kv block live: entries of slot tables, and references taken with `retain` (the prefix cache's). */
export struct References {
	Count tables;
	Count held;

	[[nodiscard]] constexpr auto operator==(References const&) const -> bool = default;
};

/**
 * `slot` holds positions [0, length) of its sequence and is about to write positions [length, until). Whatever its
 * blocks hold at or past `length` is not part of its sequence, so a block copied for this write keeps only the
 * positions below `length`.
 */
export struct Extend {
	Slot slot;
	Count length;
	Count until;
};

/** A block holding kv, counted by its holders; it is free again when both counts reach zero. */
struct KvBlock {
	References references;
};

/** A block holding part `part` of one snapshot, which owns it alone. */
struct SnapshotPart {
	Count part;
};

using BlockUse = std::variant<KvBlock, SnapshotPart>;

enum class Holder : std::uint8_t {
	Table,
	Reference,
};

// Enough that no block or snapshot record exhausts its generations in any lifetime of the server (2^48 reuses), while
// leaving 16 index bits, which every pool the cards can hold fits.
inline constexpr std::uint8_t generation_bits = 48;

template<Geometry G>
struct SnapshotRecord {
	std::array<BlockHandle, G.snapshot_blocks> parts;
};

/**
 * The host-side bookkeeping of the block pool that kv and deltanet snapshots share. It decides which block holds what
 * and keeps every slot's block table; the bytes live on the cards, changed only through the commands it hands the
 * device. A kv block shared by several holders (slot tables through a fork or the prefix cache, and the cache's own
 * references) is copied before a slot writes into it. Snapshots draw from the same free list, and their table holds at
 * most `snapshot_capacity`, so the cap on the blocks they take holds by construction.
 *
 * Large (a block table per slot): the owner keeps it behind a `std::unique_ptr`.
 */
export template<Geometry G>
class BasicPool {
	static_assert(sound(G));

	using Blocks = handle::Table<BlockUse, BlockHandle, G.pool_blocks, generation_bits>;
	using Snapshots = handle::Table<SnapshotRecord<G>, SnapshotHandle, G.snapshot_capacity, generation_bits>;

public:
	/** The blocks of one snapshot, in part order. */
	using SnapshotBlocks = std::array<BlockIndex, G.snapshot_blocks>;

	BasicPool() = default;
	BasicPool(BasicPool const&) = delete;
	auto operator=(BasicPool const&) -> BasicPool& = delete;
	BasicPool(BasicPool&&) noexcept = default;
	auto operator=(BasicPool&&) noexcept -> BasicPool& = default;
	~BasicPool() = default;

	[[nodiscard]] auto free_blocks() const -> Count {
		return G.pool_blocks - blocks_.live_count() - blocks_.retired_count();
	}

	/** Kv blocks only `retain` references keep: what releasing every such reference would free. */
	[[nodiscard]] auto held_only_blocks() const -> Count {
		return held_only_;
	}

	[[nodiscard]] auto snapshots() const -> Count {
		return snapshots_.live_count();
	}

	/** Entries in the slot's block table: it holds positions [0, table_size * block_tokens). */
	[[nodiscard]] auto table_size(Slot slot) const -> Count {
		return tables_[slot.index()].size;
	}

	/** The block at entry `entry` of the slot's table, which holds positions [entry, entry + 1) * block_tokens. */
	[[nodiscard]] auto entry(Slot slot, Count entry) const -> BlockHandle {
		core::invariant(entry < table_size(slot));
		return tables_[slot.index()].entries[entry];
	}

	/** The device index of a live block. */
	[[nodiscard]] auto index(BlockHandle block) const -> BlockIndex {
		core::invariant(blocks_.try_get(block).has_value());
		return index_of(block);
	}

	/** The holders of a live kv block; nullopt for a stale handle. */
	[[nodiscard]] auto references(BlockHandle block) const -> std::optional<References> {
		auto const use = blocks_.try_get(block);
		if (!use) {
			return std::nullopt;
		}
		core::invariant(std::holds_alternative<KvBlock>(*use));
		return std::get<KvBlock>(*use).references;
	}

	/** The blocks of a live snapshot; nullopt for a stale handle. */
	[[nodiscard]] auto snapshot_blocks(SnapshotHandle snapshot) const -> std::optional<SnapshotBlocks> {
		auto const record = snapshots_.try_get(snapshot);
		if (!record) {
			return std::nullopt;
		}
		return indices(record->parts);
	}

	/**
	 * Blocks `ensure` would take from the free list for `extends`: one per block past each table's end, and a copy of
	 * every shared block a slot writes into, except that the last writer of a block only the listed slots hold writes
	 * it in place.
	 */
	[[nodiscard]] auto need(std::span<Extend const> extends) const -> Count {
		auto needed = Count{0};
		for (auto const [position, extend] : std::views::enumerate(extends)) {
			auto const& blocks = tables_[extend.slot.index()];
			core::invariant(extend.length <= extend.until && extend.until <= table_positions(G));
			core::invariant(extend.length <= blocks.size * G.block_tokens);
			auto const want = model::divide_up(extend.until, G.block_tokens);
			needed += want - std::min(want, blocks.size);
			auto const earlier = extends.first(core::narrow<std::size_t>(position));
			for (auto const block : written(extend)) {
				auto const holders = kv(block).references;
				if (holders.tables + holders.held <= 1) {
					continue;
				}
				needed += 1;
				// Counted once, at its first writer: the block every holder of which writes it needs one copy fewer.
				auto const first = std::ranges::none_of(earlier, [&](Extend const& other) {
					return writes(other, block);
				});
				if (first && holders.held == 0 && writers(extends, block) == holders.tables) {
					needed -= 1;
				}
			}
		}
		return needed;
	}

	/**
	 * Makes positions [length, until) of every listed slot writable, all or nothing: blocks past each table's end are
	 * added zeroed, shared blocks in the written range are replaced by private copies, and the device's tables follow.
	 * On `Exhausted` nothing changed and no command was issued. A slot appears at most once.
	 */
	template<Device<G> D>
	[[nodiscard]] auto ensure(D& device, std::span<Extend const> extends) -> core::Result<void, Exhausted> {
		for (auto const [position, extend] : std::views::enumerate(extends)) {
			core::invariant(std::ranges::none_of(extends.first(core::narrow<std::size_t>(position)), [&](Extend const& other) {
				return other.slot == extend.slot;
			}));
		}
		auto const needed = need(extends);
		auto const free = free_blocks();
		if (needed > free) {
			return std::unexpected(Exhausted{.needed = needed, .free = free});
		}
		for (auto const& extend : extends) {
			auto& blocks = tables_[extend.slot.index()];
			auto const want = model::divide_up(extend.until, G.block_tokens);
			for (auto entry = extend.length / G.block_tokens; entry < std::min(blocks.size, want); ++entry) {
				auto const shared = blocks.entries[entry];
				auto const holders = kv(shared).references;
				if (holders.tables + holders.held <= 1) {
					continue;
				}
				auto const copy = allocate();
				device.apply(Command<G>{CopyPrefix{
				    .source = index_of(shared),
				    .target = index_of(copy),
				    .positions = extend.length - std::min(extend.length, entry * G.block_tokens),
				}});
				remove(shared, Holder::Table);
				place(device, extend.slot, entry, copy);
			}
			for (auto entry = blocks.size; entry < want; ++entry) {
				auto const fresh = allocate();
				device.apply(Command<G>{ZeroBlock{.block = index_of(fresh)}});
				place(device, extend.slot, entry, fresh);
			}
			blocks.size = std::max(blocks.size, want);
		}
		return {};
	}

	/** Drops the slot's blocks that hold only positions at or past `length`. */
	auto trim(Slot slot, Count length) -> void {
		auto& blocks = tables_[slot.index()];
		auto const keep = model::divide_up(length, G.block_tokens);
		while (blocks.size > keep) {
			--blocks.size;
			remove(blocks.entries[blocks.size], Holder::Table);
		}
	}

	/** Gives an empty slot the live kv blocks `prefix` (distinct, in position order) as its table, shared. */
	template<Device<G> D>
	auto attach(D& device, Slot slot, std::span<BlockHandle const> prefix) -> void {
		auto& blocks = tables_[slot.index()];
		core::invariant(blocks.size == 0 && prefix.size() <= G.table_blocks);
		for (auto const [entry, block] : std::views::enumerate(prefix)) {
			add(block, Holder::Table);
			place(device, slot, core::narrow<Count>(entry), block);
		}
		blocks.size = prefix.size();
	}

	/** Gives the empty slot `to` the blocks of `from` that hold positions [0, length), shared. */
	template<Device<G> D>
	auto fork(D& device, Slot from, Slot to, Count length) -> void {
		core::invariant(from != to && tables_[to.index()].size == 0);
		auto const keep = model::divide_up(length, G.block_tokens);
		core::invariant(keep <= tables_[from.index()].size);
		for (auto entry = Count{0}; entry < keep; ++entry) {
			auto const block = tables_[from.index()].entries[entry];
			add(block, Holder::Table);
			place(device, to, entry, block);
		}
		tables_[to.index()].size = keep;
	}

	/** A reference to the block at `entry` of the slot's table, which keeps it live until `release`. */
	[[nodiscard]] auto retain(Slot slot, Count entry) -> BlockHandle {
		auto const& blocks = tables_[slot.index()];
		core::invariant(entry < blocks.size);
		auto const block = blocks.entries[entry];
		add(block, Holder::Reference);
		return block;
	}

	/** Gives back one reference taken with `retain`. */
	auto release(BlockHandle block) -> void {
		remove(block, Holder::Reference);
	}

	/** Saves the slot's deltanet state into blocks of the pool, as a snapshot the caller holds until `drop_snapshot`. */
	template<Device<G> D>
	[[nodiscard]] auto save_snapshot(D& device, Slot slot) -> core::Result<SnapshotHandle, SnapshotRefused> {
		if (snapshots_.live_count() + snapshots_.retired_count() == G.snapshot_capacity) {
			return std::unexpected(SnapshotRefused{.shortage = SnapshotShortage::Capacity});
		}
		if (free_blocks() < G.snapshot_blocks) {
			return std::unexpected(SnapshotRefused{.shortage = SnapshotShortage::Blocks});
		}
		auto record = SnapshotRecord<G>{};
		for (auto part = Count{0}; part < G.snapshot_blocks; ++part) {
			auto const taken = blocks_.insert(SnapshotPart{.part = part});
			core::invariant(taken.has_value());
			record.parts[part] = *taken;
		}
		auto const parts = indices(record.parts);
		auto const snapshot = snapshots_.insert(record);
		core::invariant(snapshot.has_value());
		device.apply(Command<G>{SaveSnapshot<G>{.slot = slot, .blocks = parts}});
		return *snapshot;
	}

	/** Makes the slot's deltanet state the snapshot's; the snapshot stays held. */
	template<Device<G> D>
	auto load_snapshot(D& device, SnapshotHandle snapshot, Slot slot) -> void {
		auto const record = snapshots_.try_get(snapshot);
		core::invariant(record.has_value());
		device.apply(Command<G>{LoadSnapshot<G>{.slot = slot, .blocks = indices(record->parts)}});
	}

	/** Frees a snapshot and its blocks. */
	auto drop_snapshot(SnapshotHandle snapshot) -> void {
		auto const record = snapshots_.try_get(snapshot);
		core::invariant(record.has_value());
		for (auto const part : record->parts) {
			blocks_.release(part);
		}
		snapshots_.release(snapshot);
	}

private:
	struct SlotBlocks {
		std::array<BlockHandle, G.table_blocks> entries;
		Count size;
	};

	// The table encodes a handle as its generation above its index; the index is the block's place on the cards.
	[[nodiscard]] static auto index_of(BlockHandle block) -> BlockIndex {
		return core::narrow<BlockIndex>(block.bits & ((std::uint64_t{1} << Blocks::index_bits) - 1));
	}

	[[nodiscard]] static auto indices(std::array<BlockHandle, G.snapshot_blocks> const& parts) -> SnapshotBlocks {
		auto result = SnapshotBlocks{};
		std::ranges::transform(parts, result.begin(), index_of);
		return result;
	}

	[[nodiscard]] auto kv(BlockHandle block) const -> KvBlock const& {
		auto const use = blocks_.try_get(block);
		core::invariant(use.has_value() && std::holds_alternative<KvBlock>(*use));
		return std::get<KvBlock>(*use);
	}

	[[nodiscard]] auto kv(BlockHandle block) -> KvBlock& {
		auto& use = blocks_.get(block);
		core::invariant(std::holds_alternative<KvBlock>(use));
		return std::get<KvBlock>(use);
	}

	/** The entries of the slot's table that `extend` writes into, before any block is added past its end. */
	[[nodiscard]] auto written(Extend const& extend) const -> std::span<BlockHandle const> {
		auto const& blocks = tables_[extend.slot.index()];
		auto const entries = std::span{blocks.entries}.first(blocks.size);
		auto const first = std::min(extend.length / G.block_tokens, entries.size());
		auto const last = std::min(entries.size(), model::divide_up(extend.until, G.block_tokens));
		return entries.subspan(first, std::max(first, last) - first);
	}

	[[nodiscard]] auto writes(Extend const& extend, BlockHandle block) const -> bool {
		return std::ranges::contains(written(extend), block);
	}

	[[nodiscard]] auto writers(std::span<Extend const> extends, BlockHandle block) const -> Count {
		return core::narrow<Count>(std::ranges::count_if(extends, [&](Extend const& extend) {
			return writes(extend, block);
		}));
	}

	[[nodiscard]] static auto held_only(References holders) -> bool {
		return holders.tables == 0 && holders.held > 0;
	}

	[[nodiscard]] auto allocate() -> BlockHandle {
		auto const block = blocks_.insert(KvBlock{.references = {.tables = 1, .held = 0}});
		core::invariant(block.has_value());
		return *block;
	}

	template<Device<G> D>
	auto place(D& device, Slot slot, Count entry, BlockHandle block) -> void {
		tables_[slot.index()].entries[entry] = block;
		device.apply(Command<G>{SetTableEntry{.slot = slot, .entry = entry, .block = index_of(block)}});
	}

	auto add(BlockHandle block, Holder holder) -> void {
		auto& holders = kv(block).references;
		auto const before = held_only(holders);
		switch (holder) {
		case Holder::Table:
			++holders.tables;
			break;
		case Holder::Reference:
			++holders.held;
			break;
		}
		held_only_ = held_only_ - (before ? 1 : 0) + (held_only(holders) ? 1 : 0);
	}

	auto remove(BlockHandle block, Holder holder) -> void {
		auto& holders = kv(block).references;
		auto const before = held_only(holders);
		switch (holder) {
		case Holder::Table:
			core::invariant(holders.tables > 0);
			--holders.tables;
			break;
		case Holder::Reference:
			core::invariant(holders.held > 0);
			--holders.held;
			break;
		}
		held_only_ = held_only_ - (before ? 1 : 0) + (held_only(holders) ? 1 : 0);
		if (holders.tables == 0 && holders.held == 0) {
			blocks_.release(block);
		}
	}

	Blocks blocks_;
	Snapshots snapshots_;
	std::array<SlotBlocks, model::slots> tables_{};
	Count held_only_ = 0;
};

/** The engine's pool, in the model's geometry. */
export using Pool = BasicPool<model_geometry>;

}
