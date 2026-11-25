// SPDX-License-Identifier: AGPL-3.0-only
export module engine.kv_harness;

import std;
import engine.core;
import engine.kv;
import engine.kv_fake;
import engine.model;

/**
 * The oracle side of the pool tests. The harness stands in for everything above the pool: slots whose sequences it
 * knows token by token, a prefix cache that holds block references, and saved snapshots. It drives the pool with
 * operations, writes every sequence's kv through the fake device the way the kernels would, and after each operation
 * checks the whole state against its own plain model: every slot's and every cached prefix's positions read back
 * exactly, no write lands in a block anyone else holds, every count equals its holders, every block is in exactly
 * one place, and a refused request changed nothing.
 *
 * Operations keep only the harness's own invariants (a slot is opened before it is used, a cache entry exists before
 * it is evicted); what the pool decides stays reachable. An operation that names something absent is skipped.
 */
namespace engine::kv_harness {

using kv::BlockHandle;
using kv::BlockIndex;
using kv::Count;
using kv::Geometry;
using kv::Slot;
using kv_fake::Cell;
using kv_fake::Content;
using kv_fake::Kv;

export using Token = std::uint32_t;

inline constexpr std::uint64_t empty_chain = 1469598103934665603ULL;

[[nodiscard]] constexpr auto chain_step(std::uint64_t chain, Token token) -> std::uint64_t {
	return (chain ^ (std::uint64_t{token} + 7)) * 1099511628211ULL;
}

[[nodiscard]] auto chain_of(std::span<Token const> tokens) -> std::uint64_t {
	return std::ranges::fold_left(tokens, empty_chain, chain_step);
}

// Draft positions hold kv the slot later rejects: a chain no real sequence of the run reaches.
inline constexpr std::uint64_t draft_mark = 0x9e37'79b9'7f4a'7c15ULL;

export enum class Source : std::uint8_t {
	Fresh,
	/** The cache entry `pick`'s blocks, attached. */
	Cached,
	/** The first `length` positions of slot `pick`, forked. */
	Fork,
};

/** Bind an empty slot to a sequence. */
export struct Open {
	Count slot;
	Source source;
	Count pick;
	Count length;
};

/** Every live slot in the bit mask commits `tokens` tokens and writes `drafts` rejected positions past them. */
export struct Grow {
	std::uint8_t slots;
	Count tokens;
	Count drafts;
	std::uint64_t seed;
};

/** The slot's sequence shrinks to at most `length`, and the pool keeps blocks for `keep` positions past it. */
export struct Truncate {
	Count slot;
	Count length;
	Count keep;
};

/** The cache takes references to the blocks holding the first `length` positions of the slot. */
export struct Publish {
	Count slot;
	Count length;
};

export struct Evict {
	Count entry;
};

export struct Close {
	Count slot;
};

export struct Save {
	Count slot;
};

export struct Load {
	Count snapshot;
	Count slot;
};

export struct Drop {
	Count snapshot;
};

export using Op = std::variant<Open, Grow, Truncate, Publish, Evict, Close, Save, Load, Drop>;

inline constexpr Count op_kinds = std::variant_size_v<Op>;

/** How often a run draws each operation kind, in `Op`'s order; drawn per run, so each run explores its own corner. */
export using Weights = std::array<Count, op_kinds>;

/** The largest commit and draft run one `Grow` draws. */
export struct Scale {
	Count tokens;
	Count drafts;
};

/** What a run reached, so a test can insist that the interesting outcomes happened at all. */
export struct Reached {
	Count grown;
	Count exhausted;
	Count copies;
	Count snapshots;
	Count refused;
	Count loads;
};

template<class... F>
struct Overloaded : F... {
	using F::operator()...;
};

export template<Geometry G>
class Harness {
public:
	Harness() : pool_{std::make_unique<kv::BasicPool<G>>()}, device_{std::make_unique<kv_fake::FakeDevice<G>>()} {}

	/** Applies one operation and checks the whole state; after the first failure, does nothing. */
	auto run(Op const& op) -> void {
		if (failure_) {
			return;
		}
		std::visit([&](auto const& each) {
			apply(each);
		},
		           op);
		check();
	}

	/** Closes every slot and gives back every reference and snapshot, then checks the pool is empty again. */
	auto finish() -> void {
		for (auto slot = Count{0}; slot < model::slots; ++slot) {
			run(Close{.slot = slot});
		}
		while (!cache_.empty() && !failure_) {
			run(Evict{.entry = 0});
		}
		while (!saved_.empty() && !failure_) {
			run(Drop{.snapshot = 0});
		}
		expect(pool_->free_blocks() == G.pool_blocks, "the pool returns to empty");
		expect(pool_->held_only_blocks() == 0 && pool_->snapshots() == 0, "no cached or snapshot block is left");
	}

	[[nodiscard]] auto failure() const -> std::optional<std::string> const& {
		return failure_;
	}

	[[nodiscard]] auto reached() const -> Reached {
		return reached_;
	}

	/** A random operation over the current state, drawn by `weights`. */
	[[nodiscard]] auto draw(std::mt19937_64& rng, Weights const& weights, Scale scale) const -> Op {
		auto const below = [&](Count bound) -> Count {
			return bound == 0 ? 0 : rng() % bound;
		};
		auto pick = below(std::ranges::fold_left(weights, Count{0}, std::plus{}));
		auto kind = Count{0};
		while (pick >= weights[kind]) {
			pick -= weights[kind];
			++kind;
		}
		auto const any_slot = below(model::slots);
		switch (kind) {
		case 0: {
			auto const source = std::array{Source::Fresh, Source::Cached, Source::Fork}[below(3)];
			auto const origin = source == Source::Cached ? below(cache_.size()) : below(model::slots);
			return Open{.slot = any_slot, .source = source, .pick = origin, .length = below(length_of(origin) + 2)};
		}
		case 1: {
			auto mask = std::uint8_t{0};
			for (auto slot = Count{0}; slot < model::slots; ++slot) {
				mask |= rng() % 2 == 0 ? std::uint8_t{0} : core_bit(slot);
			}
			return Grow{.slots = mask, .tokens = 1 + below(scale.tokens), .drafts = below(scale.drafts + 1), .seed = rng()};
		}
		case 2:
			return Truncate{.slot = any_slot, .length = below(length_of(any_slot) + 1), .keep = below(2 * G.block_tokens)};
		case 3:
			return Publish{.slot = any_slot, .length = 1 + below(length_of(any_slot) + 1)};
		case 4:
			return Evict{.entry = below(cache_.size())};
		case 5:
			return Close{.slot = any_slot};
		case 6:
			return Save{.slot = any_slot};
		case 7:
			return Load{.snapshot = below(saved_.size()), .slot = any_slot};
		default:
			return Drop{.snapshot = below(saved_.size())};
		}
	}

private:
	struct Cached {
		std::vector<BlockHandle> blocks;
		std::vector<Token> tokens;
	};

	struct Saved {
		kv::SnapshotHandle handle;
		std::uint64_t state;
	};

	[[nodiscard]] static constexpr auto core_bit(Count slot) -> std::uint8_t {
		return core::narrow<std::uint8_t>(1U << slot);
	}

	[[nodiscard]] auto length_of(Count slot) const -> Count {
		return slot < model::slots && live_[slot] ? live_[slot]->size() : 0;
	}

	auto expect(bool holds, std::string_view what) -> void {
		if (!holds && !failure_) {
			failure_ = std::string{what};
		}
	}

	auto settle(Slot slot) -> void {
		device_->set_state(slot, chain_of(*live_[slot.index()]));
	}

	auto apply(Open const& open) -> void {
		if (open.slot >= model::slots || live_[open.slot]) {
			return;
		}
		auto const slot = Slot::at(open.slot);
		switch (open.source) {
		case Source::Fresh:
			live_[open.slot] = std::vector<Token>{};
			break;
		case Source::Cached:
			if (open.pick >= cache_.size()) {
				return;
			}
			pool_->attach(*device_, slot, cache_[open.pick].blocks);
			live_[open.slot] = cache_[open.pick].tokens;
			break;
		case Source::Fork: {
			if (open.pick >= model::slots || open.pick == open.slot || !live_[open.pick]) {
				return;
			}
			auto const& origin = *live_[open.pick];
			auto const length = std::min(open.length, origin.size());
			pool_->fork(*device_, Slot::at(open.pick), slot, length);
			live_[open.slot] = std::vector<Token>(origin.begin(), origin.begin() + core::narrow<std::ptrdiff_t>(length));
			break;
		}
		}
		settle(slot);
	}

	auto apply(Grow const& grow) -> void {
		auto extends = std::vector<kv::Extend>{};
		for (auto slot = Count{0}; slot < model::slots; ++slot) {
			auto const until = length_of(slot) + grow.tokens + grow.drafts;
			if ((grow.slots & core_bit(slot)) != 0 && live_[slot] && until <= kv::table_positions(G)) {
				extends.push_back({.slot = Slot::at(slot), .length = length_of(slot), .until = until});
			}
		}
		if (extends.empty()) {
			return;
		}
		auto const tables = tables_now();
		auto const needed = pool_->need(extends);
		auto const free = pool_->free_blocks();
		std::ignore = device_->take_log();
		auto const done = pool_->ensure(*device_, extends);
		auto const log = device_->take_log();
		if (!done) {
			reached_.exhausted += 1;
			expect(needed > free && done.error() == kv::Exhausted{.needed = needed, .free = free}, "exhaustion is reported exactly");
			expect(tables == tables_now() && pool_->free_blocks() == free && log.empty(), "a refused ensure changes nothing");
			return;
		}
		reached_.grown += 1;
		expect(needed <= free && pool_->free_blocks() == free - needed, "ensure takes exactly the blocks need() counts");
		auto fills = Count{0};
		for (auto const& command : log) {
			std::visit(Overloaded{
			               [&](kv::CopyPrefix const& copy) {
				               fills += 1;
				               reached_.copies += 1;
				               expect(holders_by_index().contains(copy.source), "copy-on-write copies only a block someone else still holds");
				               expect(copy.positions == kept_by_copy(extends, copy.target), "a copy keeps exactly the positions its writer holds");
			               },
			               [&](kv::ZeroBlock const&) {
				               fills += 1;
			               },
			               [](auto const&) {},
			           },
			           command);
		}
		expect(fills == needed, "every block ensure takes is filled exactly once");
		auto const by_index = holders_by_index();
		auto rng = std::mt19937_64{grow.seed};
		for (auto const& extend : extends) {
			auto& tokens = *live_[extend.slot.index()];
			auto chain = chain_of(tokens);
			for (auto position = extend.length; position < extend.until; ++position) {
				expect_exclusive(by_index, extend.slot, position);
				if (position < extend.length + grow.tokens) {
					tokens.push_back(core::narrow<Token>(rng() % 3));
					chain = chain_step(chain, tokens.back());
					device_->write(extend.slot, position, Cell{Kv{.chain = chain}});
				} else {
					device_->write(extend.slot, position, Cell{Kv{.chain = chain ^ draft_mark}});
				}
			}
			settle(extend.slot);
		}
	}

	auto apply(Truncate const& truncate) -> void {
		if (truncate.slot >= model::slots || !live_[truncate.slot]) {
			return;
		}
		auto& tokens = *live_[truncate.slot];
		tokens.resize(std::min(truncate.length, tokens.size()));
		pool_->trim(Slot::at(truncate.slot), tokens.size() + truncate.keep);
		settle(Slot::at(truncate.slot));
	}

	auto apply(Publish const& publish) -> void {
		if (publish.slot >= model::slots || !live_[publish.slot] || live_[publish.slot]->empty()) {
			return;
		}
		auto const& tokens = *live_[publish.slot];
		auto const length = std::clamp(publish.length, Count{1}, tokens.size());
		auto entry = Cached{.blocks = {}, .tokens = {tokens.begin(), tokens.begin() + core::narrow<std::ptrdiff_t>(length)}};
		for (auto block = Count{0}; block < model::divide_up(length, G.block_tokens); ++block) {
			entry.blocks.push_back(pool_->retain(Slot::at(publish.slot), block));
		}
		cache_.push_back(std::move(entry));
	}

	auto apply(Evict const& evict) -> void {
		if (evict.entry >= cache_.size()) {
			return;
		}
		for (auto const block : cache_[evict.entry].blocks) {
			pool_->release(block);
		}
		cache_.erase(cache_.begin() + core::narrow<std::ptrdiff_t>(evict.entry));
	}

	auto apply(Close const& close) -> void {
		if (close.slot >= model::slots || !live_[close.slot]) {
			return;
		}
		pool_->trim(Slot::at(close.slot), 0);
		live_[close.slot].reset();
	}

	auto apply(Save const& save) -> void {
		if (save.slot >= model::slots || !live_[save.slot]) {
			return;
		}
		auto const slot = Slot::at(save.slot);
		auto const expected = saved_.size() == G.snapshot_capacity  ? std::optional{kv::SnapshotShortage::Capacity}
		                      : model_free() < G.snapshot_blocks ? std::optional{kv::SnapshotShortage::Blocks}
		                                                            : std::nullopt;
		std::ignore = device_->take_log();
		auto const taken = pool_->save_snapshot(*device_, slot);
		auto const log = device_->take_log();
		if (!taken) {
			reached_.refused += 1;
			expect(expected == taken.error().shortage && log.empty(), "a snapshot is refused exactly when its cap or the pool is full");
			return;
		}
		reached_.snapshots += 1;
		expect(!expected.has_value(), "a snapshot is taken whenever the cap and the pool allow");
		saved_.push_back({.handle = *taken, .state = device_->state(slot)});
	}

	auto apply(Load const& load) -> void {
		if (load.snapshot >= saved_.size() || load.slot >= model::slots) {
			return;
		}
		auto const slot = Slot::at(load.slot);
		reached_.loads += 1;
		pool_->load_snapshot(*device_, saved_[load.snapshot].handle, slot);
		expect(device_->state(slot) == saved_[load.snapshot].state, "a loaded snapshot restores the state it saved");
		if (live_[load.slot]) {
			settle(slot);
		}
	}

	auto apply(Drop const& drop) -> void {
		if (drop.snapshot >= saved_.size()) {
			return;
		}
		pool_->drop_snapshot(saved_[drop.snapshot].handle);
		saved_.erase(saved_.begin() + core::narrow<std::ptrdiff_t>(drop.snapshot));
	}

	[[nodiscard]] auto table_of(Slot slot) const -> std::vector<BlockHandle> {
		auto table = std::vector<BlockHandle>{};
		for (auto entry = Count{0}; entry < pool_->table_size(slot); ++entry) {
			table.push_back(pool_->entry(slot, entry));
		}
		return table;
	}

	[[nodiscard]] auto tables_now() const -> std::vector<std::vector<BlockHandle>> {
		auto tables = std::vector<std::vector<BlockHandle>>{};
		for (auto slot = Count{0}; slot < model::slots; ++slot) {
			auto const table = table_of(Slot::at(slot));
			tables.emplace_back(table.begin(), table.end());
		}
		return tables;
	}

	/** Every kv block a slot table or the cache holds, with its holders as the harness counts them. */
	[[nodiscard]] auto holders() const -> std::map<std::uint64_t, kv::References> {
		auto counts = std::map<std::uint64_t, kv::References>{};
		for (auto slot = Count{0}; slot < model::slots; ++slot) {
			for (auto const block : table_of(Slot::at(slot))) {
				counts[block.bits].tables += 1;
			}
		}
		for (auto const& entry : cache_) {
			for (auto const block : entry.blocks) {
				counts[block.bits].held += 1;
			}
		}
		return counts;
	}

	/** The positions the slot that now holds `target` keeps in it: those below its length. */
	[[nodiscard]] auto kept_by_copy(std::span<kv::Extend const> extends, BlockIndex target) const -> std::optional<Count> {
		for (auto const& extend : extends) {
			auto const table = table_of(extend.slot);
			for (auto const [entry, block] : std::views::enumerate(table)) {
				if (pool_->index(block) == target) {
					auto const first = core::narrow<Count>(entry) * G.block_tokens;
					return extend.length - std::min(extend.length, first);
				}
			}
		}
		return std::nullopt;
	}

	/** The holders of every held kv block, by device index. */
	[[nodiscard]] auto holders_by_index() const -> std::map<BlockIndex, Count> {
		auto counts = std::map<BlockIndex, Count>{};
		for (auto const& [bits, references] : holders()) {
			counts[pool_->index(BlockHandle{bits})] += references.tables + references.held;
		}
		return counts;
	}

	[[nodiscard]] auto model_free() const -> Count {
		return G.pool_blocks - holders().size() - saved_.size() * G.snapshot_blocks;
	}

	/** The block a kernel writing this position would reach is the slot's own, and no other holder's. */
	auto expect_exclusive(std::map<BlockIndex, Count> const& by_index, Slot slot, Count position) -> void {
		auto const entry = position / G.block_tokens;
		if (entry >= pool_->table_size(slot)) {
			expect(false, "a slot writes a position its table does not cover");
			return;
		}
		auto const index = pool_->index(pool_->entry(slot, entry));
		expect(device_->entry(slot, entry) == index, "the device table names the block the pool assigned");
		auto const found = by_index.find(index);
		expect(found != by_index.end() && found->second == 1, "a slot never writes into a block another holder keeps");
	}

	[[nodiscard]] auto reads_back(BlockIndex index, Count offset, std::uint64_t chain) const -> bool {
		auto const& content = device_->content(index);
		return std::holds_alternative<std::vector<Cell>>(content) &&
		       std::get<std::vector<Cell>>(content)[offset] == Cell{Kv{.chain = chain}};
	}

	auto check() -> void {
		expect(device_->violations().empty(), "the device sees only well-formed commands");
		auto const counts = holders();
		auto indices = std::set<BlockIndex>{};
		auto held_only = Count{0};
		for (auto const& [bits, expected] : counts) {
			auto const block = BlockHandle{bits};
			auto const references = pool_->references(block);
			expect(references.has_value(), "no table or reference names a freed block");
			if (!references) {
				return;
			}
			expect(*references == expected, "every count equals its holders");
			held_only += expected.tables == 0 ? 1 : 0;
			auto const index = pool_->index(block);
			expect(index < G.pool_blocks && indices.insert(index).second, "every live block has its own index");
		}
		for (auto const& entry : saved_) {
			auto const blocks = pool_->snapshot_blocks(entry.handle);
			expect(blocks.has_value(), "a held snapshot stays live");
			if (!blocks) {
				return;
			}
			for (auto const [part, index] : std::views::enumerate(*blocks)) {
				expect(index < G.pool_blocks && indices.insert(index).second, "a snapshot block is in no other place");
				expect(device_->content(index) == Content{kv_fake::Part{.state = entry.state, .part = core::narrow<Count>(part)}},
				       "a snapshot's blocks are never overwritten");
			}
		}
		expect(pool_->free_blocks() + indices.size() == G.pool_blocks, "blocks are conserved: free plus held is the pool");
		expect(pool_->held_only_blocks() == held_only, "the cache-only count equals the blocks only references keep");
		expect(pool_->snapshots() == saved_.size(), "the snapshot count equals the snapshots held");
		for (auto slot = Count{0}; slot < model::slots; ++slot) {
			auto const table = table_of(Slot::at(slot));
			if (!live_[slot]) {
				expect(table.empty(), "a closed slot holds no blocks");
				continue;
			}
			auto distinct = std::set<std::uint64_t>{};
			for (auto const [entry, block] : std::views::enumerate(table)) {
				auto const index = pool_->index(block);
				auto const& content = device_->content(index);
				expect(distinct.insert(block.bits).second, "a table names each block once");
				expect(device_->entry(Slot::at(slot), core::narrow<Count>(entry)) == index, "the device table follows the pool's");
				expect(std::holds_alternative<std::vector<Cell>>(content) &&
				           std::ranges::none_of(std::get<std::vector<Cell>>(content),
				                                [](Cell const& cell) {
					                                return std::holds_alternative<kv_fake::Garbage>(cell);
				                                }),
				       "every block in a table holds only kv and zeros, never garbage");
			}
			auto const& tokens = *live_[slot];
			expect(table.size() * G.block_tokens >= tokens.size(), "a slot's table covers its sequence");
			auto chain = empty_chain;
			for (auto position = Count{0}; position < tokens.size() && position < table.size() * G.block_tokens; ++position) {
				chain = chain_step(chain, tokens[position]);
				expect(reads_back(pool_->index(table[position / G.block_tokens]), position % G.block_tokens, chain),
				       "every slot's sequence reads back exactly");
			}
		}
		for (auto const& entry : cache_) {
			auto chain = empty_chain;
			for (auto position = Count{0}; position < entry.tokens.size(); ++position) {
				chain = chain_step(chain, entry.tokens[position]);
				expect(reads_back(pool_->index(entry.blocks[position / G.block_tokens]), position % G.block_tokens, chain),
				       "a cached prefix is never mutated");
			}
		}
	}

	std::unique_ptr<kv::BasicPool<G>> pool_;
	std::unique_ptr<kv_fake::FakeDevice<G>> device_;
	std::array<std::optional<std::vector<Token>>, model::slots> live_{};
	std::vector<Cached> cache_;
	std::vector<Saved> saved_;
	std::optional<std::string> failure_;
	Reached reached_{};
};

}
