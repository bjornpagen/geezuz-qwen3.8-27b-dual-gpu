// SPDX-License-Identifier: AGPL-3.0-only
export module engine.prefix:cache;

import std;
import engine.core;
import engine.kv;
import engine.model;
import :pending;
import :point;
import :tree;

namespace engine::prefix {

/**
 * A cached prefix a prompt can resume from: positions [0, length) are in `blocks` and the deltanet state after them is
 * `snapshot`. `length` is a snapshot point, never just where the match ends, because the state cannot be cut back.
 * Valid until the cache next evicts; `resume` takes it at once.
 */
export struct Hit {
	Count length;
	std::vector<kv::BlockHandle> blocks;
	kv::SnapshotHandle snapshot;
	Mark mark;
	NodeHandle node;
};

/**
 * What the cache offers a prompt: the deepest cached prefix it can resume from, and the deepest point beyond that
 * another slot's prefill will cache soon. Both are absent for a prompt nothing shares.
 */
export struct Match {
	std::optional<Hit> hit;
	std::optional<Wait> wait;
};

export enum class Insertion : std::uint8_t {
	/** The slot's state was saved as a new snapshot. */
	Saved,
	/** The cache already held a snapshot at that sequence; the slot now uses it. */
	Present,
};

/** What an eviction gave back. */
export struct Evicted {
	/** Blocks that returned to the pool's free list, kv and snapshot. */
	Count blocks;
	/** Cached snapshot points dropped. */
	Count snapshots;

	[[nodiscard]] constexpr auto operator==(Evicted const&) const -> bool = default;
};

/** One cached snapshot point, as the cache holds it. */
export struct Entry {
	std::vector<Token> tokens;
	std::vector<kv::BlockHandle> blocks;
	kv::SnapshotHandle snapshot;
	Mark mark;
	/** Live slots using it; zero means it may be evicted once it is a leaf. */
	Count pins;
};

/**
 * The prefix cache: a radix tree over token blocks whose nodes hold counted references to their kv blocks, and
 * deltanet snapshots at the chat format's points. It owns no blocks: the scheduler owns the pool and the device and
 * lends them to each call. A slot uses one cached point at a time, the one it resumed from or saved last; a point in
 * use is pinned and outside the eviction order altogether. Eviction takes resting leaves, oldest first, so a point at a
 * block's end goes only after every point that runs through that block; a point inside a block holds that block alone
 * (whatever continues past it runs through a copy) and is a leaf from the start.
 *
 * Large (a node table sized by the pool): the owner keeps it behind a `std::unique_ptr`.
 */
export template<kv::Geometry G>
class BasicCache {
public:
	BasicCache() = default;
	BasicCache(BasicCache const&) = delete;
	auto operator=(BasicCache const&) -> BasicCache& = delete;
	BasicCache(BasicCache&&) noexcept = default;
	auto operator=(BasicCache&&) noexcept -> BasicCache& = default;
	~BasicCache() = default;

	/**
	 * The deepest cached snapshot point whose prefix `prompt` starts with, short of its last token, which is always
	 * read because its logits pick the first generated token; and the deepest point beyond it that an announced prefill
	 * of another slot shares with `prompt`.
	 */
	[[nodiscard]] auto lookup(std::span<Token const> prompt) const -> Match {
		core::invariant(!prompt.empty());
		auto const limit = prompt.size() - 1;
		auto hit = tree_.deepest(prompt, limit).transform([&](NodeHandle node) {
			auto const& found = tree_.get(node);
			return Hit{
			    .length = found.end(),
			    .blocks = tree_.path(node),
			    .snapshot = found.snapshot->handle,
			    .mark = found.snapshot->mark,
			    .node = node,
			};
		});
		auto const above = hit ? hit->length : 0;
		return {.hit = std::move(hit), .wait = prefills_.deepest(prompt, above, limit)};
	}

	/** Gives the empty slot the hit's blocks and deltanet state, and pins the hit for it. */
	template<kv::Device<G> D>
	auto resume(D& device, kv::BasicPool<G>& pool, kv::Slot slot, Hit const& hit) -> void {
		core::invariant(tree_.live(hit.node) && !pins_[slot.index()]);
		core::invariant(tree_.get(hit.node).end() == hit.length && tree_.get(hit.node).snapshot->handle == hit.snapshot);
		pool.attach(device, slot, hit.blocks);
		pool.load_snapshot(device, hit.snapshot, slot);
		pin(slot, hit.node);
	}

	/**
	 * The slot prefills `prompt` from `from`: its points past `from` become pending, so a prompt sharing one can wait
	 * for it. A slot announces at most one prompt; announcing again replaces it.
	 */
	auto announce(kv::Slot slot, std::span<Token const> prompt, Points const& points, Count from) -> void {
		prefills_.announce(slot, prompt, points, from);
	}

	/**
	 * Caches `sequence` at its end: the slot's table holds its kv and its deltanet state is the state after it. Blocks
	 * the tree lacks are retained from the slot's table, and the state is saved as a snapshot unless one is there
	 * already. When the pool refuses the snapshot, resting leaves are evicted, oldest first, until it fits; refused only
	 * when none is left, and then the slot keeps using the point it used before. Saved or present, the slot uses this
	 * point from now on. Either way its pending points up to here are reached.
	 */
	template<kv::Device<G> D>
	[[nodiscard]] auto insert(D& device, kv::BasicPool<G>& pool, kv::Slot slot, std::span<Token const> sequence, Mark mark)
	    -> core::Result<Insertion, kv::SnapshotRefused> {
		core::invariant(!sequence.empty() && model::divide_up(sequence.size(), G.block_tokens) <= pool.table_size(slot));
		prefills_.reached(slot, sequence.size());
		if (auto const present = tree_.find(sequence); present && tree_.get(*present).snapshot) {
			pin(slot, *present);
			return Insertion::Present;
		}
		auto saved = pool.save_snapshot(device, slot);
		while (!saved) {
			if (!evict_one(pool)) {
				return std::unexpected(saved.error());
			}
			saved = pool.save_snapshot(device, slot);
		}
		// Built after any eviction, so it is built against the tree as it now stands.
		auto node = std::optional<NodeHandle>{};
		for (auto start = Count{0}; start < sequence.size(); start += G.block_tokens) {
			auto const piece = sequence.subspan(start, std::min(G.block_tokens, sequence.size() - start));
			auto const found = tree_.child(node, piece);
			node = found ? *found : tree_.add(node, piece, pool.retain(slot, start / G.block_tokens));
		}
		auto& snapshot = tree_.get(*node).snapshot;
		core::invariant(!snapshot.has_value());
		snapshot = Snapshot{.handle = *saved, .mark = mark, .use = Pinned{.slots = 0}};
		pin(slot, *node);
		return Insertion::Saved;
	}

	/**
	 * The slot's request ends, finished or preempted, and `sequence` is everything its state has read: it is cached at
	 * its end, so the conversation's next turn, which starts with this prompt and reply, resumes there. Then the slot
	 * leaves as in `abandon`. A slot that has read nothing has nothing to keep and leaves through `abandon` alone.
	 */
	template<kv::Device<G> D>
	[[nodiscard]] auto close(D& device, kv::BasicPool<G>& pool, kv::Slot slot, std::span<Token const> sequence)
	    -> core::Result<Insertion, kv::SnapshotRefused> {
		auto const kept = insert(device, pool, slot, sequence, SequenceEnd{});
		abandon(pool, slot);
		return kept;
	}

	/** The slot leaves without caching anything more: its point rests, its prefill is withdrawn, its table emptied. */
	auto abandon(kv::BasicPool<G>& pool, kv::Slot slot) -> void {
		prefills_.withdraw(slot);
		if (auto const pinned = std::exchange(pins_[slot.index()], std::nullopt)) {
			unpin(*pinned);
		}
		pool.trim(slot, 0);
	}

	/**
	 * Evicts resting leaves, oldest first, until the pool has `free` free blocks or nothing evictable is left. A pinned
	 * point and every prefix of one stay; a block a slot's table holds stays live whatever the cache drops.
	 */
	[[nodiscard]] auto make_room(kv::BasicPool<G>& pool, Count free) -> Evicted {
		auto const before = pool.free_blocks();
		auto dropped = Count{0};
		while (pool.free_blocks() < free && evict_one(pool)) {
			dropped += 1;
		}
		return {.blocks = pool.free_blocks() - before, .snapshots = dropped};
	}

	/** Whether the point a lookup offered to wait for is still to come. */
	[[nodiscard]] auto pending(Wait wait) const -> bool {
		return prefills_.pending(wait);
	}

	/** Every cached snapshot point. */
	[[nodiscard]] auto entries() const -> std::vector<Entry> {
		auto entries = std::vector<Entry>{};
		for (auto const handle : tree_.handles()) {
			auto const& node = tree_.get(handle);
			if (!node.snapshot) {
				continue;
			}
			auto const pins = core::alternative<Pinned>(node.snapshot->use)
			                      .transform([](Pinned const& pinned) {
				                      return pinned.slots;
			                      })
			                      .value_or(0);
			entries.push_back({
			    .tokens = tree_.sequence(handle),
			    .blocks = tree_.path(handle),
			    .snapshot = node.snapshot->handle,
			    .mark = node.snapshot->mark,
			    .pins = pins,
			});
		}
		return entries;
	}

	/** Blocks of the tree: each holds one reference to its block. */
	[[nodiscard]] auto nodes() const -> Count {
		return tree_.size();
	}

private:
	/** A resting point's place in the eviction order: its stamp, then its node. */
	using Recency = std::pair<std::uint64_t, std::uint64_t>;

	auto pin(kv::Slot slot, NodeHandle node) -> void {
		auto const previous = pins_[slot.index()];
		if (previous == node) {
			return;
		}
		auto& use = tree_.get(node).snapshot->use;
		if (auto const resting = core::alternative<Resting>(use)) {
			resting_.erase({resting->stamp, node.bits});
			use = Pinned{.slots = 1};
		} else {
			std::get<Pinned>(use).slots += 1;
		}
		pins_[slot.index()] = node;
		if (previous) {
			unpin(*previous);
		}
	}

	auto unpin(NodeHandle node) -> void {
		auto& use = tree_.get(node).snapshot->use;
		core::invariant(std::holds_alternative<Pinned>(use));
		auto& pinned = std::get<Pinned>(use);
		pinned.slots -= 1;
		if (pinned.slots == 0) {
			clock_ += 1;
			use = Resting{.stamp = clock_};
			resting_.insert({clock_, node.bits});
		}
	}

	/** The oldest resting point that is a leaf. */
	[[nodiscard]] auto victim() const -> std::optional<NodeHandle> {
		auto const found = std::ranges::find_if(resting_, [&](Recency const& recency) {
			return tree_.get(NodeHandle{recency.second}).children.empty();
		});
		return found == resting_.end() ? std::nullopt : std::optional{NodeHandle{found->second}};
	}

	/** Drops the victim's snapshot and every block of its branch nothing else leads through. */
	[[nodiscard]] auto evict_one(kv::BasicPool<G>& pool) -> bool {
		auto const leaf = victim();
		if (!leaf) {
			return false;
		}
		auto& snapshot = tree_.get(*leaf).snapshot;
		resting_.erase({std::get<Resting>(snapshot->use).stamp, leaf->bits});
		pool.drop_snapshot(snapshot->handle);
		snapshot.reset();
		auto bare = std::optional{*leaf};
		while (bare && !tree_.get(*bare).snapshot && tree_.get(*bare).children.empty()) {
			auto const parent = tree_.get(*bare).parent;
			pool.release(tree_.remove(*bare));
			bare = parent;
		}
		return true;
	}

	Tree<G> tree_;
	Prefills prefills_;
	std::array<std::optional<NodeHandle>, model::slots> pins_{};
	std::set<Recency> resting_;
	std::uint64_t clock_ = 0;
};

/** The engine's prefix cache, in the model's pool geometry. */
export using Cache = BasicCache<kv::model_geometry>;

}
