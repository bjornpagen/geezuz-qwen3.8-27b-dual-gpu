// SPDX-License-Identifier: AGPL-3.0-only
export module engine.prefix:tree;

import std;
import engine.core;
import engine.handle;
import engine.kv;
import :point;

namespace engine::prefix {

/** A generational reference to a node of the prefix tree. */
export struct NodeHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator==(NodeHandle const&) const -> bool = default;
};

/** No live slot uses the snapshot; `stamp` orders it on the cache's clock, oldest first out. */
struct Resting {
	std::uint64_t stamp;
};

/** `slots` live slots resumed from the snapshot or saved it last; it is not a candidate for eviction. */
struct Pinned {
	Count slots;
};

using Use = std::variant<Resting, Pinned>;

/** The deltanet state after the node's last position, held in the pool. */
struct Snapshot {
	kv::SnapshotHandle handle;
	Mark mark;
	Use use;
};

/**
 * One block of a cached sequence: positions [start, start + size) and the pool block holding their kv, kept live by
 * one `retain` reference. A full node (size == block_tokens) may have children; a partial one is a leaf ending at a
 * snapshot point inside its block. A snapshot sits at the node's end, so a node carries at most one.
 */
template<kv::Geometry G>
struct Node {
	/** Absent for a child of the root, which holds no block. */
	std::optional<NodeHandle> parent;
	Count start;
	std::array<Token, G.block_tokens> tokens;
	Count size;
	kv::BlockHandle block;
	std::vector<NodeHandle> children;
	std::optional<Snapshot> snapshot;

	[[nodiscard]] auto sequence() const -> std::span<Token const> {
		return std::span{tokens}.first(size);
	}

	[[nodiscard]] auto end() const -> Count {
		return start + size;
	}

	[[nodiscard]] auto full() const -> bool {
		return size == G.block_tokens;
	}
};

// Enough that no node record exhausts its generations in any lifetime of the server (2^48 reuses).
inline constexpr std::uint8_t generation_bits = 48;

/**
 * The radix tree over token blocks. It holds structure only; the cache decides what enters and leaves and keeps the
 * pool's references in step. Every node leads to a snapshot: a node with neither a snapshot nor children holds kv no
 * slot can resume into, and the cache removes it in the same operation that left it bare.
 */
template<kv::Geometry G>
class Tree {
	// Full nodes hold distinct blocks (equal content under equal parents is one node), and every partial node carries
	// one of the pool's snapshots.
	static constexpr Count capacity = G.pool_blocks + G.snapshot_capacity;

	using Nodes = handle::Table<Node<G>, NodeHandle, capacity, generation_bits>;

public:
	[[nodiscard]] auto get(NodeHandle node) const -> Node<G> const& {
		auto const found = nodes_.try_get(node);
		core::invariant(found.has_value());
		return *found;
	}

	[[nodiscard]] auto get(NodeHandle node) -> Node<G>& {
		return nodes_.get(node);
	}

	[[nodiscard]] auto live(NodeHandle node) const -> bool {
		return nodes_.try_get(node).has_value();
	}

	[[nodiscard]] auto size() const -> Count {
		return nodes_.live_count();
	}

	/** Every node, in table order; valid while nothing is added or removed. */
	[[nodiscard]] auto handles() const {
		return nodes_.handles();
	}

	[[nodiscard]] auto children(std::optional<NodeHandle> parent) const -> std::span<NodeHandle const> {
		return parent ? std::span<NodeHandle const>{get(*parent).children} : std::span<NodeHandle const>{roots_};
	}

	/** The child of `parent` holding exactly `piece`. */
	[[nodiscard]] auto child(std::optional<NodeHandle> parent, std::span<Token const> piece) const -> std::optional<NodeHandle> {
		auto const found = std::ranges::find_if(children(parent), [&](NodeHandle child) {
			return std::ranges::equal(get(child).sequence(), piece);
		});
		return found == children(parent).end() ? std::nullopt : std::optional{*found};
	}

	/** The node whose sequence is exactly `sequence`. */
	[[nodiscard]] auto find(std::span<Token const> sequence) const -> std::optional<NodeHandle> {
		auto node = std::optional<NodeHandle>{};
		for (auto start = Count{0}; start < sequence.size(); start += G.block_tokens) {
			node = child(node, sequence.subspan(start, std::min(G.block_tokens, sequence.size() - start)));
			if (!node) {
				return std::nullopt;
			}
		}
		return node;
	}

	/**
	 * The deepest node carrying a snapshot whose whole sequence is a prefix of `prompt` and ends at or before `limit`.
	 * Every child of each matched full node is weighed, so a snapshot on a branch that ends before the prompt diverges
	 * from it counts too.
	 */
	[[nodiscard]] auto deepest(std::span<Token const> prompt, Count limit) const -> std::optional<NodeHandle> {
		auto best = std::optional<NodeHandle>{};
		auto best_end = Count{0};
		auto parent = std::optional<NodeHandle>{};
		for (auto start = Count{0}; start < limit; start += G.block_tokens) {
			auto next = std::optional<NodeHandle>{};
			for (auto const handle : children(parent)) {
				auto const& node = get(handle);
				if (node.end() > prompt.size() || !std::ranges::equal(node.sequence(), prompt.subspan(start, node.size))) {
					continue;
				}
				if (node.snapshot && node.end() <= limit && node.end() > best_end) {
					best = handle;
					best_end = node.end();
				}
				if (node.full()) {
					next = handle;
				}
			}
			if (!next) {
				break;
			}
			parent = next;
		}
		return best;
	}

	/** Adds the node holding `piece` under `parent`, its kv in `block`, which the caller has retained for it. */
	[[nodiscard]] auto add(std::optional<NodeHandle> parent, std::span<Token const> piece, kv::BlockHandle block) -> NodeHandle {
		core::invariant(!piece.empty() && piece.size() <= G.block_tokens && !child(parent, piece));
		core::invariant(!parent || get(*parent).full());
		auto node = Node<G>{
		    .parent = parent,
		    .start = parent ? get(*parent).end() : 0,
		    .tokens = {},
		    .size = piece.size(),
		    .block = block,
		    .children = {},
		    .snapshot = std::nullopt,
		};
		std::ranges::copy(piece, node.tokens.begin());
		auto const handle = nodes_.insert(std::move(node));
		core::invariant(handle.has_value());
		children_of(parent).push_back(*handle);
		return *handle;
	}

	/** Removes a node with neither children nor a snapshot; returns its block, whose reference the caller gives back. */
	[[nodiscard]] auto remove(NodeHandle handle) -> kv::BlockHandle {
		auto const& node = get(handle);
		core::invariant(node.children.empty() && !node.snapshot);
		auto const block = node.block;
		auto const parent = node.parent;
		auto const erased = std::erase(children_of(parent), handle);
		core::invariant(erased == 1);
		nodes_.release(handle);
		return block;
	}

	/** The blocks from the root to `node`, in position order. */
	[[nodiscard]] auto path(NodeHandle node) const -> std::vector<kv::BlockHandle> {
		auto blocks = std::vector<kv::BlockHandle>{};
		for (auto at = std::optional{node}; at; at = get(*at).parent) {
			blocks.push_back(get(*at).block);
		}
		std::ranges::reverse(blocks);
		return blocks;
	}

	/** The tokens from the root through `node`. */
	[[nodiscard]] auto sequence(NodeHandle node) const -> std::vector<Token> {
		auto tokens = std::vector<Token>(get(node).end());
		for (auto at = std::optional{node}; at; at = get(*at).parent) {
			auto const& each = get(*at);
			std::ranges::copy(each.sequence(), tokens.begin() + core::narrow<std::ptrdiff_t>(each.start));
		}
		return tokens;
	}

private:
	[[nodiscard]] auto children_of(std::optional<NodeHandle> parent) -> std::vector<NodeHandle>& {
		return parent ? get(*parent).children : roots_;
	}

	Nodes nodes_;
	std::vector<NodeHandle> roots_;
};

}
