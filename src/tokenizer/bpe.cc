// SPDX-License-Identifier: AGPL-3.0-only
export module engine.tokenizer:bpe;

import std;
import engine.core;
import :added;

namespace engine::tokenizer {

/** One merge rule: the adjacent pair (left, right) becomes `result`; lower ranks merge first. */
struct Merge {
	Id left;
	Id right;
	Id result;
	std::uint32_t rank;

	[[nodiscard]] constexpr auto pair() const -> std::uint64_t {
		return (std::uint64_t{left} << 32U) | right;
	}
};

/** The merge rules sorted by pair, for lookup; built once when the tokenizer is read. */
struct Merges {
	std::vector<Merge> by_pair;

	[[nodiscard]] auto find(Id left, Id right) const -> std::optional<Merge> {
		auto const key = Merge{.left = left, .right = right, .result = 0, .rank = 0}.pair();
		auto const found = std::ranges::lower_bound(by_pair, key, {}, &Merge::pair);
		if (found == by_pair.end() || found->pair() != key) {
			return std::nullopt;
		}
		return *found;
	}
};

/** A symbol of the word being merged, linked to its live neighbours by index. */
struct Symbol {
	Id id;
	std::optional<std::uint32_t> previous;
	std::optional<std::uint32_t> next;
};

/** A candidate merge of the live adjacent symbols `left` and `right`, as they were when it was queued. */
struct Candidate {
	Merge merge;
	std::uint32_t left;
	std::uint32_t right;
};

/** Heap order: the lowest rank first, and among equal ranks the leftmost, as Hugging Face's merge loop orders them. */
[[nodiscard]] constexpr auto later(Candidate const& first, Candidate const& second) -> bool {
	return std::pair{first.merge.rank, first.left} > std::pair{second.merge.rank, second.left};
}

/**
 * The working state of one encode call: the ids produced so far, and the buffers every word reuses, so a long input
 * allocates a bounded number of times rather than once per word.
 */
struct Scratch {
	std::vector<Id> ids;
	std::vector<Symbol> symbols;
	std::vector<Candidate> heap;
};

auto enqueue(Merges const& merges, Scratch& scratch, std::uint32_t left) -> void {
	auto const right = scratch.symbols[left].next;
	if (!right) {
		return;
	}
	if (auto const merge = merges.find(scratch.symbols[left].id, scratch.symbols[*right].id)) {
		scratch.heap.push_back(Candidate{.merge = *merge, .left = left, .right = *right});
		std::ranges::push_heap(scratch.heap, later);
	}
}

/**
 * Applies the merge rules to one word's byte tokens and appends its token ids to `scratch.ids`. A priority queue of candidate pairs
 * makes this O(n log n) in the word's length n: each merge removes a symbol and queues at most two new pairs. A
 * queued pair is stale once either side merged elsewhere; it is recognized because the two symbols are no longer
 * linked to each other or no longer carry the ids it was queued with.
 */
auto merge_word(Merges const& merges, Scratch& scratch, std::span<Id const> bytes) -> void {
	auto& symbols = scratch.symbols;
	symbols.clear();
	scratch.heap.clear();
	auto const count = core::narrow<std::uint32_t>(bytes.size());
	for (auto index = std::uint32_t{0}; index < count; ++index) {
		symbols.push_back(Symbol{
		    .id = bytes[index],
		    .previous = index == 0 ? std::nullopt : std::optional{index - 1},
		    .next = index + 1 == count ? std::nullopt : std::optional{index + 1},
		});
	}
	for (auto index = std::uint32_t{0}; index < count; ++index) {
		enqueue(merges, scratch, index);
	}
	while (!scratch.heap.empty()) {
		std::ranges::pop_heap(scratch.heap, later);
		auto const top = scratch.heap.back();
		scratch.heap.pop_back();
		auto& left = symbols[top.left];
		auto& right = symbols[top.right];
		if (left.next != top.right || right.previous != top.left || left.id != top.merge.left || right.id != top.merge.right) {
			continue;
		}
		left.id = top.merge.result;
		left.next = right.next;
		if (right.next) {
			symbols[*right.next].previous = top.left;
		}
		if (left.previous) {
			enqueue(merges, scratch, *left.previous);
		}
		enqueue(merges, scratch, top.left);
	}
	for (auto at = count == 0 ? std::optional<std::uint32_t>{} : std::optional{std::uint32_t{0}}; at; at = symbols[*at].next) {
		scratch.ids.push_back(symbols[*at].id);
	}
}

}
