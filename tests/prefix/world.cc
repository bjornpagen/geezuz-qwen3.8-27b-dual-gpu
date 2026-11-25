// SPDX-License-Identifier: AGPL-3.0-only
export module engine.prefix_world;

import std;
import engine.core;
import engine.kv;
import engine.kv_fake;
import engine.model;
import engine.prefix;

/**
 * The oracle side of the prefix cache tests. A world stands in for the scheduler and the prefill: slots that read
 * prompts and generate tokens, writing every position's kv through the content-checking fake device the way the
 * kernels would, and naming each slot's deltanet state by the tokens it has read. Beside the real cache it keeps a plain
 * reference: the list of cached sequences, with the slots using each and when each was last left. After every
 * operation it checks the cache against that list (the same sequences cached, every lookup the deepest cached prefix,
 * every eviction the oldest resting leaf of the tree) and against the device: every cached prefix and snapshot
 * reads back exactly, every count equals its holders, every live slot reads back its own sequence.
 */
namespace engine::prefix_world {

using kv::BlockHandle;
using kv::Count;
using kv::Geometry;
using kv::Slot;
using kv_fake::Cell;
using kv_fake::Content;
using kv_fake::Kv;
using prefix::Token;

inline constexpr std::uint64_t empty_chain = 1469598103934665603ULL;

[[nodiscard]] constexpr auto chain_step(std::uint64_t chain, Token token) -> std::uint64_t {
	return (chain ^ (std::uint64_t{token} + 7)) * 1099511628211ULL;
}

/** The name of the kv at the last position, and of the deltanet state, after reading `tokens`. */
export [[nodiscard]] auto chain_of(std::span<Token const> tokens) -> std::uint64_t {
	return std::ranges::fold_left(tokens, empty_chain, chain_step);
}

// Draft positions hold kv the slot later rejects: a chain no real sequence reaches.
inline constexpr std::uint64_t draft_mark = 0x9e37'79b9'7f4a'7c15ULL;

/** Where a slot is in its request. */
export enum class Phase : std::uint8_t {
	Free,
	Reading,
	Generating,
};

/** Whether the slot stays bound after an insertion, or leaves in the same call. */
enum class Afterwards : std::uint8_t {
	Stays,
	Leaves,
};

export enum class Progress : std::uint8_t {
	/** Stopped at a snapshot point or the chunk's end, with prompt left to read. */
	Advanced,
	/** The whole prompt is read. */
	Read,
	/** The pool could not hold the chunk even after the cache gave back what it could. */
	Stalled,
};

/** What a run reached, so a test can insist that the interesting outcomes happened at all. */
export struct Reached {
	Count hits;
	Count tail_hits;
	Count waits;
	Count saved;
	Count present;
	Count refused;
	Count evicted;
	Count stalls;
};

[[nodiscard]] auto starts_with(std::span<Token const> sequence, std::span<Token const> prefix) -> bool {
	return prefix.size() <= sequence.size() && std::ranges::equal(sequence.first(prefix.size()), prefix);
}

export template<Geometry G>
class World {
public:
	World()
	    : pool_{std::make_unique<kv::BasicPool<G>>()}, device_{std::make_unique<kv_fake::FakeDevice<G>>()},
	      cache_{std::make_unique<prefix::BasicCache<G>>()} {}

	/** The cache's answer for `prompt`, checked against the reference. */
	[[nodiscard]] auto peek(std::span<Token const> prompt) -> prefix::Match {
		auto match = cache_->lookup(prompt);
		auto const expected = deepest_cached(prompt);
		expect(match.hit.has_value() == expected.has_value(), "a lookup hits exactly when a cached sequence prefixes the prompt");
		if (match.hit && expected) {
			auto const& hit = *match.hit;
			auto const& entry = cached_[*expected];
			reached_.hits += 1;
			reached_.tail_hits += hit.length % G.block_tokens == 0 ? 0 : 1;
			expect(hit.length == entry.tokens.size(), "a lookup resumes at the deepest cached prefix short of the last token");
			expect(hit.mark == entry.mark, "a hit carries its point's mark");
			expect(hit.blocks.size() == model::divide_up(hit.length, G.block_tokens), "a hit's blocks cover its length");
			auto chain = empty_chain;
			for (auto position = Count{0}; position < hit.length && hit.blocks.size() * G.block_tokens >= hit.length; ++position) {
				chain = chain_step(chain, prompt[position]);
				expect(reads_back(pool_->index(hit.blocks[position / G.block_tokens]), position % G.block_tokens, chain),
				       "a hit's blocks read back the prompt's prefix");
			}
			expect(snapshot_holds(hit.snapshot, chain_of(prompt.first(hit.length))), "a hit's snapshot is the state after its prefix");
		}
		auto const wait = pending_point(prompt, match.hit ? match.hit->length : 0);
		expect(match.wait.transform([](prefix::Wait const& offered) { return offered.position; }) == wait, "a lookup offers the deepest shared point a prefill will cache");
		if (match.wait) {
			reached_.waits += 1;
			expect(cache_->pending(*match.wait), "an offered wait is pending");
		}
		check();
		return match;
	}

	/** Binds the free slot to `prompt`, resuming from the cache's hit; returns where its prefill starts. */
	[[nodiscard]] auto open(Count index, std::vector<Token> prompt, std::span<prefix::Point const> marked) -> Count {
		core::invariant(!live_[index] && !prompt.empty());
		auto const slot = Slot::at(index);
		auto const match = peek(prompt);
		auto const points = prefix::Points::of(prompt.size(), marked);
		auto sequence = std::vector<Token>{};
		if (match.hit) {
			cache_->resume(*device_, *pool_, slot, *match.hit);
			sequence.assign(prompt.begin(), prompt.begin() + core::narrow<std::ptrdiff_t>(match.hit->length));
			reference_pin(slot, sequence);
		} else {
			device_->set_state(slot, empty_chain);
		}
		auto const from = sequence.size();
		cache_->announce(slot, prompt, points, from);
		auto const ahead = points.after(from);
		live_[index] = Live{.sequence = std::move(sequence), .prompt = std::move(prompt), .ahead = {ahead.begin(), ahead.end()}};
		check();
		return from;
	}

	/** Reads up to `chunk` more prompt tokens, stopping at the next snapshot point and caching the state there. */
	[[nodiscard]] auto prefill(Count index, Count chunk) -> Progress {
		core::invariant(phase(index) == Phase::Reading && chunk > 0);
		auto& slot = *live_[index];
		auto const fed = slot.sequence.size();
		auto const point = slot.ahead.empty() ? slot.prompt.size() : slot.ahead.front().position;
		auto const stop = std::min(fed + chunk, point);
		auto const read = std::vector<Token>{slot.prompt.begin() + core::narrow<std::ptrdiff_t>(fed),
		                                     slot.prompt.begin() + core::narrow<std::ptrdiff_t>(stop)};
		if (!grow(index, read, 0)) {
			return Progress::Stalled;
		}
		auto& grown = *live_[index];
		if (!grown.ahead.empty() && grown.ahead.front().position == stop) {
			auto const kind = grown.ahead.front().kind;
			insert(index, kind, Afterwards::Stays, [&] {
				return cache_->insert(*device_, *pool_, Slot::at(index), grown.sequence, prefix::Mark{kind});
			});
		}
		check();
		return stop == grown.prompt.size() ? Progress::Read : Progress::Advanced;
	}

	/** Commits `count` generated tokens and writes `drafts` rejected positions past them. */
	[[nodiscard]] auto generate(Count index, Count count, Count drafts, std::uint64_t seed) -> bool {
		core::invariant(phase(index) == Phase::Generating);
		auto rng = std::mt19937_64{seed};
		auto tokens = std::vector<Token>(count);
		std::ranges::generate(tokens, [&] {
			return core::narrow<Token>(rng() % 3);
		});
		auto const grown = grow(index, tokens, drafts);
		check();
		return grown;
	}

	/** The slot's request ends and its sequence is cached at its end, as for a finished turn or a preemption. */
	auto close(Count index) -> void {
		core::invariant(live_[index].has_value());
		auto const sequence = live_[index]->sequence;
		if (sequence.empty()) {
			abandon(index);
			return;
		}
		insert(index, prefix::SequenceEnd{}, Afterwards::Leaves, [&] {
			return cache_->close(*device_, *pool_, Slot::at(index), sequence);
		});
		reference_leave(index);
		history_.push_back(sequence);
		check();
	}

	auto abandon(Count index) -> void {
		core::invariant(live_[index].has_value());
		cache_->abandon(*pool_, Slot::at(index));
		reference_leave(index);
		check();
	}

	/** Asks the cache for `more` free blocks than the pool has now. */
	[[nodiscard]] auto evict(Count more) -> prefix::Evicted {
		auto const evicted = make_room(pool_->free_blocks() + more);
		check();
		return evicted;
	}

	/** Every slot leaves, the cache gives back everything, and the pool is empty again. */
	auto finish() -> void {
		for (auto index = Count{0}; index < model::slots; ++index) {
			if (live_[index]) {
				abandon(index);
			}
		}
		std::ignore = evict(G.pool_blocks);
		expect(cached_.empty() && cache_->nodes() == 0, "with no slot left, the cache evicts everything");
		expect(pool_->free_blocks() == G.pool_blocks && pool_->snapshots() == 0 && pool_->held_only_blocks() == 0,
		       "the pool returns to empty");
	}

	[[nodiscard]] auto phase(Count index) const -> Phase {
		if (!live_[index]) {
			return Phase::Free;
		}
		return live_[index]->sequence.size() < live_[index]->prompt.size() ? Phase::Reading : Phase::Generating;
	}

	[[nodiscard]] auto sequence(Count index) const -> std::span<Token const> {
		core::invariant(live_[index].has_value());
		return live_[index]->sequence;
	}

	/** Sequences slots left with, oldest first. */
	[[nodiscard]] auto history() const -> std::span<std::vector<Token> const> {
		return history_;
	}

	[[nodiscard]] auto cache() const -> prefix::BasicCache<G> const& {
		return *cache_;
	}

	[[nodiscard]] auto pool() const -> kv::BasicPool<G> const& {
		return *pool_;
	}

	[[nodiscard]] auto failure() const -> std::optional<std::string> const& {
		return failure_;
	}

	[[nodiscard]] auto reached() const -> Reached {
		return reached_;
	}

private:
	struct Live {
		std::vector<Token> sequence;
		std::vector<Token> prompt;
		/** Snapshot points of the prompt not reached yet, increasing. */
		std::vector<prefix::Point> ahead;
	};

	/** The reference's cached sequence: who uses it, and when it was last left. */
	struct Cached {
		std::vector<Token> tokens;
		prefix::Mark mark;
		Count pins;
		std::uint64_t stamp;
	};

	auto expect(bool holds, std::string_view what) -> void {
		if (!holds && !failure_) {
			failure_ = std::string{what};
		}
	}

	[[nodiscard]] auto find_cached(std::span<Token const> tokens) const -> std::optional<std::size_t> {
		auto const found = std::ranges::find_if(cached_, [&](Cached const& entry) {
			return std::ranges::equal(entry.tokens, tokens);
		});
		return found == cached_.end() ? std::nullopt : std::optional{core::narrow<std::size_t>(found - cached_.begin())};
	}

	/** The deepest cached sequence that prefixes `prompt` and leaves its last token to read. */
	[[nodiscard]] auto deepest_cached(std::span<Token const> prompt) const -> std::optional<std::size_t> {
		auto best = std::optional<std::size_t>{};
		for (auto const [at, entry] : std::views::enumerate(cached_)) {
			if (entry.tokens.size() < prompt.size() && starts_with(prompt, entry.tokens) &&
			    (!best || entry.tokens.size() > cached_[*best].tokens.size())) {
				best = core::narrow<std::size_t>(at);
			}
		}
		return best;
	}

	/** The deepest point a live slot has yet to reach, past `above` and short of the prompt's last token, that it shares. */
	[[nodiscard]] auto pending_point(std::span<Token const> prompt, Count above) const -> std::optional<Count> {
		auto best = std::optional<Count>{};
		for (auto const& slot : live_) {
			if (!slot) {
				continue;
			}
			for (auto const& point : slot->ahead) {
				auto const position = point.position;
				if (position > std::max(above, best.value_or(0)) && position < prompt.size() &&
				    std::ranges::equal(prompt.first(position), std::span{slot->prompt}.first(position))) {
					best = position;
				}
			}
		}
		return best;
	}

	/**
	 * Entries the cache may evict next, in the order it must: unused, oldest first, and leaves of the tree: no other
	 * entry runs through their last block. An entry ending inside a block owns that block alone, so a longer entry
	 * extending its tokens runs through a block of its own and leaves it a leaf.
	 */
	[[nodiscard]] static auto candidates(std::span<Cached const> cached) -> std::vector<std::size_t> {
		auto found = std::vector<std::size_t>{};
		for (auto const [at, entry] : std::views::enumerate(cached)) {
			auto const extended = std::ranges::any_of(cached, [&](Cached const& other) {
				return entry.tokens.size() % G.block_tokens == 0 && other.tokens.size() > entry.tokens.size() &&
				       starts_with(other.tokens, entry.tokens);
			});
			if (entry.pins == 0 && !extended) {
				found.push_back(core::narrow<std::size_t>(at));
			}
		}
		std::ranges::sort(found, {}, [&](std::size_t at) {
			return cached[at].stamp;
		});
		return found;
	}

	auto reference_pin(Slot slot, std::span<Token const> tokens) -> void {
		auto& pinned = pinned_[slot.index()];
		if (pinned && std::ranges::equal(*pinned, tokens)) {
			return;
		}
		auto const at = find_cached(tokens);
		core::invariant(at.has_value());
		cached_[*at].pins += 1;
		if (auto const previous = std::exchange(pinned, std::vector<Token>{tokens.begin(), tokens.end()})) {
			reference_unpin(*previous);
		}
	}

	auto reference_unpin(std::span<Token const> tokens) -> void {
		auto const at = find_cached(tokens);
		core::invariant(at.has_value() && cached_[*at].pins > 0);
		cached_[*at].pins -= 1;
		if (cached_[*at].pins == 0) {
			clock_ += 1;
			cached_[*at].stamp = clock_;
		}
	}

	auto reference_leave(Count index) -> void {
		if (auto const pinned = std::exchange(pinned_[index], std::nullopt)) {
			reference_unpin(*pinned);
		}
		live_[index].reset();
	}

	/** Takes out of the reference what the cache evicted, after checking it is what the eviction order allows. */
	[[nodiscard]] auto evicted() -> Count {
		auto const entries = cache_->entries();
		auto removed = std::vector<std::vector<Token>>{};
		for (auto const& entry : cached_) {
			auto const kept = std::ranges::any_of(entries, [&](prefix::Entry const& each) {
				return std::ranges::equal(each.tokens, entry.tokens);
			});
			if (!kept) {
				removed.push_back(entry.tokens);
			}
		}
		auto simulated = cached_;
		auto order = std::vector<std::vector<Token>>{};
		for (auto step = std::size_t{0}; step < removed.size(); ++step) {
			auto const next = candidates(simulated);
			if (next.empty()) {
				break;
			}
			order.push_back(simulated[next.front()].tokens);
			simulated.erase(simulated.begin() + core::narrow<std::ptrdiff_t>(next.front()));
		}
		std::ranges::sort(removed);
		std::ranges::sort(order);
		expect(removed == order, "eviction takes unused leaves, oldest first");
		std::erase_if(cached_, [&](Cached const& entry) {
			return std::ranges::binary_search(removed, entry.tokens);
		});
		reached_.evicted += removed.size();
		return removed.size();
	}

	/** The cache's eviction toward `free` free blocks, checked against the reference's order and stopping rule. */
	[[nodiscard]] auto make_room(Count free) -> prefix::Evicted {
		auto const before = pool_->free_blocks();
		auto const done = cache_->make_room(*pool_, free);
		auto const removed = evicted();
		expect(done.snapshots == removed && done.blocks == pool_->free_blocks() - before, "an eviction reports what it gave back");
		expect(pool_->free_blocks() >= free || candidates(cached_).empty(), "eviction stops only when enough is free or nothing is left");
		expect(removed == 0 || before < free, "nothing is evicted while enough is free");
		return done;
	}

	[[nodiscard]] auto snapshot_would_be_refused() const -> std::optional<kv::SnapshotShortage> {
		if (pool_->snapshots() == G.snapshot_capacity) {
			return kv::SnapshotShortage::Capacity;
		}
		if (pool_->free_blocks() < G.snapshot_blocks) {
			return kv::SnapshotShortage::Blocks;
		}
		return std::nullopt;
	}

	/**
	 * One insertion through `call`, checked against the reference, which then follows it. A slot that leaves in the same
	 * call has emptied its table by the time the pool can be read, so only the snapshot count still shows the refusal.
	 */
	template<class Call>
	auto insert(Count index, prefix::Mark mark, Afterwards afterwards, Call call) -> void {
		auto const slot = Slot::at(index);
		auto const sequence = live_[index]->sequence;
		auto const present = find_cached(sequence).has_value();
		auto const refused_before = snapshot_would_be_refused();
		auto const done = call();
		auto const removed = evicted();
		std::erase_if(live_[index]->ahead, [&](prefix::Point const& point) {
			return point.position <= sequence.size();
		});
		if (!done) {
			reached_.refused += 1;
			expect(!present && candidates(cached_).empty(), "an insertion is refused only when nothing is left to evict");
			auto const capacity = done.error().shortage == kv::SnapshotShortage::Capacity;
			expect(capacity == (pool_->snapshots() == G.snapshot_capacity), "a refusal names the pool's shortage");
			expect(afterwards == Afterwards::Leaves || snapshot_would_be_refused() == done.error().shortage, "a refusal names the pool's shortage");
			return;
		}
		switch (*done) {
		case prefix::Insertion::Present:
			reached_.present += 1;
			expect(present && removed == 0, "a sequence already cached is used as it is");
			break;
		case prefix::Insertion::Saved:
			reached_.saved += 1;
			expect(!present, "a sequence not cached is saved");
			expect(removed == 0 || refused_before.has_value(), "an insertion evicts only when the pool refuses the snapshot");
			cached_.push_back({.tokens = sequence, .mark = mark, .pins = 0, .stamp = 0});
			break;
		}
		reference_pin(slot, sequence);
	}

	/** Makes room for and writes `tokens` and `drafts` rejected positions past them; false when the pool cannot. */
	[[nodiscard]] auto grow(Count index, std::span<Token const> tokens, Count drafts) -> bool {
		auto const slot = Slot::at(index);
		auto& live = *live_[index];
		auto const length = live.sequence.size();
		auto const extend = std::array{kv::Extend{.slot = slot, .length = length, .until = length + tokens.size() + drafts}};
		core::invariant(extend[0].until <= kv::table_positions(G));
		auto done = pool_->ensure(*device_, extend);
		if (!done) {
			std::ignore = make_room(done.error().needed);
			done = pool_->ensure(*device_, extend);
		}
		if (!done) {
			reached_.stalls += 1;
			return false;
		}
		auto chain = chain_of(live.sequence);
		for (auto const token : tokens) {
			chain = chain_step(chain, token);
			device_->write(slot, live.sequence.size(), Cell{Kv{.chain = chain}});
			live.sequence.push_back(token);
		}
		for (auto position = live.sequence.size(); position < extend[0].until; ++position) {
			device_->write(slot, position, Cell{Kv{.chain = chain ^ draft_mark}});
		}
		pool_->trim(slot, live.sequence.size());
		device_->set_state(slot, chain);
		return true;
	}

	[[nodiscard]] auto reads_back(kv::BlockIndex index, Count offset, std::uint64_t chain) const -> bool {
		auto const& content = device_->content(index);
		return std::holds_alternative<std::vector<Cell>>(content) &&
		       std::get<std::vector<Cell>>(content)[offset] == Cell{Kv{.chain = chain}};
	}

	[[nodiscard]] auto snapshot_holds(kv::SnapshotHandle snapshot, std::uint64_t state) const -> bool {
		auto const blocks = pool_->snapshot_blocks(snapshot);
		return blocks.has_value() && std::ranges::all_of(std::views::enumerate(*blocks), [&](auto const& part) {
			       auto const [at, index] = part;
			       return device_->content(index) == Content{kv_fake::Part{.state = state, .part = core::narrow<Count>(at)}};
		       });
	}

	auto check() -> void {
		expect(device_->violations().empty(), "the device sees only well-formed commands");
		auto const entries = cache_->entries();
		expect(entries.size() == cached_.size(), "the cache holds exactly the reference's sequences");
		auto nodes = std::map<std::vector<Token>, std::uint64_t>{};
		auto holders = std::map<std::uint64_t, kv::References>{};
		for (auto const& entry : entries) {
			auto const at = find_cached(entry.tokens);
			expect(at.has_value(), "every cached sequence is one the reference holds");
			if (at) {
				expect(entry.mark == cached_[*at].mark && entry.pins == cached_[*at].pins, "every entry's mark and users match");
			}
			expect(entry.blocks.size() == model::divide_up(entry.tokens.size(), G.block_tokens), "an entry's blocks cover it");
			if (entry.blocks.size() != model::divide_up(entry.tokens.size(), G.block_tokens)) {
				return;
			}
			auto chain = empty_chain;
			for (auto const [block, handle] : std::views::enumerate(entry.blocks)) {
				auto const first = core::narrow<Count>(block) * G.block_tokens;
				auto const last = std::min(entry.tokens.size(), first + G.block_tokens);
				auto const prefix = std::vector<Token>{entry.tokens.begin(), entry.tokens.begin() + core::narrow<std::ptrdiff_t>(last)};
				auto const [node, fresh] = nodes.emplace(prefix, handle.bits);
				expect(node->second == handle.bits, "one prefix is one node, holding one block");
				holders[handle.bits].held += fresh ? 1 : 0;
				auto const references = pool_->references(handle);
				expect(references.has_value(), "a cached block is live");
				if (!references) {
					return;
				}
				for (auto position = first; position < last; ++position) {
					chain = chain_step(chain, entry.tokens[position]);
					expect(reads_back(pool_->index(handle), position % G.block_tokens, chain), "a cached prefix reads back exactly");
				}
			}
			expect(snapshot_holds(entry.snapshot, chain), "a cached snapshot is the state after its sequence");
		}
		expect(nodes.size() == cache_->nodes(), "every node of the tree leads to a cached point");
		for (auto index = Count{0}; index < model::slots; ++index) {
			auto const slot = Slot::at(index);
			auto const size = pool_->table_size(slot);
			if (!live_[index]) {
				expect(size == 0, "a free slot holds no blocks");
				continue;
			}
			auto const& sequence = live_[index]->sequence;
			expect(size * G.block_tokens >= sequence.size(), "a slot's table covers its sequence");
			for (auto entry = Count{0}; entry < size; ++entry) {
				holders[pool_->entry(slot, entry).bits].tables += 1;
			}
			auto chain = empty_chain;
			for (auto position = Count{0}; position < sequence.size() && position < size * G.block_tokens; ++position) {
				chain = chain_step(chain, sequence[position]);
				auto const index_of = pool_->index(pool_->entry(slot, position / G.block_tokens));
				expect(device_->entry(slot, position / G.block_tokens) == index_of, "the device table follows the pool's");
				expect(reads_back(index_of, position % G.block_tokens, chain), "every live slot reads back its own sequence");
			}
			expect(device_->state(slot) == chain_of(sequence), "every live slot's state is the state after its sequence");
		}
		auto held_only = Count{0};
		for (auto const& [bits, expected] : holders) {
			expect(pool_->references(BlockHandle{bits}) == expected, "every block's counts equal its holders");
			held_only += expected.tables == 0 ? 1 : 0;
		}
		expect(pool_->held_only_blocks() == held_only, "blocks only the cache holds are counted as such");
		expect(pool_->snapshots() == entries.size(), "every snapshot in the pool is a cached point's");
		expect(pool_->free_blocks() + holders.size() + entries.size() * G.snapshot_blocks == G.pool_blocks,
		       "blocks are conserved: free, kv and snapshot make the pool");
	}

	std::unique_ptr<kv::BasicPool<G>> pool_;
	std::unique_ptr<kv_fake::FakeDevice<G>> device_;
	std::unique_ptr<prefix::BasicCache<G>> cache_;
	std::array<std::optional<Live>, model::slots> live_{};
	std::vector<Cached> cached_;
	std::array<std::optional<std::vector<Token>>, model::slots> pinned_{};
	std::uint64_t clock_ = 0;
	std::vector<std::vector<Token>> history_;
	std::optional<std::string> failure_;
	Reached reached_{};
};

}
