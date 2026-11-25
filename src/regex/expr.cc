// SPDX-License-Identifier: AGPL-3.0-only
export module engine.regex:expr;

import std;
import engine.core;

namespace engine::regex {

/** A set of bytes, one bit per value. */
export struct ByteSet {
	std::array<std::uint64_t, 4> words;

	[[nodiscard]] static constexpr auto none() -> ByteSet {
		return ByteSet{.words = {}};
	}

	[[nodiscard]] static constexpr auto range(std::uint8_t low, std::uint8_t high) -> ByteSet {
		auto set = none();
		for (auto byte = unsigned{low}; byte <= high; ++byte) {
			set.add(static_cast<std::uint8_t>(byte));
		}
		return set;
	}

	constexpr auto add(std::uint8_t byte) -> void {
		words.at(byte / 64U) |= std::uint64_t{1} << (byte % 64U);
	}

	[[nodiscard]] constexpr auto contains(std::uint8_t byte) const -> bool {
		return ((words.at(byte / 64U) >> (byte % 64U)) & 1U) != 0;
	}

	[[nodiscard]] constexpr auto empty() const -> bool {
		return std::ranges::all_of(words, [](std::uint64_t word) { return word == 0; });
	}

	[[nodiscard]] constexpr auto united(ByteSet const& other) const -> ByteSet {
		auto out = *this;
		for (auto index = std::size_t{0}; index < words.size(); ++index) {
			out.words.at(index) |= other.words.at(index);
		}
		return out;
	}

	[[nodiscard]] constexpr auto intersected(ByteSet const& other) const -> ByteSet {
		auto out = *this;
		for (auto index = std::size_t{0}; index < words.size(); ++index) {
			out.words.at(index) &= other.words.at(index);
		}
		return out;
	}

	[[nodiscard]] constexpr auto operator==(ByteSet const&) const -> bool = default;
};

/** A regular expression over bytes, hash-consed in its `Store`: equal ids are equal expressions. */
export struct Expr {
	std::uint32_t id;

	[[nodiscard]] constexpr auto operator==(Expr const&) const -> bool = default;
	[[nodiscard]] constexpr auto operator<=>(Expr const&) const = default;
};

/** The empty language. */
export inline constexpr auto nothing = Expr{.id = 0};
/** The empty string alone. */
export inline constexpr auto empty = Expr{.id = 1};
/** Every byte string. */
export inline constexpr auto everything = Expr{.id = 2};

/** A repetition with no upper bound. */
export inline constexpr auto unbounded = std::numeric_limits<std::uint32_t>::max();

struct Nothing {
	[[nodiscard]] constexpr auto operator==(Nothing const&) const -> bool = default;
};
struct Empty {
	[[nodiscard]] constexpr auto operator==(Empty const&) const -> bool = default;
};
struct Bytes {
	ByteSet set;

	[[nodiscard]] constexpr auto operator==(Bytes const&) const -> bool = default;
};
/** A sequence, kept right-nested: the head is never itself a sequence. */
struct Concat {
	Expr head;
	Expr tail;

	[[nodiscard]] constexpr auto operator==(Concat const&) const -> bool = default;
};
/** A union of at least two alternatives, sorted and distinct; byte sets merged into one. */
struct Or {
	std::vector<Expr> alternatives;

	[[nodiscard]] auto operator==(Or const&) const -> bool = default;
};
/** An intersection of at least two expressions, sorted and distinct. */
struct And {
	std::vector<Expr> conjuncts;

	[[nodiscard]] auto operator==(And const&) const -> bool = default;
};
struct Not {
	Expr negated;

	[[nodiscard]] constexpr auto operator==(Not const&) const -> bool = default;
};
/** `body{min,max}`, max `unbounded` for none; never the trivial forms `{1,1}` or `{0,0}`. */
struct Repeat {
	Expr body;
	std::uint32_t min;
	std::uint32_t max;

	[[nodiscard]] constexpr auto operator==(Repeat const&) const -> bool = default;
};

using Node = std::variant<Nothing, Empty, Bytes, Concat, Or, And, Not, Repeat>;

/** FNV-1a over 64-bit words. */
struct Fnv {
	std::uint64_t value;

	auto operator()(std::uint64_t word) -> void {
		value = (value ^ word) * 0x100000001b3;
	}
};

/** A node's hash, mixed from its kind and fields without building a key. */
[[nodiscard]] auto hash_of(Node const& node) -> std::uint64_t {
	auto mix = Fnv{.value = 0xcbf29ce484222325};
	mix(node.index());
	std::visit(
	    [&mix](auto const& held) {
		    using Held = std::remove_cvref_t<decltype(held)>;
		    if constexpr (std::is_same_v<Held, Bytes>) {
			    for (auto const word : held.set.words) {
				    mix(word);
			    }
		    } else if constexpr (std::is_same_v<Held, Concat>) {
			    mix(held.head.id);
			    mix(held.tail.id);
		    } else if constexpr (std::is_same_v<Held, Or>) {
			    for (auto const part : held.alternatives) {
				    mix(part.id);
			    }
		    } else if constexpr (std::is_same_v<Held, And>) {
			    for (auto const part : held.conjuncts) {
				    mix(part.id);
			    }
		    } else if constexpr (std::is_same_v<Held, Not>) {
			    mix(held.negated.id);
		    } else if constexpr (std::is_same_v<Held, Repeat>) {
			    mix(held.body.id);
			    mix(held.min);
			    mix(held.max);
		    }
	    },
	    node);
	// The table indexes by the low bits, which FNV leaves weak for small words: they are mixed down from the high ones.
	auto const hash = mix.value;
	return ((hash ^ (hash >> 31U)) * 0x9e3779b97f4a7c15U) ^ (hash >> 29U);
}

/**
 * Ids by hash, in open addressing: finding an existing node allocates nothing. Slots hold id + 1, 0 when empty; the
 * table doubles before it is half full, rehashing from the hashes the store keeps per node.
 */
class IdTable {
public:
	template<class Same>
	[[nodiscard]] auto find(std::uint64_t hash, Same same) const -> std::optional<std::uint32_t> {
		if (slots_.empty()) {
			return std::nullopt;
		}
		for (auto at = hash & (slots_.size() - 1);; at = (at + 1) & (slots_.size() - 1)) {
			auto const slot = slots_[at];
			if (slot == 0) {
				return std::nullopt;
			}
			if (same(slot - 1)) {
				return slot - 1;
			}
		}
	}

	template<class HashOf>
	auto insert(std::uint64_t hash, std::uint32_t id, HashOf hash_of_id) -> void {
		if ((count_ + 1) * 2 > slots_.size()) {
			auto const old = std::exchange(slots_, std::vector<std::uint32_t>(std::max<std::size_t>(64, slots_.size() * 2), 0));
			for (auto const slot : old) {
				if (slot != 0) {
					place(hash_of_id(slot - 1), slot);
				}
			}
		}
		place(hash, id + 1);
		++count_;
	}

private:
	auto place(std::uint64_t hash, std::uint32_t slot) -> void {
		auto at = hash & (slots_.size() - 1);
		while (slots_[at] != 0) {
			at = (at + 1) & (slots_.size() - 1);
		}
		slots_[at] = slot;
	}

	std::vector<std::uint32_t> slots_;
	std::size_t count_ = 0;
};

/** Derivatives by (expression, byte), in open addressing like `IdTable`. */
class DerivativeTable {
public:
	[[nodiscard]] auto find(std::uint64_t key) const -> std::optional<std::uint32_t> {
		if (keys_.empty()) {
			return std::nullopt;
		}
		for (auto at = mixed(key) & (keys_.size() - 1);; at = (at + 1) & (keys_.size() - 1)) {
			if (keys_[at] == empty_key) {
				return std::nullopt;
			}
			if (keys_[at] == key) {
				return values_[at];
			}
		}
	}

	auto insert(std::uint64_t key, std::uint32_t value) -> void {
		if ((count_ + 1) * 2 > keys_.size()) {
			auto const old_keys = std::exchange(keys_, std::vector<std::uint64_t>(std::max<std::size_t>(256, keys_.size() * 2), empty_key));
			auto const old_values = std::exchange(values_, std::vector<std::uint32_t>(keys_.size(), 0));
			for (auto at = std::size_t{0}; at < old_keys.size(); ++at) {
				if (old_keys[at] != empty_key) {
					place(old_keys[at], old_values[at]);
				}
			}
		}
		place(key, value);
		++count_;
	}

private:
	static constexpr auto empty_key = std::numeric_limits<std::uint64_t>::max();

	[[nodiscard]] static auto mixed(std::uint64_t key) -> std::uint64_t {
		return (key ^ (key >> 29U)) * 0x9e3779b97f4a7c15U;
	}

	auto place(std::uint64_t key, std::uint32_t value) -> void {
		auto at = mixed(key) & (keys_.size() - 1);
		while (keys_[at] != empty_key) {
			at = (at + 1) & (keys_.size() - 1);
		}
		keys_[at] = key;
		values_[at] = value;
	}

	std::vector<std::uint64_t> keys_;
	std::vector<std::uint32_t> values_;
	std::size_t count_ = 0;
};

/**
 * The expressions of one grammar, hash-consed, with their nullability and derivatives. Smart constructors keep every
 * node in a normal form (flattened, sorted, distinct unions and intersections, right-nested sequences), which is what
 * makes the derivatives of an expression finitely many: each lexeme's automaton is the set of its derivatives.
 *
 * A store holds at most `limit` nodes; past that every constructor returns `nothing` and `full()` turns true, so a
 * caller building from client input checks `full()` once at its boundary instead of at every step.
 */
export class Store {
public:
	explicit Store(std::size_t limit) : limit_{limit} {
		std::ignore = intern(Nothing{});
		std::ignore = intern(Empty{});
		std::ignore = intern(Not{.negated = nothing});
	}

	[[nodiscard]] auto full() const -> bool {
		return full_;
	}

	[[nodiscard]] auto size() const -> std::size_t {
		return nodes_.size();
	}

	[[nodiscard]] auto bytes(ByteSet const& set) -> Expr {
		if (set.empty()) {
			return nothing;
		}
		return intern(Bytes{.set = set});
	}

	[[nodiscard]] auto byte(std::uint8_t value) -> Expr {
		auto set = ByteSet::none();
		set.add(value);
		return bytes(set);
	}

	[[nodiscard]] auto literal(std::string_view text) -> Expr {
		auto out = empty;
		for (auto const letter : text | std::views::reverse) {
			out = concat(byte(std::bit_cast<std::uint8_t>(letter)), out);
		}
		return out;
	}

	[[nodiscard]] auto concat(Expr head, Expr tail) -> Expr {
		if (head == nothing || tail == nothing) {
			return nothing;
		}
		if (head == empty) {
			return tail;
		}
		if (tail == empty) {
			return head;
		}
		if (auto const sequence = core::alternative<Concat>(std::as_const(nodes_[head.id]))) {
			auto const inner_head = sequence->head;
			auto const inner_tail = sequence->tail;
			return concat(inner_head, concat(inner_tail, tail));
		}
		return intern(Concat{.head = head, .tail = tail});
	}

	[[nodiscard]] auto sequence(std::span<Expr const> parts) -> Expr {
		auto out = empty;
		for (auto const part : parts | std::views::reverse) {
			out = concat(part, out);
		}
		return out;
	}

	[[nodiscard]] auto either(Expr first, Expr second) -> Expr {
		auto const both = std::array{first, second};
		return any_of(both);
	}

	[[nodiscard]] auto any_of(std::span<Expr const> choices) -> Expr {
		auto flat = std::vector<Expr>{};
		auto merged = ByteSet::none();
		auto has_bytes = false;
		for (auto const choice : choices) {
			if (choice == everything) {
				return everything;
			}
			if (choice == nothing) {
				continue;
			}
			if (auto const union_node = core::alternative<Or>(std::as_const(nodes_[choice.id]))) {
				for (auto const inner : union_node->alternatives) {
					if (auto const set = core::alternative<Bytes>(std::as_const(nodes_[inner.id]))) {
						merged = merged.united(set->set);
						has_bytes = true;
					} else {
						flat.push_back(inner);
					}
				}
			} else if (auto const set = core::alternative<Bytes>(std::as_const(nodes_[choice.id]))) {
				merged = merged.united(set->set);
				has_bytes = true;
			} else {
				flat.push_back(choice);
			}
		}
		if (has_bytes) {
			flat.push_back(bytes(merged));
		}
		std::ranges::sort(flat);
		auto const [first, last] = std::ranges::unique(flat);
		flat.erase(first, last);
		if (flat.empty()) {
			return nothing;
		}
		if (flat.size() == 1) {
			return flat.front();
		}
		return intern(Or{.alternatives = std::move(flat)});
	}

	[[nodiscard]] auto both(Expr first, Expr second) -> Expr {
		auto const pair = std::array{first, second};
		return all_of(pair);
	}

	[[nodiscard]] auto all_of(std::span<Expr const> conjuncts) -> Expr {
		auto flat = std::vector<Expr>{};
		for (auto const conjunct : conjuncts) {
			if (conjunct == nothing) {
				return nothing;
			}
			if (conjunct == everything) {
				continue;
			}
			if (auto const intersection = core::alternative<And>(std::as_const(nodes_[conjunct.id]))) {
				std::ranges::copy(intersection->conjuncts, std::back_inserter(flat));
			} else {
				flat.push_back(conjunct);
			}
		}
		std::ranges::sort(flat);
		auto const [first, last] = std::ranges::unique(flat);
		flat.erase(first, last);
		if (flat.empty()) {
			return everything;
		}
		if (flat.size() == 1) {
			return flat.front();
		}
		// Byte sets intersect to one; an intersection with a set of single bytes and the empty string is empty.
		auto const sets = std::ranges::count_if(flat, [this](Expr part) { return std::holds_alternative<Bytes>(nodes_[part.id]); });
		if (sets > 1) {
			auto merged = ByteSet{.words = {~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}, ~std::uint64_t{0}}};
			auto rest = std::vector<Expr>{};
			for (auto const part : flat) {
				if (auto const set = core::alternative<Bytes>(std::as_const(nodes_[part.id]))) {
					merged = merged.intersected(set->set);
				} else {
					rest.push_back(part);
				}
			}
			rest.push_back(bytes(merged));
			return all_of(rest);
		}
		if (std::ranges::contains(flat, empty)) {
			return std::ranges::all_of(flat, [this](Expr part) { return nullable(part); }) ? empty : nothing;
		}
		return intern(And{.conjuncts = std::move(flat)});
	}

	[[nodiscard]] auto complement(Expr negated) -> Expr {
		if (auto const inner = core::alternative<Not>(std::as_const(nodes_[negated.id]))) {
			return inner->negated;
		}
		return intern(Not{.negated = negated});
	}

	/** `body{min,max}`; max is `unbounded` for none. */
	[[nodiscard]] auto repeat(Expr body, std::uint32_t min, std::uint32_t max) -> Expr {
		core::invariant(min <= max);
		if (max == 0 || body == empty) {
			return empty;
		}
		if (body == nothing) {
			return min == 0 ? empty : nothing;
		}
		if (min == 1 && max == 1) {
			return body;
		}
		return intern(Repeat{.body = body, .min = min, .max = max});
	}

	[[nodiscard]] auto optional(Expr body) -> Expr {
		return either(empty, body);
	}

	/** The bytes of an expression that is one literal string (a chain of single bytes), if it is one. */
	[[nodiscard]] auto literal_text(Expr expr) const -> std::optional<std::string> {
		auto out = std::string{};
		while (expr != empty) {
			auto const sequence = core::alternative<Concat>(nodes_[expr.id]);
			auto const head = core::alternative<Bytes>(sequence ? nodes_[sequence->head.id] : nodes_[expr.id]);
			if (!head) {
				return std::nullopt;
			}
			auto const set = head->set;
			auto const first = std::ranges::find_if(std::views::iota(0U, 256U), [&set](unsigned byte) { return set.contains(static_cast<std::uint8_t>(byte)); });
			auto single = ByteSet::none();
			single.add(static_cast<std::uint8_t>(*first));
			if (!(single == set)) {
				return std::nullopt;
			}
			out.push_back(static_cast<char>(static_cast<std::uint8_t>(*first)));
			expr = sequence ? sequence->tail : empty;
		}
		return out;
	}

	[[nodiscard]] auto nullable(Expr expr) const -> bool {
		return nullable_[expr.id] != 0;
	}

	/** The words that follow `byte` in the language: `{ w | byte w in L(expr) }`. */
	[[nodiscard]] auto derivative(Expr expr, std::uint8_t byte) -> Expr {
		auto const key = (std::uint64_t{expr.id} << 8U) | byte;
		if (auto const found = derivatives_.find(key)) {
			return Expr{.id = *found};
		}
		// A reference stays valid while the recursion below interns new nodes: the deque never moves what it holds.
		auto const& node = nodes_[expr.id];
		auto const out = std::visit(
		    [this, byte](auto const& held) -> Expr {
			    using Held = std::remove_cvref_t<decltype(held)>;
			    if constexpr (std::is_same_v<Held, Nothing> || std::is_same_v<Held, Empty>) {
				    return nothing;
			    } else if constexpr (std::is_same_v<Held, Bytes>) {
				    return held.set.contains(byte) ? empty : nothing;
			    } else if constexpr (std::is_same_v<Held, Concat>) {
				    auto const first = concat(derivative(held.head, byte), held.tail);
				    return nullable(held.head) ? either(first, derivative(held.tail, byte)) : first;
			    } else if constexpr (std::is_same_v<Held, Or>) {
				    auto parts = std::vector<Expr>{};
				    parts.reserve(held.alternatives.size());
				    for (auto const alternative : held.alternatives) {
					    parts.push_back(derivative(alternative, byte));
				    }
				    return any_of(parts);
			    } else if constexpr (std::is_same_v<Held, And>) {
				    auto parts = std::vector<Expr>{};
				    parts.reserve(held.conjuncts.size());
				    for (auto const conjunct : held.conjuncts) {
					    parts.push_back(derivative(conjunct, byte));
				    }
				    return all_of(parts);
			    } else if constexpr (std::is_same_v<Held, Not>) {
				    return complement(derivative(held.negated, byte));
			    } else {
				    auto const rest = repeat(held.body, held.min == 0 ? 0 : held.min - 1, held.max == unbounded ? unbounded : held.max - 1);
				    return concat(derivative(held.body, byte), rest);
			    }
		    },
		    node);
		derivatives_.insert(key, out.id);
		return out;
	}

	/**
	 * The partition of the 256 byte values that `roots` cannot tell apart: two bytes in one class have equal
	 * derivatives for every expression reachable from the roots. Returns each byte's class; classes count up from 0.
	 */
	[[nodiscard]] auto byte_classes(std::span<Expr const> roots) const -> std::array<std::uint8_t, 256> {
		auto classes = std::array<std::uint16_t, 256>{};
		// A set, not a bitmap over the store: a grammar's store is large and each lexeme reaches little of it.
		auto seen = std::unordered_set<std::uint32_t>{};
		auto pending = std::vector<Expr>{roots.begin(), roots.end()};
		while (!pending.empty()) {
			auto const expr = pending.back();
			pending.pop_back();
			if (!seen.insert(expr.id).second) {
				continue;
			}
			std::visit(
			    [&](auto const& held) {
				    using Held = std::remove_cvref_t<decltype(held)>;
				    if constexpr (std::is_same_v<Held, Bytes>) {
					    refine(classes, held.set);
				    } else if constexpr (std::is_same_v<Held, Concat>) {
					    pending.push_back(held.head);
					    pending.push_back(held.tail);
				    } else if constexpr (std::is_same_v<Held, Or>) {
					    std::ranges::copy(held.alternatives, std::back_inserter(pending));
				    } else if constexpr (std::is_same_v<Held, And>) {
					    std::ranges::copy(held.conjuncts, std::back_inserter(pending));
				    } else if constexpr (std::is_same_v<Held, Not>) {
					    pending.push_back(held.negated);
				    } else if constexpr (std::is_same_v<Held, Repeat>) {
					    pending.push_back(held.body);
				    }
			    },
			    nodes_[expr.id]);
		}
		auto out = std::array<std::uint8_t, 256>{};
		std::ranges::transform(classes, out.begin(), [](std::uint16_t value) { return core::narrow<std::uint8_t>(value); });
		return out;
	}

private:
	/** Splits every class by membership in `set`; the class numbers stay dense from 0. */
	static auto refine(std::array<std::uint16_t, 256>& classes, ByteSet const& set) -> void {
		constexpr auto unassigned = std::numeric_limits<std::uint16_t>::max();
		auto renumbered = std::array<std::uint16_t, 512>{};
		renumbered.fill(unassigned);
		auto next = std::uint16_t{0};
		for (auto byte = 0U; byte < 256U; ++byte) {
			auto const key = (std::size_t{classes.at(byte)} * 2) + (set.contains(static_cast<std::uint8_t>(byte)) ? 1U : 0U);
			if (renumbered.at(key) == unassigned) {
				renumbered.at(key) = next++;
			}
			classes.at(byte) = renumbered.at(key);
		}
	}

	[[nodiscard]] auto intern(Node node) -> Expr {
		auto const hash = hash_of(node);
		if (auto const found = index_.find(hash, [&](std::uint32_t id) { return nodes_[id] == node; })) {
			return Expr{.id = *found};
		}
		if (nodes_.size() >= limit_) {
			full_ = true;
			return nothing;
		}
		auto const expr = Expr{.id = core::narrow<std::uint32_t>(nodes_.size())};
		nullable_.push_back(compute_nullable(node) ? 1 : 0);
		nodes_.push_back(std::move(node));
		hashes_.push_back(hash);
		index_.insert(hash, expr.id, [this](std::uint32_t id) { return hashes_[id]; });
		return expr;
	}

	[[nodiscard]] auto compute_nullable(Node const& node) const -> bool {
		return std::visit(
		    [this](auto const& held) -> bool {
			    using Held = std::remove_cvref_t<decltype(held)>;
			    if constexpr (std::is_same_v<Held, Nothing> || std::is_same_v<Held, Bytes>) {
				    return false;
			    } else if constexpr (std::is_same_v<Held, Empty>) {
				    return true;
			    } else if constexpr (std::is_same_v<Held, Concat>) {
				    return nullable(held.head) && nullable(held.tail);
			    } else if constexpr (std::is_same_v<Held, Or>) {
				    return std::ranges::any_of(held.alternatives, [this](Expr part) { return nullable(part); });
			    } else if constexpr (std::is_same_v<Held, And>) {
				    return std::ranges::all_of(held.conjuncts, [this](Expr part) { return nullable(part); });
			    } else if constexpr (std::is_same_v<Held, Not>) {
				    return !nullable(held.negated);
			    } else {
				    return held.min == 0 || nullable(held.body);
			    }
		    },
		    node);
	}

	std::size_t limit_;
	bool full_ = false;
	std::deque<Node> nodes_;
	std::vector<std::uint8_t> nullable_;
	std::vector<std::uint64_t> hashes_;
	IdTable index_;
	DerivativeTable derivatives_;
};

}
