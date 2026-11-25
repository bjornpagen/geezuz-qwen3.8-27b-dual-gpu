// SPDX-License-Identifier: AGPL-3.0-only
export module engine.prefix:point;

import std;
import engine.core;
import engine.kv;
import engine.tokenizer;

namespace engine::prefix {

export using kv::Count;
export using Token = tokenizer::Id;

/**
 * The places in a rendered prompt where the chat format puts a deltanet snapshot: where a later request is likely to
 * diverge from this one. The chat layer marks them from the conversation's structure; nothing here guesses them from
 * tokens. Deltanet state is not positional, so a cached prefix resumes only at one of these.
 */
export enum class PointKind : std::uint8_t {
	/** The end of a leading system message, its tools included: every conversation under that system prompt shares it. */
	SystemEnd,
	/** The start of the last user message: a regenerated or edited last message diverges after it. */
	LastUserStart,
	/** The start of the generation prompt, which a later turn may render differently (without its thinking opener). */
	GenerationStart,
	/** The position before the prompt's last token: the same prompt sent again reads only that token. */
	PromptEnd,
};

inline constexpr Count point_kinds = 4;

/** A snapshot point of a prompt: the deltanet state after reading positions [0, position). */
export struct Point {
	PointKind kind;
	Count position;

	[[nodiscard]] constexpr auto operator==(Point const&) const -> bool = default;
};

/**
 * Where a slot's sequence stood when it left: a finished reply's end, or where a preempted request stopped. The next
 * turn of a conversation starts with the previous prompt and its reply, so this is where that turn resumes.
 */
export struct SequenceEnd {
	[[nodiscard]] constexpr auto operator==(SequenceEnd const&) const -> bool = default;
};

/** What a cached snapshot marks. */
export using Mark = std::variant<PointKind, SequenceEnd>;

/**
 * A prompt's snapshot points, strictly increasing and inside the prompt: what the chat layer marks, parsed once. A
 * point at position zero is the empty state every slot starts from and is dropped; kinds that fall on one position are
 * one point, the first marked.
 */
export class Points {
public:
	[[nodiscard]] static auto of(Count prompt_length, std::span<Point const> marked) -> Points {
		core::invariant(marked.size() <= point_kinds);
		auto points = Points{};
		for (auto const point : marked) {
			core::invariant(point.position < prompt_length);
			auto const taken = std::ranges::contains(points.all(), point.position, &Point::position);
			if (point.position == 0 || taken) {
				continue;
			}
			points.points_[points.size_] = point;
			points.size_ += 1;
		}
		std::ranges::sort(std::span{points.points_}.first(points.size_), {}, &Point::position);
		return points;
	}

	[[nodiscard]] auto all() const -> std::span<Point const> {
		return std::span{points_}.first(size_);
	}

	/** The points a prefill that resumes at `position` still reaches. */
	[[nodiscard]] auto after(Count position) const -> std::span<Point const> {
		auto const passed = std::ranges::count_if(all(), [&](Point const& point) {
			return point.position <= position;
		});
		return all().subspan(core::narrow<std::size_t>(passed));
	}

private:
	Points() = default;

	std::array<Point, point_kinds> points_{};
	Count size_ = 0;
};

}
