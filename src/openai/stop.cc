// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:stop;

import std;
import engine.core;
import engine.json;

namespace engine::openai {

/**
 * The longest stop sequence. A scan holds back at most one byte less than this, so it bounds both the scan's memory
 * and how long a stream's text can wait behind a possible match.
 */
export inline constexpr std::size_t max_stop_bytes = 256;

export enum class StopFault : std::uint8_t {
	/** An empty sequence would match before the first token. */
	Empty,
	TooLong,
};

/** A refused stop list: what is wrong, and which sequence. */
export struct [[nodiscard]] StopError {
	StopFault fault;
	std::size_t index;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(StopError const&) const -> bool = default;
};

/** A request's stop sequences: at most json::max_stop_sequences, each 1 to max_stop_bytes of UTF-8. */
export class StopSequences {
public:
	[[nodiscard]] static auto parse(std::vector<std::string> sequences) -> core::Result<StopSequences, StopError> {
		core::invariant(sequences.size() <= json::max_stop_sequences);
		for (auto index = std::size_t{0}; index < sequences.size(); ++index) {
			if (sequences[index].empty()) {
				return std::unexpected(StopError{.fault = StopFault::Empty, .index = index});
			}
			if (sequences[index].size() > max_stop_bytes) {
				return std::unexpected(StopError{.fault = StopFault::TooLong, .index = index});
			}
		}
		return StopSequences{std::move(sequences)};
	}

	[[nodiscard]] auto sequences() const -> std::span<std::string const> {
		return sequences_;
	}

	[[nodiscard]] auto operator==(StopSequences const&) const -> bool = default;

private:
	explicit StopSequences(std::vector<std::string> sequences) : sequences_{std::move(sequences)} {}

	std::vector<std::string> sequences_;
};

/** No stop sequence has completed: `text` is safe to emit, the scan goes on. */
export struct Continuing {
	std::string text;

	[[nodiscard]] auto operator==(Continuing const&) const -> bool = default;
};

/** A stop sequence completed: `text` is what precedes it, the last of the completion; the sequence is not emitted. */
export struct Stopped {
	std::string text;

	[[nodiscard]] auto operator==(Stopped const&) const -> bool = default;
};

export using Scanned = std::variant<Continuing, Stopped>;

/**
 * Finds a request's stop sequences in its completion's content as it is generated, holding back only text that may
 * still begin a match. The pipeline feeds it the detokenized content (not reasoning, not tool-call arguments) one
 * delta at a time, at the token that produced the delta, so the completion ends at that token with exact usage.
 * Deltas are whole UTF-8 characters; so are the sequences, so a held-back suffix starts on a character boundary and
 * every released text is UTF-8.
 */
export class StopScan {
public:
	explicit StopScan(StopSequences sequences) : sequences_{std::move(sequences)} {}

	/** The next delta. After a `Stopped`, the completion is over and nothing more may be fed. */
	[[nodiscard]] auto feed(std::string_view delta) -> Scanned {
		core::invariant(state_ == State::Scanning);
		auto window = std::move(held_);
		window += delta;
		auto earliest = std::string::npos;
		for (auto const& sequence : sequences_.sequences()) {
			earliest = std::min(earliest, window.find(sequence));
		}
		if (earliest != std::string::npos) {
			state_ = State::Stopped;
			window.resize(earliest);
			held_ = std::string{};
			return Stopped{.text = std::move(window)};
		}
		auto const keep = held_suffix(window);
		held_.assign(window, window.size() - keep);
		window.resize(window.size() - keep);
		return Continuing{.text = std::move(window)};
	}

	/** The generation ended without a match: the held-back text belongs to the completion after all. */
	[[nodiscard]] auto finish() && -> std::string {
		core::invariant(state_ == State::Scanning);
		return std::move(held_);
	}

	/** Bytes held back now; always below the longest sequence. */
	[[nodiscard]] auto held() const -> std::size_t {
		return held_.size();
	}

private:
	enum class State : std::uint8_t {
		Scanning,
		Stopped,
	};

	/** The longest suffix of `window` that is a proper prefix of some sequence: a match it may still begin. */
	[[nodiscard]] auto held_suffix(std::string_view window) const -> std::size_t {
		auto longest = std::size_t{0};
		for (auto const& sequence : sequences_.sequences()) {
			for (auto length = std::min(sequence.size() - 1, window.size()); length > longest; --length) {
				if (window.ends_with(std::string_view{sequence}.substr(0, length))) {
					longest = length;
					break;
				}
			}
		}
		return longest;
	}

	StopSequences sequences_;
	std::string held_;
	State state_ = State::Scanning;
};

}
