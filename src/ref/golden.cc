// SPDX-License-Identifier: AGPL-3.0-only
//
// Golden logit hashes. A prompt's logits hash in one canonical byte form: the ascii line "geezuz golden logits\n",
// then every row in position order, each the row's logits over the sampled vocabulary in token order as little-endian
// fp32 bit patterns. The golden file is text, one fact per line:
//
//   geezuz-golden
//   reason <why the entries below were recorded>        one per recording that changed them, oldest first
//   entry <prompt> <live slots> <sha256 hex>
//
// Batch invariance makes a slot's logits independent of the other slots, so the reference records each prompt alone
// (one live slot) and every slot mix the engine runs is checked against that one hash.
export module engine.ref:golden;

import std;
import engine.core;
import engine.model;
import engine.sha256;
import :numeric;

namespace engine::ref {

namespace model = engine::model;

using model::Count;

inline constexpr auto logits_domain = std::string_view{"geezuz golden logits\n"};
inline constexpr auto golden_header = std::string_view{"geezuz-golden"};

/** Hashes a prompt's logits in the canonical form, rows fed in position order in pieces of any size. */
export class LogitsHasher {
public:
	LogitsHasher() {
		hasher_.update(logits_domain);
	}

	auto update(Rows<float> const& logits) -> void {
		core::invariant(logits.width == model::sampled_vocabulary);
		auto bytes = std::vector<std::byte>{};
		bytes.reserve(logits.values.size() * 4);
		for (auto const logit : logits.values) {
			auto const bits = std::bit_cast<std::uint32_t>(logit);
			for (auto shift = 0U; shift < 32U; shift += 8U) {
				bytes.push_back(std::byte{static_cast<std::uint8_t>(bits >> shift)});
			}
		}
		hasher_.update(std::span<std::byte const>{bytes});
	}

	[[nodiscard]] auto finish() && -> sha256::Digest {
		return std::move(hasher_).finish();
	}

private:
	sha256::Hasher hasher_;
};

export struct GoldenEntry {
	std::string prompt;
	Count live_slots;
	sha256::Digest digest;

	[[nodiscard]] auto operator==(GoldenEntry const&) const -> bool = default;
};

export struct Golden {
	std::vector<std::string> reasons;
	std::vector<GoldenEntry> entries;

	[[nodiscard]] auto operator==(Golden const&) const -> bool = default;
};

export enum class GoldenFailure : std::uint8_t {
	/** The text is not a golden file: a missing header, an unknown line, or a malformed entry. */
	Malformed,
	/** A recording without a reason, or a reason or prompt name that would not stay one field of one line. */
	Unreadable,
	/** Two entries for one prompt and slot count. */
	Duplicate,
};

export struct [[nodiscard]] GoldenError {
	GoldenFailure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case GoldenFailure::Malformed:
		case GoldenFailure::Unreadable:
		case GoldenFailure::Duplicate:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(GoldenError const&) const -> bool = default;
};

export [[nodiscard]] auto render(Golden const& golden) -> std::string {
	auto text = std::format("{}\n", golden_header);
	for (auto const& reason : golden.reasons) {
		text += std::format("reason {}\n", reason);
	}
	for (auto const& entry : golden.entries) {
		text += std::format("entry {} {} {}\n", entry.prompt, entry.live_slots, entry.digest);
	}
	return text;
}

[[nodiscard]] constexpr auto one_field(std::string_view text) -> bool {
	return !text.empty() && std::ranges::none_of(text, [](char letter) {
		return letter == ' ' || letter == '\t' || letter == '\n' || letter == '\r';
	});
}

[[nodiscard]] constexpr auto one_line(std::string_view text) -> bool {
	return !text.empty() && std::ranges::none_of(text, [](char letter) {
		return letter == '\n' || letter == '\r';
	});
}

[[nodiscard]] auto entry_of(std::string_view fields) -> std::optional<GoldenEntry> {
	auto words = std::vector<std::string_view>{};
	for (auto const word : std::views::split(fields, ' ')) {
		words.emplace_back(word.begin(), word.end());
	}
	if (words.size() != 3 || !one_field(words[0])) {
		return std::nullopt;
	}
	auto slots = Count{0};
	auto const read = std::from_chars(words[1].data(), std::to_address(words[1].end()), slots);
	auto const digest = sha256::parse(words[2]);
	if (std::make_error_code(read.ec) || read.ptr != std::to_address(words[1].end()) || slots < 1 || slots > model::slots || !digest) {
		return std::nullopt;
	}
	return GoldenEntry{.prompt = std::string{words[0]}, .live_slots = slots, .digest = *digest};
}

[[nodiscard]] auto distinct(std::span<GoldenEntry const> entries) -> bool {
	auto seen = std::set<std::pair<std::string_view, Count>>{};
	return std::ranges::all_of(entries, [&](GoldenEntry const& entry) {
		return seen.emplace(entry.prompt, entry.live_slots).second;
	});
}

export [[nodiscard]] auto parse_golden(std::string_view text) -> core::Result<Golden, GoldenError> {
	auto const malformed = std::unexpected(GoldenError{.failure = GoldenFailure::Malformed});
	auto lines = std::vector<std::string_view>{};
	for (auto const line : std::views::split(text, '\n')) {
		lines.emplace_back(line.begin(), line.end());
	}
	if (lines.empty() || lines.front() != golden_header || !lines.back().empty()) {
		return malformed;
	}
	auto golden = Golden{};
	for (auto const line : std::span{lines}.subspan(1, lines.size() - 2)) {
		if (line.starts_with("reason ") && golden.entries.empty() && one_line(line.substr(7))) {
			golden.reasons.emplace_back(line.substr(7));
		} else if (auto entry = line.starts_with("entry ") ? entry_of(line.substr(6)) : std::nullopt) {
			golden.entries.push_back(std::move(*entry));
		} else {
			return malformed;
		}
	}
	if (golden.reasons.empty()) {
		return malformed;
	}
	if (!distinct(golden.entries)) {
		return std::unexpected(GoldenError{.failure = GoldenFailure::Duplicate});
	}
	return golden;
}

/**
 * The golden file after recording `fresh` for `reason`: unchanged when the entries are, else the fresh entries with
 * the reason appended to the history. Every recording names its reason, the first one too.
 */
export [[nodiscard]] auto regolden(std::optional<Golden> const& previous, std::vector<GoldenEntry> fresh, std::string const& reason)
    -> core::Result<Golden, GoldenError> {
	if (!one_line(reason) || !std::ranges::all_of(fresh, [](GoldenEntry const& entry) {
		    return one_field(entry.prompt);
	    })) {
		return std::unexpected(GoldenError{.failure = GoldenFailure::Unreadable});
	}
	if (!distinct(fresh)) {
		return std::unexpected(GoldenError{.failure = GoldenFailure::Duplicate});
	}
	if (previous && previous->entries == fresh) {
		return *previous;
	}
	auto reasons = previous ? previous->reasons : std::vector<std::string>{};
	reasons.push_back(reason);
	return Golden{.reasons = std::move(reasons), .entries = std::move(fresh)};
}

/** A candidate's entry whose hash is not its prompt's golden one; or a golden prompt the candidate lacks. */
export struct Mismatch {
	std::string prompt;
	Count live_slots;
	sha256::Digest golden;
	std::optional<sha256::Digest> found;
};

/**
 * Every entry of `candidate` against the golden hash of its prompt, whatever its slot count; and every golden prompt
 * the candidate has no entry for.
 */
export [[nodiscard]] auto check(Golden const& golden, Golden const& candidate) -> std::vector<Mismatch> {
	auto mismatches = std::vector<Mismatch>{};
	for (auto const& expected : golden.entries) {
		auto covered = false;
		for (auto const& found : candidate.entries) {
			if (found.prompt != expected.prompt) {
				continue;
			}
			covered = true;
			if (found.digest != expected.digest) {
				mismatches.push_back({.prompt = found.prompt, .live_slots = found.live_slots, .golden = expected.digest, .found = found.digest});
			}
		}
		if (!covered) {
			mismatches.push_back({.prompt = expected.prompt, .live_slots = expected.live_slots, .golden = expected.digest, .found = std::nullopt});
		}
	}
	return mismatches;
}

}
