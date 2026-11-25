// SPDX-License-Identifier: AGPL-3.0-only
//
// Golden hashes: the canonical logit hash is SHA-256 of its documented byte form however the rows arrive; the golden
// file round-trips; a recording changes the file only with a reason; and a check reports exactly the mismatches.
import std;
import engine.model;
import engine.ref;
import engine.sha256;
import engine.testing;

namespace {

namespace ref = engine::ref;
namespace model = engine::model;
namespace sha256 = engine::sha256;

using model::Count;

[[nodiscard]] auto logits(std::uint64_t seed, Count rows) -> ref::Rows<float> {
	auto generator = engine::testing::generator(seed);
	auto value = std::uniform_real_distribution<float>{-20.0F, 20.0F};
	auto out = ref::Rows<float>::zeros(rows, model::sampled_vocabulary);
	for (auto& each : out.values) {
		each = value(generator);
	}
	return out;
}

/** The documented byte form, written out independently: the domain line, then each fp32 little-endian. */
[[nodiscard]] auto documented(ref::Rows<float> const& rows) -> sha256::Digest {
	auto bytes = std::vector<std::byte>{};
	for (auto const letter : std::string_view{"geezuz golden logits\n"}) {
		bytes.push_back(static_cast<std::byte>(letter));
	}
	for (auto const value : rows.values) {
		auto const bits = std::bit_cast<std::array<std::byte, 4>>(value);
		static_assert(std::endian::native == std::endian::little, "the development and production hosts are little-endian");
		bytes.insert(bytes.end(), bits.begin(), bits.end());
	}
	return sha256::digest(std::span<std::byte const>{bytes});
}

[[nodiscard]] auto hash_is_documented_and_piecewise(std::uint64_t seed) -> bool {
	auto const all = logits(seed, 5);
	auto whole = ref::LogitsHasher{};
	whole.update(all);
	auto pieces = ref::LogitsHasher{};
	pieces.update(ref::Rows<float>{.width = all.width, .values = {all.values.begin(), all.values.begin() + (2 * static_cast<std::ptrdiff_t>(all.width))}});
	pieces.update(ref::Rows<float>{.width = all.width, .values = {all.values.begin() + (2 * static_cast<std::ptrdiff_t>(all.width)), all.values.end()}});
	auto const digest = std::move(whole).finish();
	return digest == std::move(pieces).finish() && digest == documented(all);
}

[[nodiscard]] auto entry(std::string prompt, Count slots, std::uint64_t seed) -> ref::GoldenEntry {
	auto hasher = ref::LogitsHasher{};
	hasher.update(logits(seed, 1));
	return {.prompt = std::move(prompt), .live_slots = slots, .digest = std::move(hasher).finish()};
}

[[nodiscard]] auto recording_needs_reasons() -> bool {
	auto const first = ref::regolden(std::nullopt, {entry("alpha", 1, 1), entry("beta", 1, 2)}, "initial recording");
	if (!first || first->reasons.size() != 1) {
		return false;
	}
	auto const same = ref::regolden(*first, {entry("alpha", 1, 1), entry("beta", 1, 2)}, "nothing changed");
	auto const changed = ref::regolden(*first, {entry("alpha", 1, 1), entry("beta", 1, 3)}, "attention merge reordered");
	auto const no_reason = ref::regolden(*first, {entry("alpha", 1, 4)}, "");
	auto const two_lines = ref::regolden(*first, {entry("alpha", 1, 4)}, "a\nb");
	auto const duplicate = ref::regolden(std::nullopt, {entry("alpha", 1, 1), entry("alpha", 1, 2)}, "twice");
	return same && *same == *first && changed && changed->reasons.size() == 2 && changed->reasons.back() == "attention merge reordered" &&
	       !no_reason && !two_lines && duplicate.error().failure == ref::GoldenFailure::Duplicate;
}

[[nodiscard]] auto file_round_trips() -> bool {
	auto const golden = ref::Golden{.reasons = {"initial recording", "clock change"}, .entries = {entry("alpha", 1, 1), entry("beta", 8, 2)}};
	auto const text = ref::render(golden);
	auto const back = ref::parse_golden(text);
	auto const broken = ref::parse_golden(text.substr(0, text.size() - 5));
	return back && *back == golden && !broken && !ref::parse_golden("geezuz-golden\nentry alpha 1 00\n");
}

[[nodiscard]] auto check_reports_exactly_the_mismatches() -> bool {
	auto const golden = ref::Golden{.reasons = {"initial"}, .entries = {entry("alpha", 1, 1), entry("beta", 1, 2), entry("gamma", 1, 3)}};
	auto const candidate = ref::Golden{.reasons = {"engine run"}, .entries = {entry("alpha", 8, 1), entry("beta", 8, 9), entry("alpha", 3, 1)}};
	auto const found = ref::check(golden, candidate);
	return found.size() == 2 && found[0].prompt == "beta" && found[0].found.has_value() && found[1].prompt == "gamma" && !found[1].found;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the logit hash is SHA-256 of the documented bytes, however the rows are fed (seed 41)", .holds = hash_is_documented_and_piecewise(41)},
	    engine::testing::Case{.name = "a recording changes the file only with a one-line reason, appended to the history", .holds = recording_needs_reasons()},
	    engine::testing::Case{.name = "the golden file round-trips, and a cut or malformed file is refused", .holds = file_round_trips()},
	    engine::testing::Case{.name = "a check reports each differing slot mix and each missing prompt, nothing else", .holds = check_reports_exactly_the_mismatches()},
	};
	return engine::testing::report(cases);
}
