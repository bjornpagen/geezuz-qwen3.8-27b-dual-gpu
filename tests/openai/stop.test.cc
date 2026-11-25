// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.openai;
import engine.testing;

namespace openai = engine::openai;

namespace {

/** A small alphabet, so that sequences recur and overlap; one character is two bytes, one is four. */
constexpr auto alphabet = std::array<std::string_view, 5>{"a", "b", "c", "é", "\U0001F600"};

template<class G>
[[nodiscard]] auto word(G& generator, std::size_t longest) -> std::string {
	auto length = std::uniform_int_distribution<std::size_t>{1, longest}(generator);
	auto out = std::string{};
	while (length-- > 0) {
		out += alphabet[std::uniform_int_distribution<std::size_t>{0, alphabet.size() - 1}(generator)];
	}
	return out;
}

/** The oracle: feed the whole text so far after every delta; the first delta whose text holds a sequence ends it. */
struct Expected {
	std::string text;
	std::optional<std::size_t> stopped_at;
};

[[nodiscard]] auto expected(std::span<std::string const> sequences, std::span<std::string const> deltas) -> Expected {
	auto so_far = std::string{};
	for (auto index = std::size_t{0}; index < deltas.size(); ++index) {
		so_far += deltas[index];
		auto earliest = std::string::npos;
		for (auto const& sequence : sequences) {
			earliest = std::min(earliest, so_far.find(sequence));
		}
		if (earliest != std::string::npos) {
			return Expected{.text = so_far.substr(0, earliest), .stopped_at = index};
		}
	}
	return Expected{.text = so_far, .stopped_at = std::nullopt};
}

/**
 * For random sequences and random deltas, the scan releases exactly the oracle's text, reports the stop at the
 * oracle's delta, and never holds back as much as a whole sequence.
 */
[[nodiscard]] auto scan_matches_the_oracle(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	for (auto trial = 0; trial < 20000; ++trial) {
		auto sequences = std::vector<std::string>{};
		auto const count = std::uniform_int_distribution<std::size_t>{0, 3}(generator);
		auto longest = std::size_t{1};
		for (auto index = std::size_t{0}; index < count; ++index) {
			sequences.push_back(word(generator, 4));
			longest = std::max(longest, sequences.back().size());
		}
		auto deltas = std::vector<std::string>{};
		auto const pieces = std::uniform_int_distribution<std::size_t>{0, 8}(generator);
		for (auto index = std::size_t{0}; index < pieces; ++index) {
			deltas.push_back(std::uniform_int_distribution{0, 5}(generator) == 0 ? std::string{} : word(generator, 6));
		}
		auto const oracle = expected(sequences, deltas);
		auto parsed = openai::StopSequences::parse(sequences);
		if (!parsed) {
			return false;
		}
		auto scan = openai::StopScan{*std::move(parsed)};
		auto released = std::string{};
		auto stopped_at = std::optional<std::size_t>{};
		for (auto index = std::size_t{0}; index < deltas.size() && !stopped_at; ++index) {
			auto scanned = scan.feed(deltas[index]);
			if (auto const stopped = engine::core::alternative<openai::Stopped>(scanned)) {
				released += stopped->text;
				stopped_at = index;
			} else {
				released += std::get<openai::Continuing>(scanned).text;
				if (scan.held() >= longest) {
					return false;
				}
			}
		}
		if (!stopped_at) {
			released += std::move(scan).finish();
		}
		if (released != oracle.text || stopped_at != oracle.stopped_at) {
			std::println("seed {} trial {}: released {:?}, expected {:?}", seed, trial, released, oracle.text);
			return false;
		}
	}
	return true;
}

/** Every sequence is 1 to max_stop_bytes: the empty one and the one byte too long are refused, by position. */
[[nodiscard]] auto sequences_are_bounded() -> bool {
	auto const empty = openai::StopSequences::parse({"x", ""});
	auto const long_one = openai::StopSequences::parse({std::string(openai::max_stop_bytes + 1, 'x')});
	auto const longest = openai::StopSequences::parse({std::string(openai::max_stop_bytes, 'x')});
	return !empty && empty.error() == openai::StopError{.fault = openai::StopFault::Empty, .index = 1} && !long_one &&
	       long_one.error() == openai::StopError{.fault = openai::StopFault::TooLong, .index = 0} && longest.has_value();
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the stop scan agrees with a whole-text oracle (seed 1)", .holds = scan_matches_the_oracle(1)},
	    engine::testing::Case{.name = "the stop scan agrees with a whole-text oracle (seed 2)", .holds = scan_matches_the_oracle(2)},
	    engine::testing::Case{.name = "stop sequences are 1 to max_stop_bytes", .holds = sequences_are_bounded()},
	};
	return engine::testing::report(cases);
}
