// SPDX-License-Identifier: AGPL-3.0-only
export module engine.throughput:measure;

import std;
import engine.core;
import engine.net;
import :api;
import :error;

/**
 * The arithmetic of the measurement, over facts alone: the instants a client saw and the counts the server reported.
 * Nothing here reads a clock or a socket, so every figure is a function of what a run recorded.
 */
namespace engine::throughput {

/** A point on the monotonic clock the client stamps arrivals with. */
export using Instant = net::Deadline;

/** One stream of a wave, as the client saw it. */
export struct StreamFacts {
	/** The prompt's place in the workload. */
	std::size_t prompt;
	/** When the client began sending the request, its connection already open. */
	Instant sent;
	/** When the first event that carries output arrived: the first evidence of a generated token. */
	Instant first;
	/** When the terminal event arrived. */
	Instant last;
	Usage usage;
	Ending ending;

	[[nodiscard]] constexpr auto operator==(StreamFacts const&) const -> bool = default;
};

/** One wave: its streams, and the engine's counters at rest before and after it. */
export struct WaveFacts {
	std::vector<StreamFacts> streams;
	Counters before;
	Counters after;

	[[nodiscard]] auto operator==(WaveFacts const&) const -> bool = default;
};

/** One run of the workload: its waves in order. */
export struct RunFacts {
	std::vector<WaveFacts> waves;

	[[nodiscard]] auto operator==(RunFacts const&) const -> bool = default;
};

/** A stream's figures. */
export struct StreamMeasure {
	std::size_t prompt;
	/** From the request's sending to the first output event. */
	std::chrono::nanoseconds first_token;
	/** From the first output event to the terminal one: the span the stream's tokens are timed over. */
	std::chrono::nanoseconds span;
	/** Output tokens over the span. The tokens of the first event are counted and its cycle is not, a cycle's worth. */
	double tokens_per_second;
	Usage usage;
	Ending ending;

	[[nodiscard]] constexpr auto operator==(StreamMeasure const&) const -> bool = default;
};

/** A run's figures: its streams', and the aggregate over its waves. */
export struct RunMeasure {
	std::vector<StreamMeasure> streams;
	/** Every stream's usage, summed. */
	Usage usage;
	/** The waves' decode windows, summed: in each, from the first output event of any stream to the last terminal one. */
	std::chrono::nanoseconds window;
	/** Output tokens over the window: the aggregate rate. */
	double tokens_per_second;
	/** What the engine's counters advanced by over the run's waves. */
	Counters cycles;
	/**
	 * Output tokens per decode cycle, over all streams: tau. At one stream it is the tokens a cycle yields; at N streams
	 * a cycle yields this many between them.
	 */
	double tokens_per_cycle;
	/** The window over the cycles: one cycle's time, prefill that ran between cycles included. */
	std::chrono::duration<double, std::milli> cycle;

	[[nodiscard]] auto operator==(RunMeasure const&) const -> bool = default;
};

[[nodiscard]] constexpr auto seconds(std::chrono::nanoseconds span) -> double {
	return std::chrono::duration<double>{span}.count();
}

[[nodiscard]] constexpr auto unmeasurable(Unmeasured why, std::size_t run) -> std::unexpected<Unmeasurable> {
	return std::unexpected(Unmeasurable{.why = why, .run = run});
}

/** A wave's window: from the first output event of any of its streams to the last terminal event. */
[[nodiscard]] constexpr auto window_of(WaveFacts const& wave) -> std::chrono::nanoseconds {
	// A wave runs its mode's streams, never none.
	core::invariant(!wave.streams.empty());
	auto const first = std::ranges::min(wave.streams, {}, &StreamFacts::first).first;
	auto const last = std::ranges::max(wave.streams, {}, &StreamFacts::last).last;
	return last - first;
}

/** The figures of run `index`; the refusal names it. */
export [[nodiscard]] constexpr auto measure(RunFacts const& run, std::size_t index) -> core::Result<RunMeasure, Unmeasurable> {
	auto out = RunMeasure{
	    .streams = {},
	    .usage = Usage{.input_tokens = 0, .cached_tokens = 0, .output_tokens = 0, .reasoning_tokens = 0},
	    .window = std::chrono::nanoseconds::zero(),
	    .tokens_per_second = 0.0,
	    .cycles = Counters{.cycles = 0, .drafted_tokens = 0, .accepted_tokens = 0},
	    .tokens_per_cycle = 0.0,
	    .cycle = std::chrono::duration<double, std::milli>::zero(),
	};
	for (auto const& wave : run.waves) {
		for (auto const& stream : wave.streams) {
			// The clock is monotonic and the stamps are taken in this order.
			core::invariant(stream.sent <= stream.first && stream.first <= stream.last);
			if (stream.first == stream.last) {
				return unmeasurable(Unmeasured::NoSpan, index);
			}
			if (stream.usage.output_tokens == 0) {
				return unmeasurable(Unmeasured::NoTokens, index);
			}
			auto const span = std::chrono::nanoseconds{stream.last - stream.first};
			out.streams.push_back(StreamMeasure{
			    .prompt = stream.prompt,
			    .first_token = stream.first - stream.sent,
			    .span = span,
			    .tokens_per_second = static_cast<double>(stream.usage.output_tokens) / seconds(span),
			    .usage = stream.usage,
			    .ending = stream.ending,
			});
			out.usage.input_tokens += stream.usage.input_tokens;
			out.usage.cached_tokens += stream.usage.cached_tokens;
			out.usage.output_tokens += stream.usage.output_tokens;
			out.usage.reasoning_tokens += stream.usage.reasoning_tokens;
		}
		// Counters only grow within one boot: a count that stood still or fell is another engine's or another boot's.
		if (wave.after.cycles <= wave.before.cycles || wave.after.drafted_tokens < wave.before.drafted_tokens ||
		    wave.after.accepted_tokens < wave.before.accepted_tokens) {
			return unmeasurable(Unmeasured::NoCycles, index);
		}
		out.window += window_of(wave);
		out.cycles.cycles += wave.after.cycles - wave.before.cycles;
		out.cycles.drafted_tokens += wave.after.drafted_tokens - wave.before.drafted_tokens;
		out.cycles.accepted_tokens += wave.after.accepted_tokens - wave.before.accepted_tokens;
	}
	// A run is its workload's waves, never none; each added a span and a cycle.
	core::invariant(!run.waves.empty());
	auto const tokens = static_cast<double>(out.usage.output_tokens);
	auto const cycles = static_cast<double>(out.cycles.cycles);
	out.tokens_per_second = tokens / seconds(out.window);
	out.tokens_per_cycle = tokens / cycles;
	out.cycle = std::chrono::duration<double, std::milli>{out.window} / cycles;
	return out;
}

/** The figures of every run, in order. */
export [[nodiscard]] constexpr auto measure(std::span<RunFacts const> runs) -> core::Result<std::vector<RunMeasure>, Unmeasurable> {
	auto out = std::vector<RunMeasure>{};
	out.reserve(runs.size());
	for (auto const [index, run] : std::views::enumerate(runs)) {
		auto measured = measure(run, core::narrow<std::size_t>(index));
		if (!measured) {
			return std::unexpected(measured.error());
		}
		out.push_back(std::move(*measured));
	}
	return out;
}

/** The runs judged together: which was the best, and how far apart they lay. */
export struct Verdict {
	/** The run with the highest aggregate rate, the least time per token; the earliest of equals. */
	std::size_t best;
	double slowest;
	double fastest;

	/** How far below the best the slowest run lay, as a share of the best. */
	[[nodiscard]] constexpr auto spread() const -> double {
		return (fastest - slowest) / fastest;
	}

	[[nodiscard]] constexpr auto operator==(Verdict const&) const -> bool = default;
};

/** Best of n, by the aggregate rate. */
export [[nodiscard]] constexpr auto judge(std::span<RunMeasure const> runs) -> Verdict {
	// The protocol runs the workload at least once.
	core::invariant(!runs.empty());
	auto const [slowest, fastest] = std::ranges::minmax_element(runs, {}, &RunMeasure::tokens_per_second);
	// minmax_element names the last of equal maxima; the earliest of them is the first run that reached the rate.
	auto const best = std::ranges::find(runs, fastest->tokens_per_second, &RunMeasure::tokens_per_second);
	return Verdict{
	    .best = core::narrow<std::size_t>(best - runs.begin()),
	    .slowest = slowest->tokens_per_second,
	    .fastest = fastest->tokens_per_second,
	};
}

}
