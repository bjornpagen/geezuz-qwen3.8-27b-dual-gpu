// SPDX-License-Identifier: AGPL-3.0-only
//
// The measurement protocol every driver follows: warm-up launches first, then the best of a fixed number of timed
// launches, each awaited before an absolute deadline; results come back from device memory through mapped memory
// once the kernel that wrote them has finished. The protocol is fixed, not a choice: the same counts every run, so
// runs compare.
export module engine.ubench:protocol;

import std;
import engine.core;
import engine.device;
import engine.model;
import engine.ubench_device;
import :outcome;

namespace engine::ubench {

/** Launches before timing: the jit, the caches and the clocks settle. */
export inline constexpr int warmup_launches = 2;
/** Timed launches; a benchmark reports the best. */
export inline constexpr int timed_launches = 5;
/** How long a launch, a copy or a resident kernel's answer may take before the card counts as hung. */
export inline constexpr std::chrono::seconds deadline = std::chrono::seconds{60};

/** The SM clock each card ran at while calibrating, which turns cycle counts into time. */
export struct Calibration {
	std::array<double, model::cards> megahertz;

	[[nodiscard]] auto nanoseconds(model::Card card, double cycles) const -> double {
		return cycles * 1000.0 / megahertz.at(std::to_underlying(card));
	}
};

/** A view of one card for the duration of one synchronous driver call: its context and the stream work goes on. */
export struct CardView {
	model::Card card;
	device::Context const& context;
	device::Stream const& stream;
};

/** The machine as one run of drivers sees it: both cards, a stream on each, and the calibrated clocks. */
export struct BenchView {
	device::Devices const& devices;
	std::array<device::Stream, model::cards> const& streams;
	Calibration calibration;

	[[nodiscard]] auto card(model::Card card) const -> CardView {
		return {.card = card, .context = devices.context(card), .stream = streams.at(std::to_underlying(card))};
	}
};

/** Waits for everything enqueued on the view's stream so far; a stream still busy at the deadline is a hung `kernel`. */
export [[nodiscard]] auto finish(CardView view, std::string_view kernel) -> Result<void> {
	auto const marker = view.context.create_event();
	if (!marker) {
		return std::unexpected(UbenchError{.reason = marker.error()});
	}
	if (auto recorded = marker->record(view.stream); !recorded) {
		return std::unexpected(UbenchError{.reason = recorded.error()});
	}
	auto const state = marker->wait_until(std::chrono::steady_clock::now() + deadline);
	if (!state) {
		return std::unexpected(UbenchError{.reason = state.error()});
	}
	if (*state == device::EventState::Pending) {
		return std::unexpected(UbenchError{.reason = Hung{.kernel = kernel}});
	}
	return {};
}

/**
 * Enqueues `K` on the view's stream: a kernel this component declares is built from the typed owners it takes; a
 * kernel another component declares takes its parameter struct as that component builds it.
 */
export template<class K, class... Ks, class... Arguments>
[[nodiscard]] auto enqueue(CardView view, device::Program<Ks...> const& program, Arguments const&... arguments) -> Result<void> {
	if constexpr (ubench_device::Typed<K, Arguments...>) {
		return lift(ubench_device::launch<K>(view.context, program, view.stream, arguments...));
	} else {
		return lift(view.context.launch<K>(program, view.stream, arguments...));
	}
}

/** Launches `K` once on the view's stream and waits for it. */
export template<class K, class... Ks, class... Arguments>
[[nodiscard]] auto run_once(CardView view, device::Program<Ks...> const& program, Arguments const&... arguments) -> Result<void> {
	return enqueue<K>(view, program, arguments...).and_then([&] {
		return finish(view, K::entry);
	});
}

/** One timed run of `Run`'s launches, in milliseconds between markers recorded around them. */
template<class Run, class... Ks, class... Arguments>
[[nodiscard]] auto timed_once(CardView view, device::Program<Ks...> const& program, Arguments const&... arguments) -> Result<double> {
	auto start = view.context.create_timed_event();
	auto end = view.context.create_timed_event();
	if (!start || !end) {
		return std::unexpected(UbenchError{.reason = !start ? start.error() : end.error()});
	}
	auto const enqueued = lift(start->record(view.stream))
	                          .and_then([&] {
		                          return Run::enqueue(view, program, arguments...);
	                          })
	                          .and_then([&] {
		                          return lift(end->record(view.stream));
	                          });
	if (!enqueued) {
		return std::unexpected(enqueued.error());
	}
	auto const state = end->wait_until(std::chrono::steady_clock::now() + deadline);
	if (!state) {
		return std::unexpected(UbenchError{.reason = state.error()});
	}
	if (*state == device::EventState::Pending) {
		return std::unexpected(UbenchError{.reason = Hung{.kernel = Run::entry}});
	}
	auto const elapsed = end->since(*start);
	if (!elapsed) {
		return std::unexpected(UbenchError{.reason = elapsed.error()});
	}
	// Both markers have completed, so the driver has their times.
	core::invariant(elapsed->has_value());
	return static_cast<double>((*elapsed)->count());
}

/** Kernels launched back to back on one stream with the same arguments, timed as one run; named by the first. */
template<class First, class... Rest>
struct Launches {
	static constexpr std::string_view entry = First::entry;

	template<class... Ks, class... Arguments>
	[[nodiscard]] static auto enqueue(CardView view, device::Program<Ks...> const& program, Arguments const&... arguments) -> Result<void> {
		auto result = ubench::enqueue<First>(view, program, arguments...);
		auto const next = [&]<class K> {
			result = result.and_then([&] {
				return ubench::enqueue<K>(view, program, arguments...);
			});
		};
		(next.template operator()<Rest>(), ...);
		return result;
	}
};

/**
 * The protocol for event-timed kernels: `warmup_launches` runs, then the best (shortest) of `timed_launches`, in
 * milliseconds. A run launches `First`, then each of `Rest`, with the same arguments. `between` runs before every run
 * (a reset of what the kernels consume) and must leave the stream ordered before it.
 */
export template<class First, class... Rest, class... Ks, class Between, class... Arguments>
[[nodiscard]] auto best_time(CardView view, device::Program<Ks...> const& program, Between between, Arguments const&... arguments) -> Result<double> {
	using Run = Launches<First, Rest...>;
	for (auto launch = 0; launch < warmup_launches; ++launch) {
		auto const ran = between()
		                     .and_then([&] {
			                     return Run::enqueue(view, program, arguments...);
		                     })
		                     .and_then([&] {
			                     return finish(view, Run::entry);
		                     });
		if (!ran) {
			return std::unexpected(ran.error());
		}
	}
	auto best = std::numeric_limits<double>::infinity();
	for (auto launch = 0; launch < timed_launches; ++launch) {
		auto const took = between().and_then([&] {
			return timed_once<Run>(view, program, arguments...);
		});
		if (!took) {
			return std::unexpected(took.error());
		}
		best = std::min(best, *took);
	}
	if (!(best > 0.0)) {
		return no_reading(Run::entry, "the timed launches took no time");
	}
	return best;
}

/** Nothing to reset between launches. */
export [[nodiscard]] auto nothing() -> Result<void> {
	return {};
}

/**
 * Device memory a kernel writes its results to, and the mapped host memory they are read back through. `clear` zeroes
 * the device side for a kernel that accumulates into it; `fetch` copies it back once the stream's work is done.
 */
export template<device::Wire T, std::size_t Count>
class Readback {
public:
	[[nodiscard]] static auto create(device::Context const& context) -> Result<Readback> {
		auto device = context.allocate<T, Count>();
		if (!device) {
			return std::unexpected(UbenchError{.reason = device.error()});
		}
		auto host = context.allocate_mapped<T, Count>();
		if (!host) {
			return std::unexpected(UbenchError{.reason = host.error()});
		}
		return Readback{std::move(*device), std::move(*host)};
	}

	[[nodiscard]] auto device() const -> device::DeviceArray<T, Count> const& {
		return device_;
	}

	/** Zeroes the device side, ordered before whatever the view's stream runs next. */
	[[nodiscard]] auto clear(CardView view) -> Result<void> {
		std::ranges::fill(host_.host(), T{});
		auto transfer = view.context.upload(view.stream, std::move(device_), std::move(host_), 0);
		return settle(std::move(transfer));
	}

	/** Writes `values` to the device side, ordered before whatever the view's stream runs next. */
	[[nodiscard]] auto put(CardView view, std::span<T const> values) -> Result<void> {
		core::invariant(values.size() == Count);
		std::ranges::copy(values, host_.host().begin());
		auto transfer = view.context.upload(view.stream, std::move(device_), std::move(host_), 0);
		return settle(std::move(transfer));
	}

	/** The device side's values once everything on the view's stream has finished; valid until the next call. */
	[[nodiscard]] auto fetch(CardView view) -> Result<std::span<T const, Count>> {
		auto transfer = view.context.download(view.stream, std::move(host_), std::move(device_));
		return settle(std::move(transfer)).transform([&] {
			return std::span<T const, Count>{host_.host()};
		});
	}

private:
	Readback(device::DeviceArray<T, Count> device, device::MappedArray<T, Count> host) : device_{std::move(device)}, host_{std::move(host)} {}

	/** Waits out a copy and takes both owners back. */
	template<class Destination, class Source>
	[[nodiscard]] auto settle(device::Transfer<Destination, Source> transfer) -> Result<void> {
		// A copy is only enqueued behind work already awaited, so it cannot outlast its deadline unless the card is
		// lost, and a transfer cannot be abandoned while it runs: that state ends the process.
		core::invariant(transfer.wait_until(std::chrono::steady_clock::now() + deadline) == device::EventState::Complete);
		auto delivered = transfer.try_finish();
		core::invariant(delivered.has_value());
		take(std::move(delivered->destination));
		take(std::move(delivered->source));
		if (!delivered->copied) {
			return std::unexpected(UbenchError{.reason = delivered->copied.error()});
		}
		return {};
	}

	void take(device::DeviceArray<T, Count> device) {
		device_ = std::move(device);
	}

	void take(device::MappedArray<T, Count> host) {
		host_ = std::move(host);
	}

	device::DeviceArray<T, Count> device_;
	device::MappedArray<T, Count> host_;
};

/** Allocates device memory a kernel only reads or only writes in place, whose contents the benchmark does not care for. */
export template<device::Wire T, std::size_t Count>
[[nodiscard]] auto scratch(device::Context const& context) -> Result<device::DeviceArray<T, Count>> {
	return lift(context.allocate<T, Count>());
}

/** The median of some values, which outliers (a block descheduled, a straggler SM) do not move. */
export [[nodiscard]] auto median(std::vector<double> values) -> double {
	core::invariant(!values.empty());
	auto const middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
	std::ranges::nth_element(values, middle);
	return *middle;
}

/** A card's memory clock now, as a reading and the peak it gives the 384-bit gddr6x bus. */
export struct MemoryState {
	device::MemoryClock clock;

	/** Two transfers per clock on 384 bits: 96 bytes per memory-clock cycle (9501 MHz gives 912 GB/s). */
	[[nodiscard]] auto peak_gigabytes_per_second() const -> double {
		return static_cast<double>(clock.megahertz) * 96.0 / 1000.0;
	}
};

export [[nodiscard]] auto memory_state(BenchView const& bench, model::Card card) -> Result<MemoryState> {
	return lift(bench.devices.memory_clock(card)).transform([](device::MemoryClock clock) {
		return MemoryState{.clock = clock};
	});
}

/**
 * Runs `step` on a value of every shape of `Sweep` (a tuple of shape types), in order, stopping at the first failure.
 * The shapes are empty compile-time descriptions, so a value of one only carries its type.
 */
export template<class Sweep, class Step>
[[nodiscard]] auto for_each_shape(Step step) -> Result<void> {
	auto result = Result<void>{};
	auto const one = [&](auto shape) {
		result = step(shape);
		return result.has_value();
	};
	std::apply(
	    [&](auto... shapes) {
		    std::ignore = (one(shapes) && ...);
	    },
	    Sweep{});
	return result;
}

/** Runs `step` for each card in rank order, stopping at the first failure. */
export template<class Step>
[[nodiscard]] auto for_each_card(Step step) -> Result<void> {
	for (auto const card : device::card_pair) {
		if (auto done = step(card); !done) {
			return done;
		}
	}
	return {};
}

/** Gigabytes per second for `bytes` moved in `milliseconds`. */
export [[nodiscard]] constexpr auto gigabytes_per_second(double bytes, double milliseconds) -> double {
	return bytes / (milliseconds * 1.0e6);
}

/** The first failure among some results, if any failed. */
export template<class... Results>
[[nodiscard]] auto first_failure(Results const&... results) -> std::optional<UbenchError> {
	auto failure = std::optional<UbenchError>{};
	auto const check = [&](auto const& result) {
		if (!failure && !result) {
			failure = result.error();
		}
	};
	(check(results), ...);
	return failure;
}

}
