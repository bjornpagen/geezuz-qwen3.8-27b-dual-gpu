// SPDX-License-Identifier: AGPL-3.0-only
//
// The cross-card and host benchmarks: the nvlink flag handoff, partials pushed (and pulled) across the bridge, the
// cost of a system-scope release beside other traffic, the link facts the engine's start proves, and the host side:
// rows fetched from mapped memory and the resident mailbox's round trip.
export module engine.ubench:link;

import std;
import engine.core;
import engine.device;
import engine.model;
import engine.ubench_device;
import :outcome;
import :protocol;

namespace engine::ubench {

namespace ub = engine::device::ubench;
namespace kernels = engine::ubench_device;
namespace shapes = engine::device::ubench::link;

using model::Card;

/** The best initiator time of the flag ping-pong under one flag flavor and wait. */
export struct FlagReading {
	std::string_view kernel;
	shapes::Flavor flavor;
	ub::Wait wait;
	double nanoseconds;
};

[[nodiscard]] constexpr auto flavor_label(shapes::Flavor flavor) -> std::string_view {
	switch (flavor) {
	case shapes::Flavor::ReleaseAcquire:
		return "st.release.sys / ld.acquire.sys";
	case shapes::Flavor::Relaxed:
		return "st.relaxed.sys / ld.relaxed.sys, unordered";
	case shapes::Flavor::Fenced:
		return "fence.sc.sys + st.relaxed.sys / ld.relaxed.sys + fence.sc.sys";
	}
	std::unreachable();
}

export [[nodiscard]] auto evaluate_peer_flag(std::span<FlagReading const> readings) -> Result<Outcome> {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		if (!(reading.nanoseconds > 0.0)) {
			return no_reading(reading.kernel, "the initiator timed nothing");
		}
		outcome.readings.push_back({.label = std::format("one-way flag handoff card 0 <-> card 1, {}, {}", flavor_label(reading.flavor),
		                                                 reading.wait == ub::Wait::Backoff ? "production backoff" : "spinning"),
		                            .value = reading.nanoseconds / static_cast<double>(shapes::flag_iterations) / 2.0,
		                            .unit = Unit::Nanoseconds});
	}
	outcome.notes.emplace_back("the relaxed flags are the hardware's floor; release over relaxed is what ordering costs, fenced the "
	                           "protocol the engine does not use");
	return outcome;
}

/** The flag variants measured, in order: the engine's own (with its backoff, then spinning), and the two comparisons. */
template<shapes::Flavor flavor_, ub::Wait wait_>
struct FlagVariant {
	static constexpr shapes::Flavor flavor = flavor_;
	static constexpr ub::Wait wait = wait_;
	using Ping = kernels::FlagKernel<kernels::FlagEnd::Ping, flavor, wait>;
	using Pong = kernels::FlagKernel<kernels::FlagEnd::Pong, flavor, wait>;
};

using FlagVariants = std::tuple<FlagVariant<shapes::Flavor::ReleaseAcquire, ub::Wait::Backoff>, FlagVariant<shapes::Flavor::ReleaseAcquire, ub::Wait::Spin>,
                                FlagVariant<shapes::Flavor::Relaxed, ub::Wait::Spin>, FlagVariant<shapes::Flavor::Fenced, ub::Wait::Spin>>;

/** Launches the answering end on card 1, then the initiator on card 0, and waits for both. */
template<class Variant, class PingProgram, class PongProgram>
[[nodiscard]] auto flag_round(BenchView const& bench, PingProgram const& ping, PongProgram const& pong, std::array<Readback<unsigned long long, 1>, model::cards>& flags,
                              Readback<unsigned long long, 1>& nanoseconds, kernels::Nanoseconds const& answer_time) -> Result<double> {
	using Ping = Variant::Ping;
	using Pong = Variant::Pong;
	auto const zero = bench.card(Card::Zero);
	auto const one = bench.card(Card::One);
	auto& zero_flag = flags.at(0);
	auto& one_flag = flags.at(1);
	return zero_flag.clear(zero)
	    .and_then([&] {
		    return one_flag.clear(one);
	    })
	    .and_then([&] {
		    return lift(kernels::launch<Pong>(one.context, pong, one.stream, one_flag.device(), zero_flag.device(), answer_time));
	    })
	    .and_then([&] {
		    return lift(kernels::launch<Ping>(zero.context, ping, zero.stream, zero_flag.device(), one_flag.device(), nanoseconds.device()));
	    })
	    .and_then([&] {
		    return finish(zero, Ping::entry);
	    })
	    .and_then([&] {
		    return finish(one, Pong::entry);
	    })
	    .and_then([&] {
		    return nanoseconds.fetch(zero);
	    })
	    .transform([](std::span<unsigned long long const, 1> took) {
		    return static_cast<double>(took.front());
	    });
}

export [[nodiscard]] auto peer_flag(BenchView const& bench) -> Result<Outcome> {
	auto const& zero = bench.card(Card::Zero).context;
	auto const& one = bench.card(Card::One).context;
	auto const ping = std::apply(
	    [&](auto... variants) {
		    return lift(zero.load<typename decltype(variants)::Ping...>());
	    },
	    FlagVariants{});
	auto const pong = std::apply(
	    [&](auto... variants) {
		    return lift(one.load<typename decltype(variants)::Pong...>());
	    },
	    FlagVariants{});
	auto zero_flag = Readback<unsigned long long, 1>::create(zero);
	auto one_flag = Readback<unsigned long long, 1>::create(one);
	auto nanoseconds = Readback<unsigned long long, 1>::create(zero);
	auto const answer_time = scratch<unsigned long long, 1>(one);
	if (auto const failed = first_failure(ping, pong, zero_flag, one_flag, nanoseconds, answer_time)) {
		return std::unexpected(*failed);
	}
	auto flags = std::array{std::move(*zero_flag), std::move(*one_flag)};
	auto readings = std::vector<FlagReading>{};
	auto const ran = for_each_shape<FlagVariants>([&](auto variant) -> Result<void> {
		using Variant = decltype(variant);
		auto best = std::numeric_limits<double>::infinity();
		for (auto launch = 0; launch < warmup_launches + timed_launches; ++launch) {
			auto const took = flag_round<Variant>(bench, *ping, *pong, flags, *nanoseconds, *answer_time);
			if (!took) {
				return std::unexpected(took.error());
			}
			if (launch >= warmup_launches) {
				best = std::min(best, *took);
			}
		}
		readings.push_back({.kernel = Variant::Ping::entry, .flavor = Variant::flavor, .wait = Variant::wait, .nanoseconds = best});
		return {};
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	return evaluate_peer_flag(readings);
}

/** One exchange shape's best push (receiver-timed, nanoseconds) and best pull (event-timed, milliseconds). */
export struct ExchangePoint {
	int rows;
	int senders;
	double push_nanoseconds;
	double pull_milliseconds;
};

/** The bytes of one item of `rows` rows. */
[[nodiscard]] constexpr auto item_bytes(int rows) -> double {
	return static_cast<double>(rows) * shapes::row_chunks * 16.0;
}

export [[nodiscard]] auto evaluate_peer_push(std::span<ExchangePoint const> points) -> Result<Outcome> {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& point : points) {
		if (!(point.push_nanoseconds > 0.0)) {
			return no_reading(kernels::ReceiveKernel<4>::entry, "the receiver saw the first and last items at one time");
		}
		auto const label = std::format("{} row(s) of 20 KiB per item, {} sending blocks", point.rows, point.senders);
		// The receiver's clock runs from the first item's arrival, so the first item is not in the interval.
		auto const pushed = item_bytes(point.rows) * (shapes::items - 1) / point.push_nanoseconds;
		auto const pulled = gigabytes_per_second(item_bytes(point.rows) * shapes::items, point.pull_milliseconds);
		outcome.readings.push_back({.label = label + ", pushed card 0 -> card 1", .value = pushed, .unit = Unit::GigabytesPerSecond});
		outcome.readings.push_back({.label = label + ", pulled by card 1 from card 0", .value = pulled, .unit = Unit::GigabytesPerSecond});
	}
	return outcome;
}

template<class Shape>
using PushOf = kernels::PushKernel<Shape>;
template<class Shape>
using PullOf = kernels::PullKernel<Shape>;

template<template<class> class Kernel>
[[nodiscard]] auto load_exchange(device::Context const& context) {
	return std::apply(
	    [&](auto... rows) {
		    return std::apply(
		        [&](auto... shapes_of_all_rows) {
			        return context.load<Kernel<decltype(shapes_of_all_rows)>...>();
		        },
		        std::tuple_cat(decltype(rows){}...));
	    },
	    kernels::PushSweep{});
}

export [[nodiscard]] auto peer_push(BenchView const& bench) -> Result<Outcome> {
	auto const zero = bench.card(Card::Zero);
	auto const one = bench.card(Card::One);
	auto const pushes = lift(load_exchange<PushOf>(zero.context));
	auto const pulls = lift(load_exchange<PullOf>(one.context));
	auto const receivers = lift(one.context.load<kernels::ReceiveKernel<4>, kernels::ReceiveKernel<16>, kernels::ReceiveKernel<64>>());
	auto const receive = scratch<ub::Chunk, shapes::receive_chunks>(one.context);
	auto const send = scratch<ub::Chunk, shapes::receive_chunks>(zero.context);
	auto const sinks = scratch<unsigned, shapes::max_senders>(one.context);
	auto flags = Readback<unsigned long long, shapes::max_senders>::create(one.context);
	auto go = Readback<unsigned long long, 1>::create(zero.context);
	auto reading = Readback<shapes::ReceiveReading, 1>::create(one.context);
	if (auto const failed = first_failure(pushes, pulls, receivers, receive, send, sinks, flags, go, reading)) {
		return std::unexpected(*failed);
	}
	auto points = std::vector<ExchangePoint>{};
	auto const ran = for_each_shape<kernels::PushSweep>([&](auto row) {
		return for_each_shape<decltype(row)>([&](auto shape) -> Result<void> {
			using Shape = decltype(shape);
			using Receive = kernels::ReceiveKernel<Shape::senders>;
			auto best = std::numeric_limits<double>::infinity();
			for (auto launch = 0; launch < warmup_launches + timed_launches; ++launch) {
				auto const read = flags->clear(one)
				                      .and_then([&] {
					                      return go->clear(zero);
				                      })
				                      .and_then([&] {
					                      return lift(kernels::launch<Receive>(one.context, *receivers, one.stream, flags->device(), go->device(), reading->device()));
				                      })
				                      .and_then([&] {
					                      return lift(kernels::launch<PushOf<Shape>>(zero.context, *pushes, zero.stream, *receive, flags->device(), go->device()));
				                      })
				                      .and_then([&] {
					                      return finish(zero, PushOf<Shape>::entry);
				                      })
				                      .and_then([&] {
					                      return finish(one, Receive::entry);
				                      })
				                      .and_then([&] {
					                      return reading->fetch(one);
				                      });
				if (!read) {
					return std::unexpected(read.error());
				}
				if (launch >= warmup_launches) {
					best = std::min(best, static_cast<double>(read->front().last_ns - read->front().first_ns));
				}
			}
			return best_time<PullOf<Shape>>(one, *pulls, nothing, *send, *sinks).transform([&](double milliseconds) {
				points.push_back({.rows = Shape::rows, .senders = Shape::senders, .push_nanoseconds = best, .pull_milliseconds = milliseconds});
			});
		});
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	return evaluate_peer_push(points);
}

/** One card's least cycles for the timed releases beside one kind of background traffic. */
export struct ReleaseReading {
	Card card;
	shapes::Background background;
	double cycles;
};

[[nodiscard]] constexpr auto background_name(shapes::Background background) -> std::string_view {
	switch (background) {
	case shapes::Background::Idle:
		return "nothing else running";
	case shapes::Background::PeerStores:
		return "81 SMs storing to the peer";
	case shapes::Background::HostWrites:
		return "81 SMs storing to mapped host memory";
	}
	std::unreachable();
}

export [[nodiscard]] auto evaluate_release_cost(std::span<ReleaseReading const> readings, Calibration calibration) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& reading : readings) {
		outcome.readings.push_back({.label = std::format("{}, one weak store then st.release.sys, {}", on(reading.card), background_name(reading.background)),
		                            .value = calibration.nanoseconds(reading.card, reading.cycles / static_cast<double>(shapes::release_iterations)),
		                            .unit = Unit::Nanoseconds});
	}
	return outcome;
}

template<shapes::Background background>
using ReleaseOf = kernels::ReleaseKernel<background>;

export [[nodiscard]] auto release_cost(BenchView const& bench) -> Result<Outcome> {
	auto readings = std::vector<ReleaseReading>{};
	auto const ran = for_each_card([&](Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const& peer = bench.card(device::peer(card)).context;
		auto const program = lift(view.context.load<ReleaseOf<shapes::Background::Idle>, ReleaseOf<shapes::Background::PeerStores>,
		                                             ReleaseOf<shapes::Background::HostWrites>>());
		auto const flag = scratch<unsigned long long, 1>(view.context);
		auto const release_scratch = scratch<ub::Chunk, ub::sms>(view.context);
		auto const peer_slots = scratch<ub::Chunk, static_cast<std::size_t>(ub::sms) * shapes::background_chunks>(peer);
		auto const host = lift(view.context.allocate_mapped<ub::Chunk, static_cast<std::size_t>(ub::sms) * shapes::background_chunks>());
		auto stop = Readback<unsigned long long, 1>::create(view.context);
		auto cycles = Readback<unsigned long long, 1>::create(view.context);
		if (auto const failed = first_failure(program, flag, release_scratch, peer_slots, host, stop, cycles)) {
			return std::unexpected(*failed);
		}
		auto const measure = [&]<shapes::Background background> -> Result<void> {
			auto least = std::numeric_limits<double>::infinity();
			for (auto launch = 0; launch < warmup_launches + timed_launches; ++launch) {
				auto const counted = stop->clear(view)
				                         .and_then([&] {
					                         return run_once<ReleaseOf<background>>(view, *program, *flag, *release_scratch, *peer_slots, *host, stop->device(),
					                                                                cycles->device());
				                         })
				                         .and_then([&] {
					                         return cycles->fetch(view);
				                         });
				if (!counted) {
					return std::unexpected(counted.error());
				}
				if (launch >= warmup_launches) {
					least = std::min(least, static_cast<double>(counted->front()));
				}
			}
			if (!(least > 0.0)) {
				return no_reading(ReleaseOf<background>::entry, "the timed releases counted no cycles");
			}
			readings.push_back({.card = card, .background = background, .cycles = least});
			return {};
		};
		return measure.template operator()<shapes::Background::Idle>()
		    .and_then([&] {
			    return measure.template operator()<shapes::Background::PeerStores>();
		    })
		    .and_then([&] {
			    return measure.template operator()<shapes::Background::HostWrites>();
		    });
	});
	return ran.transform([&] {
		return evaluate_release_cost(readings, bench.calibration);
	});
}

/** What the management library and the driver say about one card's links and its path to the peer. */
export struct LinkReading {
	Card card;
	std::array<device::NvlinkLinkReading, device::nvlink_links> links;
	device::PeerFacts peer;
};

/** The nvlink version a 3090's links report as NVML numbers it; older drivers report the bare major version 3. */
inline constexpr auto expected_version = device::NvlinkVersion::V3_0;

/**
 * The link facts. The engine's start already proved every link up and joined to the peer and the peer mapped both
 * ways, or the runner would not be running: that is the verdict. Versions are shown raw, because NVML's numbering of
 * them is inferred and bring-up confirms it; native atomics are shown because the engine must not need them.
 */
export [[nodiscard]] auto evaluate_link_health(std::span<LinkReading const> readings) -> Outcome {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Holds, .notes = {}};
	for (auto const& reading : readings) {
		auto const card = reading.card;
		for (auto link = std::size_t{0}; link < reading.links.size(); ++link) {
			auto const& found = reading.links.at(link);
			outcome.readings.push_back({.label = std::format("{}, link {} version, raw NVML value", on(card), link),
			                            .value = static_cast<double>(std::to_underlying(found.version)),
			                            .unit = Unit::Count});
			if (found.megabytes_per_second) {
				outcome.readings.push_back({.label = std::format("{}, link {} speed", on(card), link),
				                            .value = static_cast<double>(*found.megabytes_per_second) / 1000.0,
				                            .unit = Unit::GigabytesPerSecond});
			} else {
				outcome.notes.push_back(std::format("{}: NVML has no speed field for link {}", on(card), link));
			}
			if (found.version != expected_version) {
				outcome.notes.push_back(std::format("{}: link {} reads version {}, not {} (nvlink 3.0 as nvml.h numbers it); confirm at bring-up", on(card), link,
				                                    std::to_underlying(found.version), std::to_underlying(expected_version)));
			}
		}
		outcome.readings.push_back({.label = std::format("{}, peer path performance rank", on(card)), .value = static_cast<double>(reading.peer.performance_rank), .unit = Unit::Count});
		outcome.readings.push_back({.label = std::format("{}, native peer atomics (1 yes, 0 no)", on(card)), .value = reading.peer.native_atomics ? 1.0 : 0.0, .unit = Unit::Count});
	}
	outcome.notes.emplace_back("every link up and joined to the peer, and the peer mapped both ways, were proved by the engine's start; "
	                           "the engine uses no peer atomics whatever the attribute says");
	return outcome;
}

export [[nodiscard]] auto link_health(BenchView const& bench) -> Result<Outcome> {
	auto readings = std::vector<LinkReading>{};
	auto const ran = for_each_card([&](Card card) -> Result<void> {
		auto const links = lift(bench.devices.nvlinks(card));
		auto const peer = lift(bench.devices.peer_facts(card));
		if (auto const failed = first_failure(links, peer)) {
			return std::unexpected(*failed);
		}
		readings.push_back({.card = card, .links = *links, .peer = *peer});
		return {};
	});
	return ran.transform([&] {
		return evaluate_link_health(readings);
	});
}

/** One fetch shape's best time on one card. */
export struct FetchPoint {
	Card card;
	int rows;
	double nanoseconds;
};

/** The resident mailbox's round trips on one card, host-timed, in nanoseconds. */
export struct MailboxReading {
	Card card;
	std::vector<double> round_trips;
};

export [[nodiscard]] auto evaluate_host_fetch(std::span<FetchPoint const> fetches, std::span<MailboxReading const> mailboxes) -> Result<Outcome> {
	auto outcome = Outcome{.readings = {}, .verdict = Verdict::Measured, .notes = {}};
	for (auto const& fetch : fetches) {
		if (!(fetch.nanoseconds > 0.0)) {
			return no_reading(kernels::FetchKernel<1>::entry, "a fetch took no time");
		}
		auto const label = std::format("{}, {} mapped row(s) of 10 KiB", on(fetch.card), fetch.rows);
		outcome.readings.push_back({.label = label + ", fetch", .value = fetch.nanoseconds / 1000.0, .unit = Unit::Microseconds});
		outcome.readings.push_back({.label = label + ", bandwidth", .value = fetch.rows * shapes::table_row_chunks * 16.0 / fetch.nanoseconds, .unit = Unit::GigabytesPerSecond});
	}
	for (auto const& mailbox : mailboxes) {
		core::invariant(!mailbox.round_trips.empty());
		outcome.readings.push_back({.label = std::format("{}, mailbox round trip host -> card -> host, best", on(mailbox.card)),
		                            .value = std::ranges::min(mailbox.round_trips) / 1000.0,
		                            .unit = Unit::Microseconds});
		outcome.readings.push_back({.label = std::format("{}, mailbox round trip, median", on(mailbox.card)), .value = median(mailbox.round_trips) / 1000.0, .unit = Unit::Microseconds});
	}
	outcome.notes.emplace_back("the host-to-device flag latency is about half a round trip; rerun under the other numa node to compare");
	return outcome;
}

/** Mailbox round trips per card. */
inline constexpr int mailbox_rounds = 1000;
/** A resident kernel answers in microseconds; one that has not answered in this long is not running. */
inline constexpr std::chrono::seconds answer_deadline = std::chrono::seconds{2};

using Mailbox = device::Ring<shapes::MailboxRequest, shapes::MailboxAnswer, shapes::mailbox_capacity>;

/** Waits for the oldest ticket's completion until `until`; nothing if it never came. */
[[nodiscard]] auto collect_until(Mailbox& mailbox, std::chrono::steady_clock::time_point until) -> std::optional<device::Completion<shapes::MailboxAnswer>> {
	while (std::chrono::steady_clock::now() < until) {
		if (auto completion = mailbox.try_collect()) {
			return completion;
		}
	}
	return std::nullopt;
}

/** Runs the resident mailbox on one card for `mailbox_rounds` round trips, then asks it to exit. */
[[nodiscard]] auto mailbox_round_trips(CardView view) -> Result<std::vector<double>> {
	auto const program = lift(view.context.load<kernels::MailboxKernel>());
	auto mailbox = lift(Mailbox::create(view.context));
	if (auto const failed = first_failure(program, mailbox)) {
		return std::unexpected(*failed);
	}
	auto const slots = kernels::MailboxSlots{.address = mailbox->device_address()};
	if (auto launched = lift(kernels::launch<kernels::MailboxKernel>(view.context, *program, view.stream, slots)); !launched) {
		return std::unexpected(launched.error());
	}
	auto round_trips = std::vector<double>{};
	auto answered = true;
	for (auto round = 0; round < mailbox_rounds && answered; ++round) {
		auto const sent = std::chrono::steady_clock::now();
		auto const ticket = mailbox->try_submit({.sequence = static_cast<unsigned>(round)});
		core::invariant(ticket.has_value());
		auto const completion = collect_until(*mailbox, sent + answer_deadline);
		answered = completion.has_value();
		if (answered) {
			round_trips.push_back(std::chrono::duration<double, std::nano>{std::chrono::steady_clock::now() - sent}.count());
		}
	}
	// The exit ticket always has a slot: at most one ticket is ever outstanding beside it.
	auto const exit = mailbox->try_exit();
	core::invariant(exit.has_value());
	if (!answered) {
		return std::unexpected(UbenchError{.reason = Hung{.kernel = kernels::MailboxKernel::entry}});
	}
	if (!collect_until(*mailbox, std::chrono::steady_clock::now() + answer_deadline)) {
		return std::unexpected(UbenchError{.reason = Hung{.kernel = kernels::MailboxKernel::entry}});
	}
	return finish(view, kernels::MailboxKernel::entry).transform([&] {
		return std::move(round_trips);
	});
}

export [[nodiscard]] auto host_fetch(BenchView const& bench) -> Result<Outcome> {
	auto fetches = std::vector<FetchPoint>{};
	auto mailboxes = std::vector<MailboxReading>{};
	auto const ran = for_each_card([&](Card card) -> Result<void> {
		auto const view = bench.card(card);
		auto const program = std::apply(
		    [&](auto... kernels_of_sweep) {
			    return lift(view.context.load<decltype(kernels_of_sweep)...>());
		    },
		    kernels::FetchSweep{});
		auto const table = lift(view.context.allocate_mapped<ub::Chunk, shapes::table_chunks>());
		auto reading = Readback<shapes::FetchReading, 1>::create(view.context);
		if (auto const failed = first_failure(program, table, reading)) {
			return std::unexpected(*failed);
		}
		auto const fetched = for_each_shape<kernels::FetchSweep>([&](auto kernel) -> Result<void> {
			using K = decltype(kernel);
			auto best = std::numeric_limits<double>::infinity();
			for (auto launch = 0; launch < warmup_launches + timed_launches; ++launch) {
				auto const read = run_once<K>(view, *program, *table, reading->device()).and_then([&] {
					return reading->fetch(view);
				});
				if (!read) {
					return std::unexpected(read.error());
				}
				if (launch >= warmup_launches) {
					best = std::min(best, static_cast<double>(read->front().best_ns));
				}
			}
			fetches.push_back({.card = card, .rows = K::Shape::rows, .nanoseconds = best});
			return {};
		});
		return fetched.and_then([&] {
			return mailbox_round_trips(view).transform([&](std::vector<double> round_trips) {
				mailboxes.push_back({.card = card, .round_trips = std::move(round_trips)});
			});
		});
	});
	if (!ran) {
		return std::unexpected(ran.error());
	}
	return evaluate_host_fetch(fetches, mailboxes);
}

}
