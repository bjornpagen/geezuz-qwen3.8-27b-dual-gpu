// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.net;
import engine.pipeline;
import engine.pipeline_doubles;
import engine.testing;

namespace {

namespace net = engine::net;
namespace pipeline = engine::pipeline;
namespace doubles = engine::pipeline_doubles;

constexpr auto stages = std::to_array<net::Stage>({net::Stage::Socket, net::Stage::Option, net::Stage::Bind, net::Stage::Listen,
                                                   net::Stage::Resolve, net::Stage::Accept, net::Stage::Connect, net::Stage::Queue,
                                                   net::Stage::Watch, net::Stage::Wait, net::Stage::Signal});

constexpr auto causes = std::to_array<net::Cause>(
    {net::Cause::Exhausted, net::Cause::AddressInUse, net::Cause::Refused, net::Cause::Denied, net::Cause::Invalid});

/** Two reactor failures an operator must tell apart read differently, and each names the system's error number. */
[[nodiscard]] auto every_reactor_failure_reads_its_own() -> bool {
	auto seen = std::set<std::string>{};
	for (auto const stage : stages) {
		for (auto const cause : causes) {
			auto const code = std::int32_t{1000} + engine::core::narrow<std::int32_t>(seen.size());
			auto const text = net::describe(net::NetError{.stage = stage, .cause = cause, .code = code});
			if (!text.contains(std::format("errno {}", code)) || !seen.insert(text.substr(0, text.find(" (errno"))).second) {
				return false;
			}
		}
	}
	return seen.size() == stages.size() * causes.size();
}

/**
 * The engine's failure reads as its cause does: the cards' own description of theirs, found beside the cards' error,
 * or the reactor's of the wait that failed, each said to be which.
 */
[[nodiscard]] auto an_engine_failure_reads_as_its_cause() -> bool {
	auto const cards = doubles::CardsFailure{.device = doubles::breakdown_error()};
	auto const of_cards = pipeline::describe(pipeline::EngineError<doubles::CardsFailure>{.cause = cards});
	auto const wait = net::NetError{.stage = net::Stage::Wait, .cause = net::Cause::Invalid, .code = 9};
	auto const of_wait = pipeline::describe(pipeline::EngineError<doubles::CardsFailure>{.cause = wait});
	return of_cards.contains(doubles::describe(cards)) && of_cards.contains("cuStreamQuery") && of_cards.starts_with("the cards failed") &&
	       of_wait.contains(net::describe(wait)) && of_wait.contains("wait for work") && of_cards != of_wait;
}

}

auto main() -> int {
	auto const cases = std::to_array<engine::testing::Case>({
	    {.name = "every reactor failure reads its own", .holds = every_reactor_failure_reads_its_own()},
	    {.name = "an engine failure reads as its cause", .holds = an_engine_failure_reads_as_its_cause()},
	});
	return engine::testing::report(cases);
}
