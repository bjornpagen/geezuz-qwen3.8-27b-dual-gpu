// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.exec;
import engine.testing;

namespace ex = engine::exec;

namespace {

struct ProbeError {
	std::int32_t code;

	[[nodiscard]] constexpr auto retry() const -> engine::core::Retry {
		return engine::core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(ProbeError const&) const -> bool = default;
};

enum class Channel : std::uint8_t {
	Value,
	Error,
	Stopped,
};

/** A sender declaring all three completions and taking the one chosen at run time. */
struct ProbeSender {
	using sender_concept = ex::sender_t;
	using completion_signatures =
	    ex::completion_signatures<ex::set_value_t(std::int32_t), ex::set_error_t(ProbeError), ex::set_stopped_t()>;

	Channel channel;
	std::int32_t payload;

	template<class Receiver>
	struct Operation {
		Receiver receiver;
		Channel channel;
		std::int32_t payload;

		auto start() & noexcept -> void {
			switch (channel) {
			case Channel::Value:
				ex::set_value(std::move(receiver), payload);
				return;
			case Channel::Error:
				ex::set_error(std::move(receiver), ProbeError{payload});
				return;
			case Channel::Stopped:
				ex::set_stopped(std::move(receiver));
				return;
			}
		}
	};

	template<class Receiver>
	[[nodiscard]] auto connect(Receiver receiver) const noexcept -> Operation<Receiver> {
		return Operation<Receiver>{std::move(receiver), channel, payload};
	}
};

using Waited = std::optional<std::expected<std::tuple<std::int32_t>, ProbeError>>;

[[nodiscard]] auto value(std::int32_t v) -> Waited {
	return std::expected<std::tuple<std::int32_t>, ProbeError>{std::tuple{v}};
}

[[nodiscard]] auto failure(std::int32_t code) -> Waited {
	return std::expected<std::tuple<std::int32_t>, ProbeError>{std::unexpect, ProbeError{code}};
}

[[nodiscard]] auto value_chain_doubles_and_increments() -> bool {
	return std::ranges::all_of(std::views::iota(-50, 50), [](std::int32_t seed) {
		auto chain = ex::just(seed) | ex::let_value([](std::int32_t v) {
			             return ex::just(v + v);
		             }) |
		             ex::then([](std::int32_t v) {
			             return v + 1;
		             });
		return ex::wait<ProbeError>(std::move(chain)) == value(2 * seed + 1);
	});
}

[[nodiscard]] auto upon_error_recovers_the_payload() -> bool {
	auto chain = ProbeSender{.channel = Channel::Error, .payload = 7} | ex::upon_error([](ProbeError error) {
		             return error.code + 1;
	             });
	return ex::wait<ProbeError>(std::move(chain)) == value(8);
}

[[nodiscard]] auto let_error_reroutes_to_a_new_sender() -> bool {
	auto chain = ProbeSender{.channel = Channel::Error, .payload = 21} | ex::let_error([](ProbeError error) {
		             return ex::just(error.code + error.code);
	             });
	return ex::wait<ProbeError>(std::move(chain)) == value(42);
}

[[nodiscard]] auto wait_delivers_the_typed_error() -> bool {
	return ex::wait<ProbeError>(ProbeSender{.channel = Channel::Error, .payload = 123}) == failure(123);
}

[[nodiscard]] auto wait_reports_stopped_as_absent() -> bool {
	return ex::wait<ProbeError>(ProbeSender{.channel = Channel::Stopped, .payload = 0}) == std::nullopt;
}

[[nodiscard]] auto when_all_joins_every_value() -> bool {
	auto joined =
	    ex::when_all(ex::just(std::int32_t{1}) | ex::then([](std::int32_t v) {
		                 return v + v;
	                 }),
		             ProbeSender{.channel = Channel::Value, .payload = 4}, ex::just(std::int32_t{3}) | ex::let_value([](std::int32_t v) {
		                                                                       return ex::just(v + v);
	                                                                       })) |
	    ex::then([](std::int32_t a, std::int32_t b, std::int32_t c) {
		    return a + b + c;
	    });
	return ex::wait<ProbeError>(std::move(joined)) == value(12);
}

[[nodiscard]] auto when_all_fails_with_the_child_error() -> bool {
	auto joined = ex::when_all(ex::just(std::int32_t{1}), ProbeSender{.channel = Channel::Error, .payload = 5}) |
	              ex::then([](std::int32_t a, std::int32_t b) {
		              return a + b;
	              });
	return ex::wait<ProbeError>(std::move(joined)) == failure(5);
}

[[nodiscard]] auto into_variant_round_trips() -> bool {
	auto chain = ex::just(std::int32_t{9}) | ex::into_variant() | ex::then([](std::variant<std::tuple<std::int32_t>> v) {
		             return std::get<0>(std::get<0>(v));
	             });
	return ex::wait<ProbeError>(std::move(chain)) == value(9);
}

[[nodiscard]] auto stopped_as_optional_materializes_both_channels() -> bool {
	auto const unwrap = [](std::optional<std::int32_t> v) {
		return v ? *v : std::int32_t{-1};
	};
	auto engaged = ProbeSender{.channel = Channel::Value, .payload = 41} | ex::stopped_as_optional() | ex::then(unwrap);
	auto stopped = ProbeSender{.channel = Channel::Stopped, .payload = 41} | ex::stopped_as_optional() | ex::then(unwrap);
	return ex::wait<ProbeError>(std::move(engaged)) == value(41) && ex::wait<ProbeError>(std::move(stopped)) == value(-1);
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "just | let_value | then computes 2 * seed + 1 for seeds in [-50, 50)",
	        .holds = value_chain_doubles_and_increments(),
	    },
	    engine::testing::Case{.name = "upon_error recovers a typed error to a value", .holds = upon_error_recovers_the_payload()},
	    engine::testing::Case{.name = "let_error reroutes a typed error to a new sender", .holds = let_error_reroutes_to_a_new_sender()},
	    engine::testing::Case{.name = "wait delivers the typed error with its payload", .holds = wait_delivers_the_typed_error()},
	    engine::testing::Case{.name = "wait reports the stopped channel as absent", .holds = wait_reports_stopped_as_absent()},
	    engine::testing::Case{.name = "when_all joins the values of three senders in order", .holds = when_all_joins_every_value()},
	    engine::testing::Case{.name = "when_all fails with a failing child's typed error", .holds = when_all_fails_with_the_child_error()},
	    engine::testing::Case{.name = "into_variant round-trips a value", .holds = into_variant_round_trips()},
	    engine::testing::Case{
	        .name = "stopped_as_optional passes a value through and turns stopped into absence",
	        .holds = stopped_as_optional_materializes_both_channels(),
	    },
	};
	return engine::testing::report(cases);
}
