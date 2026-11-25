// SPDX-License-Identifier: AGPL-3.0-only
//
// The deltanet step on a small shape against a float64 rendering of the model's recurrence (the checkpoint's serving
// code, step by step): over many tokens the fp32 state stays close to float64's, and each output is float64's within
// its bf16 rounding.
import std;
import engine.model;
import engine.ref;
import engine.testing;

namespace {

namespace ref = engine::ref;
namespace model = engine::model;

using model::Count;

struct Small {
	static constexpr Count key_heads = 2;
	static constexpr Count value_heads = 4;
	static constexpr Count key_dim = 32;
	static constexpr Count value_dim = 64;
	static constexpr Count conv_kernel = 4;
};

constexpr auto channels = ref::conv_channels<Small>;
constexpr auto key_features = Small::key_heads * Small::key_dim;
constexpr auto value_features = Small::value_heads * Small::value_dim;

[[nodiscard]] auto silu(double x) -> double {
	return x / (1.0 + std::exp(-x));
}

/** The recurrence in float64, from the same bf16 inputs and weights. */
struct Float64 {
	std::vector<double> state = std::vector<double>(Small::value_heads * Small::key_dim * Small::value_dim);
	std::vector<double> window = std::vector<double>((Small::conv_kernel - 1) * channels);

	auto step(ref::DeltanetWeights const& w, ref::DeltanetInputs const& in) -> std::vector<double> {
		auto convolved = std::vector<double>(channels);
		for (auto c = Count{0}; c < channels; ++c) {
			auto sum = 0.0;
			for (auto tap = Count{0}; tap < Small::conv_kernel; ++tap) {
				auto const x = tap + 1 < Small::conv_kernel ? window[(tap * channels) + c] : static_cast<double>(ref::widen(in.projection[c]));
				sum += static_cast<double>(ref::widen(w.conv[(c * Small::conv_kernel) + tap])) * x;
			}
			convolved[c] = silu(sum);
		}
		for (auto c = Count{0}; c < channels; ++c) {
			window[c] = window[channels + c];
			window[channels + c] = window[(2 * channels) + c];
			window[(2 * channels) + c] = static_cast<double>(ref::widen(in.projection[c]));
		}
		auto const normalize = [](std::span<double const> x, double scale) {
			auto squares = 0.0;
			for (auto const each : x) {
				squares += each * each;
			}
			auto const factor = scale / std::sqrt(squares + 1e-6);
			auto out = std::vector<double>{};
			for (auto const each : x) {
				out.push_back(each * factor);
			}
			return out;
		};
		auto out = std::vector<double>{};
		auto const group = Small::value_heads / Small::key_heads;
		for (auto head = Count{0}; head < Small::value_heads; ++head) {
			auto const kh = head / group;
			auto const q = normalize(std::span{convolved}.subspan(kh * Small::key_dim, Small::key_dim), 1.0 / static_cast<double>(model::deltanet::query_divisor));
			auto const k = normalize(std::span{convolved}.subspan(key_features + (kh * Small::key_dim), Small::key_dim), 1.0);
			auto const v = std::span{convolved}.subspan((2 * key_features) + (head * Small::value_dim), Small::value_dim);
			auto const beta = 1.0 / (1.0 + std::exp(-static_cast<double>(in.b[head])));
			auto const softplus = std::log1p(std::exp(static_cast<double>(in.a[head]) + static_cast<double>(ref::widen(w.dt_bias[head]))));
			auto const decay = std::exp(-std::exp(static_cast<double>(ref::widen(w.a_log[head]))) * softplus);
			auto core = std::vector<double>(Small::value_dim);
			for (auto j = Count{0}; j < Small::value_dim; ++j) {
				auto const s = std::span{state}.subspan(((head * Small::value_dim) + j) * Small::key_dim, Small::key_dim);
				auto memory = 0.0;
				for (auto i = Count{0}; i < Small::key_dim; ++i) {
					s[i] *= decay;
					memory += s[i] * k[i];
				}
				auto const delta = (v[j] - memory) * beta;
				auto read = 0.0;
				for (auto i = Count{0}; i < Small::key_dim; ++i) {
					s[i] += k[i] * delta;
					read += s[i] * q[i];
				}
				core[j] = read;
			}
			auto squares = 0.0;
			for (auto const each : core) {
				squares += each * each;
			}
			auto const factor = 1.0 / std::sqrt((squares / double{Small::value_dim}) + static_cast<double>(model::rms_norm_eps));
			for (auto j = Count{0}; j < Small::value_dim; ++j) {
				auto const z = static_cast<double>(in.gate[(head * Small::value_dim) + j]);
				out.push_back(core[j] * factor * static_cast<double>(ref::widen(w.norm[j])) * silu(z));
			}
		}
		return out;
	}
};

[[nodiscard]] auto bf16_draws(std::mt19937_64& generator, Count count, float low, float high) -> std::vector<ref::Bf16> {
	auto value = std::uniform_real_distribution<float>{low, high};
	auto out = std::vector<ref::Bf16>(count);
	for (auto& each : out) {
		each = ref::round(value(generator));
	}
	return out;
}

[[nodiscard]] auto float_draws(std::mt19937_64& generator, Count count, float low, float high) -> std::vector<float> {
	auto value = std::uniform_real_distribution<float>{low, high};
	auto out = std::vector<float>(count);
	for (auto& each : out) {
		each = value(generator);
	}
	return out;
}

struct Outcome {
	double worst_output;
	double worst_state;
};

/** Runs `tokens` steps of both and returns the worst relative output error and the worst state error. */
[[nodiscard]] auto run(std::uint64_t seed, Count tokens) -> Outcome {
	auto generator = engine::testing::generator(seed);
	auto const conv = bf16_draws(generator, channels * Small::conv_kernel, -0.6F, 0.6F);
	auto const a_log = bf16_draws(generator, Small::value_heads, -1.0F, 1.5F);
	auto const dt_bias = bf16_draws(generator, Small::value_heads, -1.0F, 1.0F);
	auto const norm = bf16_draws(generator, Small::value_dim, 0.5F, 1.5F);
	auto const weights = ref::DeltanetWeights{.conv = conv, .a_log = a_log, .dt_bias = dt_bias, .norm = norm};
	auto state = ref::DeltanetState<Small>{};
	auto exact = Float64{};
	auto outcome = Outcome{.worst_output = 0.0, .worst_state = 0.0};
	for (auto token = Count{0}; token < tokens; ++token) {
		auto const projection = bf16_draws(generator, channels, -2.0F, 2.0F);
		auto const gate = float_draws(generator, value_features, -3.0F, 3.0F);
		auto const a = float_draws(generator, Small::value_heads, -3.0F, 3.0F);
		auto const b = float_draws(generator, Small::value_heads, -3.0F, 3.0F);
		auto const in = ref::DeltanetInputs{.projection = projection, .gate = gate, .a = a, .b = b};
		auto const got = ref::deltanet_step<Small>(state, weights, in);
		auto const want = exact.step(weights, in);
		for (auto const [g, w] : std::views::zip(got, want)) {
			outcome.worst_output = std::max(outcome.worst_output, std::abs(static_cast<double>(ref::widen(g)) - w) / std::max(std::abs(w), 1e-3));
		}
		for (auto const [g, w] : std::views::zip(state.state, exact.state)) {
			outcome.worst_state = std::max(outcome.worst_state, std::abs(static_cast<double>(g) - w));
		}
	}
	return outcome;
}

}

auto main() -> int {
	auto const first = run(11, 200);
	auto const second = run(12, 200);
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "200 deltanet steps: every output within its bf16 rounding of float64's (seeds 11, 12)",
	        .holds = first.worst_output < 0x1p-7 && second.worst_output < 0x1p-7,
	    },
	    engine::testing::Case{
	        .name = "200 deltanet steps: the fp32 state within 1e-4 of float64's (seeds 11, 12)",
	        .holds = first.worst_state < 1e-4 && second.worst_state < 1e-4,
	    },
	};
	return engine::testing::report(cases);
}
