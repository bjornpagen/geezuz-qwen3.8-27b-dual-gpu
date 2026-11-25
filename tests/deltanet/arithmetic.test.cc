// SPDX-License-Identifier: AGPL-3.0-only
//
// The deltanet's fp32 arithmetic against the model's in float64. Each check's tolerance is derived from the fixed fp32
// operation order, never fitted: unit roundoff u = 2^-24, gamma_n = n u / (1 - n u) for a sum whose terms pass n
// roundings, the shared transcendentals' proven ulp bounds, and bf16's unit roundoff 2^-8 where the model rounds.
//
// The recurrence's bound is carried token by token. On a state column s the exact update is s -> d (I - beta k k^T) s +
// beta v k, which never grows a difference when 0 <= d <= 1 and 0 <= beta |k|^2 <= 2; so the gap between the fp32 and
// the float64 chains after t tokens is at most the sum of the rounding each fp32 step adds, which the step's own
// magnitudes bound.
import std;
import engine.deltanet;
import engine.deltanet_fixtures;
import engine.testing;

namespace {

namespace dn = engine::deltanet;
namespace fx = engine::deltanet_fixtures;

constexpr double unit = 0x1p-24;
constexpr double bf16_unit = 0x1p-8;
constexpr double ulp = 0x1p-23;
/** The float64 chains' own rounding, 2^-29 of the fp32 bound, and the subnormal floor, folded into the comparison. */
constexpr double slack = 1.0 + 0x1p-20;
constexpr double floor = 0x1p-120;

[[nodiscard]] constexpr auto wide(float value) -> double {
	return static_cast<double>(value);
}

[[nodiscard]] constexpr auto gamma(int roundings) -> double {
	return roundings * unit / (1.0 - roundings * unit);
}

[[nodiscard]] auto within(double got, double expected, double bound) -> bool {
	return std::abs(got - expected) <= bound * slack + floor;
}

[[nodiscard]] auto norm(std::span<double const> values) -> double {
	return std::sqrt(std::ranges::fold_left(values, 0.0, [](double total, double value) {
		return total + value * value;
	}));
}

/** A normalized query or key, as the device makes it from a convolved bf16 vector. */
[[nodiscard]] auto unit_vector(fx::Generator& generator, dn::Role role) -> dn::Vector {
	auto convolved = std::array<dn::Bf16, dn::dim>{};
	std::ranges::generate(convolved, [&] {
		return fx::bf16(fx::normal(generator, 1.0F));
	});
	return dn::normalized_vector(convolved, role);
}

/**
 * The chain under tree `T` (the probe's emulation, the device's order) against the float64 recurrence on the same
 * fp32 inputs, over `tokens` tokens on every column of every probe head: each output within its carried bound, and the
 * final state of each column within the carried state bound in the 2-norm.
 */
template<class T>
[[nodiscard]] auto chain_within_bound(std::uint64_t seed, std::uint32_t tokens) -> bool {
	auto generator = engine::testing::generator(seed);
	auto const rows = static_cast<std::size_t>(dn::probe_heads) * tokens;
	auto keys = std::vector<float>(rows * dn::dim);
	auto queries = std::vector<float>(rows * dn::dim);
	auto values = std::vector<float>(rows * dn::dim);
	auto gates = std::vector<dn::Gates>(rows);
	auto natural_keys = std::vector<std::array<float, dn::dim>>{};
	auto natural_queries = std::vector<std::array<float, dn::dim>>{};
	for (auto row = std::size_t{0}; row < rows; ++row) {
		natural_keys.push_back(unit_vector(generator, dn::Role::Key));
		natural_queries.push_back(unit_vector(generator, dn::Role::Query));
		for (auto element = 0; element < dn::dim; ++element) {
			auto const at = row * dn::dim + static_cast<std::size_t>(dn::vector_element<T>(element));
			keys.at(at) = natural_keys.back().at(static_cast<std::size_t>(element));
			queries.at(at) = natural_queries.back().at(static_cast<std::size_t>(element));
			values.at(row * dn::dim + static_cast<std::size_t>(element)) = fx::normal(generator, 1.0F);
		}
		// Long memories as well as short: the decay reaches 0.999.
		gates.at(row) = {.decay = fx::uniform(generator, 0.3F, 0.999F), .beta = fx::uniform(generator, 0.0F, 1.0F)};
	}
	auto state = std::vector<float>(static_cast<std::size_t>(dn::probe_heads * dn::head_floats));
	std::ranges::generate(state, [&] {
		return fx::normal(generator, 0.05F);
	});
	auto const initial = state;
	auto outputs = std::vector<float>(rows * dn::dim);
	dn::probe<T>({.keys = keys, .queries = queries, .values = values, .gates = gates, .tokens = tokens}, state, outputs);

	constexpr int depth = T::per_lane - 1 + std::countr_zero(static_cast<unsigned>(T::lanes));
	auto holds = true;
	for (auto head = 0; head < dn::probe_heads; ++head) {
		auto const base = static_cast<std::size_t>(head) * std::size_t{dn::head_floats};
		for (auto column = 0; column < dn::dim; ++column) {
			auto s = std::array<double, dn::dim>{};
			for (auto row = 0; row < dn::dim; ++row) {
				s.at(static_cast<std::size_t>(row)) =
				    wide(initial.at(base + static_cast<std::size_t>(dn::StateLayout<T>::element(row, column))));
			}
			auto carried = 0.0;
			for (auto token = std::size_t{0}; token < tokens; ++token) {
				auto const row_index = static_cast<std::size_t>(head) * tokens + token;
				auto const& k = natural_keys.at(row_index);
				auto const& q = natural_queries.at(row_index);
				auto const d = wide(gates.at(row_index).decay);
				auto const beta = wide(gates.at(row_index).beta);
				auto const v = wide(values.at(row_index * dn::dim + static_cast<std::size_t>(column)));
				auto key_norm = 0.0;
				auto query_norm = 0.0;
				auto magnitude = 0.0;
				auto projected = 0.0;
				for (auto i = std::size_t{0}; i < dn::dim; ++i) {
					key_norm += wide(k.at(i)) * wide(k.at(i));
					query_norm += wide(q.at(i)) * wide(q.at(i));
					magnitude += std::abs(s.at(i) * wide(k.at(i)));
					projected += d * s.at(i) * wide(k.at(i));
				}
				key_norm = std::sqrt(key_norm);
				query_norm = std::sqrt(query_norm);
				holds = holds && beta * key_norm * key_norm <= 2.0;
				// The fp32 state's magnitudes, bounded through the float64 state and the gap carried so far.
				auto const state_norm = norm(s) + carried;
				auto const a = d * (magnitude + carried * key_norm);
				auto const residual = std::abs(v) + a * (1.0 + gamma(depth + 2));
				auto const delta_error = beta * (gamma(depth + 2) * a + gamma(2) * residual);
				auto const delta_magnitude = beta * residual * (1.0 + gamma(2));
				carried += (2.0 * unit + unit * unit) * (d * state_norm + key_norm * delta_magnitude) + key_norm * delta_error;
				auto const delta = (v - projected) * beta;
				auto output = 0.0;
				for (auto i = std::size_t{0}; i < dn::dim; ++i) {
					s.at(i) = d * s.at(i) + wide(k.at(i)) * delta;
					output += s.at(i) * wide(q.at(i));
				}
				auto const output_bound = gamma(depth + 1) * (norm(s) + carried) * query_norm + carried * query_norm;
				holds = holds && within(wide(outputs.at(row_index * dn::dim + static_cast<std::size_t>(column))), output, output_bound);
			}
			auto gap = std::array<double, dn::dim>{};
			for (auto row = 0; row < dn::dim; ++row) {
				auto const got = state.at(base + static_cast<std::size_t>(dn::StateLayout<T>::element(row, column)));
				gap.at(static_cast<std::size_t>(row)) = wide(got) - s.at(static_cast<std::size_t>(row));
			}
			holds = holds && norm(gap) <= carried * slack + floor;
		}
	}
	return holds;
}

[[nodiscard]] auto silu(double x) -> double {
	return x / (1.0 + std::exp(-x));
}

[[nodiscard]] auto random_bf16(fx::Generator& generator, float deviation) -> dn::Bf16 {
	return fx::bf16(fx::normal(generator, deviation));
}

/**
 * The convolution and its silu against float64: the sum of four exact products rounds at most three times, then the
 * model rounds to bf16, takes silu (2.5 ulp, slope at most 1.1), and rounds to bf16 again.
 */
[[nodiscard]] auto convolution_within_bound(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto holds = true;
	for (auto sample = 0; sample < 200'000; ++sample) {
		auto taps = dn::Four{};
		auto inputs = dn::Four{};
		auto exact = 0.0;
		auto magnitude = 0.0;
		for (auto tap = 0; tap < 4; ++tap) {
			taps.element[tap] = random_bf16(generator, 0.5F);
			inputs.element[tap] = random_bf16(generator, 2.0F);
			auto const product = wide(dn::widen(taps.element[tap])) * wide(dn::widen(inputs.element[tap]));
			exact += product;
			magnitude += std::abs(product);
		}
		auto const got = wide(dn::widen(dn::convolve(taps, inputs)));
		auto const summed = gamma(3) * magnitude;
		auto const convolved = (std::abs(exact) + summed) * (1.0 + bf16_unit);
		auto const argument = summed + bf16_unit * convolved;
		auto const activated = std::abs(silu(exact)) + 1.1 * argument;
		auto const bound = 1.1 * argument + 2.5 * ulp * activated + bf16_unit * (activated + 2.5 * ulp * activated);
		holds = holds && within(got, silu(exact), bound);
	}
	return holds;
}

/**
 * The gates against float64. decay = e^g: the shift rounds once, e^A_log is within an ulp, softplus within two (slope
 * at most 1), the product rounds once, and e^g adds an ulp and multiplies g's error by e^g. beta: sigmoid within 2.5 ulp,
 * then the model's bf16 rounding.
 */
[[nodiscard]] auto gates_within_bound(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto holds = true;
	for (auto sample = 0; sample < 200'000; ++sample) {
		auto const a = random_bf16(generator, 2.0F);
		auto const b = random_bf16(generator, 3.0F);
		auto const a_log = fx::bf16(std::log(fx::uniform(generator, 0.01F, 16.0F)));
		auto const dt_bias = fx::bf16(fx::uniform(generator, -4.0F, 4.0F));
		auto const got = dn::gates(a, b, a_log, dt_bias);
		auto const shifted = wide(dn::widen(a)) + wide(dn::widen(dt_bias));
		auto const rate = std::exp(wide(dn::widen(a_log)));
		auto const softplus = std::log1p(std::exp(shifted));
		auto const g = -rate * softplus;
		auto const softplus_error = unit * std::abs(shifted) + 2.0 * ulp * softplus;
		auto const g_error = rate * ulp * softplus + rate * (1.0 + ulp) * softplus_error + unit * std::abs(g) * (1.0 + 4.0 * ulp);
		auto const decay = std::exp(g);
		auto const decay_bound = decay * (std::expm1(g_error) + ulp * std::exp(g_error));
		auto const sigmoid = 1.0 / (1.0 + std::exp(-wide(dn::widen(b))));
		auto const beta_bound = 2.5 * ulp * sigmoid + bf16_unit * sigmoid * (1.0 + 2.5 * ulp);
		holds = holds && within(wide(got.decay), decay, decay_bound) && within(wide(got.beta), sigmoid, beta_bound);
	}
	return holds;
}

/**
 * The l2 normalization and the gated norm, as the device's eight-lane groups compute them, against float64. The sum of
 * squares passes the tree's 19 roundings, then eps's addition; rsqrt is within 0.51 ulp and halves the sum's relative
 * error; each product or quotient rounds once, and a query's divisor is sqrt(128) rounded to fp32. The gated norm also
 * divides by 128 exactly, and the model rounds twice to bf16.
 */
[[nodiscard]] auto norms_within_bound(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	constexpr int depth = dn::Column::per_lane - 1 + 3;
	auto holds = true;
	for (auto sample = 0; sample < 2'000; ++sample) {
		auto const deviation = std::exp2(fx::uniform(generator, -12.0F, 6.0F));
		auto inputs = std::array<dn::Bf16, dn::dim>{};
		auto squares = 0.0;
		for (auto& input : inputs) {
			input = random_bf16(generator, deviation);
			squares += wide(dn::widen(input)) * wide(dn::widen(input));
		}
		auto const relative = gamma(depth + 1) * squares / (squares + wide(dn::l2_norm_eps)) + unit;
		auto const scale_error = 0.5 * relative + 0.51 * ulp;
		for (auto const role : {dn::Role::Query, dn::Role::Key}) {
			auto const normalized = dn::normalized_vector(inputs, role);
			auto const divisor = role == dn::Role::Query ? std::sqrt(128.0) : 1.0;
			auto const roundings = role == dn::Role::Query ? 3.0 : 1.0;
			for (auto element = std::size_t{0}; element < dn::dim; ++element) {
				auto const expected = wide(dn::widen(inputs.at(element))) / std::sqrt(squares + wide(dn::l2_norm_eps)) / divisor;
				auto const bound = std::abs(expected) * (scale_error * (1.0 + scale_error) + roundings * unit) * 1.01;
				holds = holds && within(wide(normalized.at(element)), expected, bound);
			}
		}
		auto const mean = squares / dn::dim;
		auto const mean_relative = gamma(depth + 1) * mean / (mean + wide(1e-6F)) + unit;
		auto const norm_error = 0.5 * mean_relative + 0.51 * ulp;
		auto const expected_scale = 1.0 / std::sqrt(mean + wide(1e-6F));
		auto weights = std::array<dn::Bf16, dn::dim>{};
		std::ranges::generate(weights, [&] {
			return fx::bf16(fx::uniform(generator, 0.5F, 1.5F));
		});
		auto outputs = std::array<dn::Bf16, dn::dim>{};
		dn::normalize_head(inputs, weights, outputs);
		for (auto element = std::size_t{0}; element < dn::dim; ++element) {
			auto const expected = wide(dn::widen(inputs.at(element))) * expected_scale * wide(dn::widen(weights.at(element)));
			auto const bound = std::abs(expected) * ((norm_error + unit) * (1.0 + bf16_unit) + bf16_unit) * (1.0 + bf16_unit) +
			                   bf16_unit * std::abs(expected);
			holds = holds && within(wide(dn::widen(outputs.at(element))), expected, bound * 1.01);
		}
	}
	return holds;
}

/** The tree is the butterfly at distances 4, 2, 1: ((x0 + x4) + (x2 + x6)) + ((x1 + x5) + (x3 + x7)). */
[[nodiscard]] auto tree_pairs_as_the_butterfly(std::uint64_t seed) -> bool {
	auto generator = engine::testing::generator(seed);
	auto holds = true;
	for (auto sample = 0; sample < 100'000; ++sample) {
		auto x = std::array<float, 8>{};
		for (auto& value : x) {
			value = fx::normal(generator, 1.0F) * std::exp2(fx::uniform(generator, -30.0F, 30.0F));
		}
		auto const left = (x[0] + x[4]) + (x[2] + x[6]);
		auto const right = (x[1] + x[5]) + (x[3] + x[7]);
		auto const expected = left + right;
		holds = holds && std::bit_cast<std::uint32_t>(dn::tree_total(x)) == std::bit_cast<std::uint32_t>(expected);
	}
	return holds;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the tree pairs lanes as the butterfly does", .holds = tree_pairs_as_the_butterfly(1)},
	    engine::testing::Case{.name = "the eight-lane chain stays within its carried bound of float64",
		                      .holds = chain_within_bound<dn::Column>(2, 96)},
	    engine::testing::Case{.name = "the whole-warp chain stays within its carried bound of float64",
		                      .holds = chain_within_bound<dn::Tree<32>>(3, 96)},
	    engine::testing::Case{.name = "the convolution and its silu stay within bound of float64", .holds = convolution_within_bound(4)},
	    engine::testing::Case{.name = "decay and beta stay within bound of float64", .holds = gates_within_bound(5)},
	    engine::testing::Case{.name = "the l2 and gated norms stay within bound of float64", .holds = norms_within_bound(6)},
	};
	return engine::testing::report(cases);
}
