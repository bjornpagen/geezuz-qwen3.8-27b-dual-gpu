// SPDX-License-Identifier: AGPL-3.0-only
//
// Weight products: the schedule's split covers every output block's k exactly once in SM order on both cards; a
// block's sums are float64's dot products within fp32 accumulation error; and a token's sums do not depend on the
// other tokens computed with it.
import std;
import engine.model;
import engine.ref;
import engine.schedule;
import engine.testing;

namespace {

namespace ref = engine::ref;
namespace model = engine::model;
namespace schedule = engine::schedule;

using model::Count;
using schedule::Op;
using schedule::Select;

/** Every block's segments tile its k units in order, and every row of the tensor's selection is one card's. */
[[nodiscard]] auto plan_covers(ref::Product const& product, Count rows_expected) -> bool {
	auto rows = std::set<std::pair<int, Count>>{};
	for (auto const& work : product.tiles) {
		auto next = Count{0};
		for (auto const& segment : work.segments) {
			if (segment.first != next || segment.count == 0) {
				return false;
			}
			next += segment.count;
		}
		if (next != product.units_per_output) {
			return false;
		}
		for (auto row = Count{0}; row < ref::tile_rows; ++row) {
			auto const source = ref::source_row(product.tensor, work.card, (work.block * ref::tile_rows) + row);
			if (!rows.emplace(static_cast<int>(std::to_underlying(work.card)), source).second) {
				return false;
			}
		}
	}
	return rows.size() == rows_expected;
}

[[nodiscard]] auto plans_cover() -> bool {
	auto const& t = schedule::tensors;
	auto const all = std::array{Select::All};
	auto const both_vocabularies = std::array{Select::DraftVocabulary, Select::OtherVocabulary};
	// Axis-1 splits put every row on both cards; axis-0 splits put each row on one.
	return plan_covers(ref::plan(Op::MlpIn, model::Stream::Body, 0, t.gate, all), model::ffn) &&
	       plan_covers(ref::plan(Op::MlpOut, model::Stream::Body, 5, t.down, all), 2 * model::hidden) &&
	       plan_covers(ref::plan(Op::DeltanetIn, model::Stream::Body, 0, t.qkv, all), model::conv_channels) &&
	       plan_covers(ref::plan(Op::AttentionIn, model::Stream::Body, 3, t.q, std::array{Select::Query}), model::query_features) &&
	       plan_covers(ref::plan(Op::AttentionGate, model::Stream::Body, 3, t.q, std::array{Select::Gate}), model::query_features) &&
	       plan_covers(ref::plan(Op::AttentionIn, model::Stream::Body, 3, t.v, all), model::kv_heads * model::head_dim) &&
	       plan_covers(ref::plan(Op::MtpIn, model::Stream::Mtp, 0, t.fc, all), 2 * model::hidden) &&
	       plan_covers(ref::plan(Op::Head, model::Stream::Head, 0, t.lm_head, both_vocabularies), model::vocab) &&
	       plan_covers(ref::plan(Op::DraftHead, model::Stream::Mtp, 0, t.lm_head, std::array{Select::DraftVocabulary}), model::draft_vocab);
}

/** Random tiles: e4m3-valued weights with a scale per row, or bf16 weights. */
struct SyntheticTiles {
	model::Format format;
	std::uint64_t seed;

	[[nodiscard]] auto read(ref::TileWork const& work, Count unit) const -> ref::TileData {
		auto generator = engine::testing::generator(seed + (work.block * 1000) + unit);
		auto const columns = schedule::tile_columns(format);
		auto value = std::uniform_real_distribution<float>{-1.0F, 1.0F};
		auto tile = ref::TileData{.weights = {}, .scales = {}};
		for (auto index = Count{0}; index < ref::tile_rows * columns; ++index) {
			tile.weights.push_back(ref::round(value(generator)));
		}
		if (format == model::Format::E4m3) {
			for (auto row = Count{0}; row < ref::tile_rows; ++row) {
				tile.scales.push_back(std::ldexp(1.0F + value(generator) * 0.5F, -6));
			}
		}
		return tile;
	}
};

[[nodiscard]] auto activations(std::uint64_t seed, Count tokens, Count width) -> ref::Rows<ref::Bf16> {
	auto generator = engine::testing::generator(seed);
	auto value = std::uniform_real_distribution<float>{-2.0F, 2.0F};
	auto x = ref::Rows<ref::Bf16>::zeros(tokens, width);
	for (auto& each : x.values) {
		each = ref::round(value(generator));
	}
	return x;
}

/** A block of `units` k units split as `cuts`, against float64, and token by token against all tokens at once. */
template<ref::Accumulation A>
[[nodiscard]] auto block_sums_hold(model::Format format, std::vector<ref::Segment> segments, Count units) -> bool {
	auto const columns = schedule::tile_columns(format);
	auto const reader = SyntheticTiles{.format = format, .seed = 77};
	auto const work = ref::TileWork{.card = model::Card::One, .block = 3, .segments = std::move(segments)};
	auto const x = activations(78, 5, units * columns);
	auto const got = ref::tile_sums<A>(work, format, reader, x);
	for (auto row = Count{0}; row < ref::tile_rows; ++row) {
		for (auto token = Count{0}; token < x.count(); ++token) {
			auto want = 0.0;
			auto magnitude = 0.0;
			for (auto unit = Count{0}; unit < units; ++unit) {
				auto const tile = reader.read(work, unit);
				auto partial = 0.0;
				for (auto k = Count{0}; k < columns; ++k) {
					auto const term = static_cast<double>(ref::widen(tile.weights[(row * columns) + k])) *
					                  static_cast<double>(ref::widen(x.row(token)[(unit * columns) + k]));
					partial += term;
					magnitude += std::abs(term);
				}
				want += format == model::Format::E4m3 ? partial * static_cast<double>(tile.scales[row]) : partial;
			}
			auto const bound = 1e-5 * magnitude * (format == model::Format::E4m3 ? 1.0 / 32.0 : 1.0);
			if (std::abs(static_cast<double>(got.row(token)[row]) - want) > bound) {
				return false;
			}
		}
	}
	for (auto token = Count{0}; token < x.count(); ++token) {
		auto const alone = ref::Rows<ref::Bf16>{.width = x.width, .values = {x.row(token).begin(), x.row(token).end()}};
		auto const single = ref::tile_sums<A>(work, format, reader, alone);
		if (!std::ranges::equal(single.row(0), got.row(token), [](float a, float b) {
			    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
		    })) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "every product's blocks have k segments tiling their units in SM order, each selected row on its cards",
	        .holds = plans_cover(),
	    },
	    engine::testing::Case{
	        .name = "e4m3 block sums over three SM segments, ieee and sm_86, are float64's, and per-token sums ignore the other tokens (seed 77)",
	        .holds = block_sums_hold<ref::Ieee>(model::Format::E4m3, {{.first = 0, .count = 2}, {.first = 2, .count = 3}, {.first = 5, .count = 1}}, 6) && block_sums_hold<ref::Tensor>(model::Format::E4m3, {{.first = 0, .count = 2}, {.first = 2, .count = 3}, {.first = 5, .count = 1}}, 6),
	    },
	    engine::testing::Case{
	        .name = "bf16 block sums over two SM segments, ieee and sm_86, are float64's, and per-token sums ignore the other tokens (seed 77)",
	        .holds = block_sums_hold<ref::Ieee>(model::Format::Bf16, {{.first = 0, .count = 4}, {.first = 4, .count = 3}}, 7) && block_sums_hold<ref::Tensor>(model::Format::Bf16, {{.first = 0, .count = 4}, {.first = 4, .count = 3}}, 7),
	    },
	};
	return engine::testing::report(cases);
}
