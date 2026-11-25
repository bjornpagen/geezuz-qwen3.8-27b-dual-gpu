// SPDX-License-Identifier: AGPL-3.0-only
//
// Where the engine computes each output element of a weight product, read from the schedule the kernels compile: which
// card, which 128-row block of the card's part, and which SMs hold which of its k units, in SM order. The order of the
// sum follows from that alone (`model::Sum::Weights`), so the reference takes it from here and nowhere else.
export module engine.ref:plan;

import std;
import engine.core;
import engine.model;
import engine.schedule;
import engine.vocabulary;

namespace engine::ref {

namespace model = engine::model;
namespace schedule = engine::schedule;

using model::Count;

inline constexpr auto layouts = std::array{schedule::image(0), schedule::image(1)};

/** One SM's share of an output tile's k units: tile-local units [first, first + count), in ascending k. */
export struct Segment {
	Count first;
	Count count;
};

/** One 128-row output block of one card, and its k units' SMs in SM order. */
export struct TileWork {
	model::Card card;
	Count block;
	std::vector<Segment> segments;
};

/** Every output block of one weight read in one phase, on both cards. */
export struct Product {
	Count tensor;
	schedule::Instance at;
	Count units_per_output;
	std::vector<TileWork> tiles;
};

[[nodiscard]] constexpr auto card_of(Count index) -> model::Card {
	return index == 0 ? model::Card::Zero : model::Card::One;
}

/** The image phase computing `op` of `stream`'s layer `layer`. */
[[nodiscard]] auto phase_of(model::Card card, schedule::Op op, model::Stream stream, Count layer) -> schedule::ImagePhase const& {
	auto const& layout = layouts.at(std::to_underlying(card));
	auto const found = std::ranges::find_if(layout.phases, [&](schedule::ImagePhase const& phase) {
		return phase.step.op == op && phase.step.stream == stream && phase.step.layer == layer;
	});
	core::invariant(found != std::ranges::end(layout.phases));
	return *found;
}

/** The selected 128-row blocks of a selection, in the order the image holds them. */
[[nodiscard]] auto selected_blocks(schedule::Blocks const& blocks) -> std::vector<Count> {
	auto out = std::vector<Count>{};
	for (auto group = Count{0}; group < blocks.groups; ++group) {
		for (auto taken = Count{0}; taken < blocks.take; ++taken) {
			out.push_back(blocks.first + (group * blocks.stride) + taken);
		}
	}
	return out;
}

/**
 * The product of `tensor` (the rows `selects` pick) in the phase computing `op` of `stream`'s layer `layer`, as
 * both cards run it: the phase's units split over the SMs by `schedule::share`.
 */
export [[nodiscard]] auto plan(schedule::Op op, model::Stream stream, Count layer, Count tensor, std::span<schedule::Select const> selects)
    -> Product {
	auto product = Product{
	    .tensor = tensor,
	    .at = {.stream = stream, .layer = layer},
	    .units_per_output = 0,
	    .tiles = {},
	};
	for (auto card_index = Count{0}; card_index < model::cards; ++card_index) {
		auto const card = card_of(card_index);
		auto const& layout = layouts.at(card_index);
		auto const& phase = phase_of(card, op, stream, layer);
		auto const units = schedule::run(layout, phase).units;
		auto ordinal = Count{0};
		auto found = false;
		for (auto index = phase.ring.begin; index < phase.ring.begin + phase.ring.length; ++index) {
			auto const& placement = layout.placements[index];
			auto const& read = placement.read;
			auto const blocks = selected_blocks(schedule::blocks(read.select, read.tensor[0], card_index));
			auto const reads_tensor = read.tensor[0] == tensor || (read.tensor_count == 2 && read.tensor[1] == tensor);
			if (!reads_tensor || std::ranges::find(selects, read.select) == selects.end()) {
				ordinal += blocks.size();
				continue;
			}
			found = true;
			auto const per_output = schedule::units_per_output(placement);
			product.units_per_output = per_output;
			for (auto const block : blocks) {
				auto work = TileWork{.card = card, .block = block, .segments = {}};
				auto const first = ordinal * per_output;
				for (auto sm = Count{0}; sm < schedule::sms; ++sm) {
					auto const share = schedule::share(units, sm);
					auto const begin = std::max(share.begin, first);
					auto const end = std::min(share.begin + share.length, first + per_output);
					if (begin < end) {
						work.segments.push_back({.first = begin - first, .count = end - begin});
					}
				}
				product.tiles.push_back(std::move(work));
				++ordinal;
			}
		}
		core::invariant(found);
	}
	return product;
}

/** Index `index` of a shard's axis, laid end to end, in the whole tensor's axis. */
[[nodiscard]] constexpr auto through(model::Shard const& shard, Count index) -> Count {
	auto remaining = index;
	for (auto range = Count{0}; range < shard.count; ++range) {
		if (remaining < shard.ranges[range].length) {
			return shard.ranges[range].begin + remaining;
		}
		remaining -= shard.ranges[range].length;
	}
	core::invariant(false);
	std::unreachable();
}

/** The checkpoint row card `card`'s row `row` of `tensor` holds. */
export [[nodiscard]] constexpr auto source_row(Count tensor, model::Card card, Count row) -> Count {
	auto const& entry = model::inventory[tensor];
	switch (entry.split) {
	case model::Split::Vocabulary:
		return vocabulary::token(card, row);
	case model::Split::Rows:
	case model::Split::DeltanetChannels:
		return through(model::shard(entry, std::to_underlying(card)), row);
	case model::Split::Replicated:
	case model::Split::Host:
	case model::Split::Columns:
	case model::Split::HiddenPair:
		return row;
	}
	std::unreachable();
}

/** Whether the cards' sums of `tensor` are partial sums over halves of k, added across the cards. */
export [[nodiscard]] constexpr auto summed_across_cards(Count tensor) -> bool {
	return model::split_axis(model::inventory[tensor].split) == 1;
}

/** The checkpoint column card `card`'s column `column` of `tensor` holds. */
export [[nodiscard]] constexpr auto source_column(Count tensor, model::Card card, Count column) -> Count {
	return summed_across_cards(tensor) ? through(model::shard(model::inventory[tensor], std::to_underlying(card)), column) : column;
}

}
