// SPDX-License-Identifier: AGPL-3.0-only
//
// The deltanet kernels emulated on the host in fp32, in the device's own order: the same arithmetic functions, the same
// lanes holding the same elements, the same butterfly over each 128-element sum, and the same reads and writes of the
// device layouts. On the cards, a kernel run on generated inputs must reproduce these bits exactly; here, the tests hold
// these bits to the model's arithmetic in float64, and to each other across decode, replay and prefill.
export module engine.deltanet:emulation;

import std;
import engine.core;
import :arithmetic;

namespace engine::deltanet {

/** One slot's deltanet memory of one layer, in the device layouts: its slices of the state image and of the ring. */
export struct LayerMemory {
	/** Each value head's state, heads in order, in `State` order. */
	std::span<float> state;
	/** Each key head group's window rows, oldest first, each the group's channels. */
	std::span<Bf16> window;
	/** Each key head group's ring entries. */
	std::span<Bf16> ring;
};

/** Owned storage for one slot's deltanet memory of one layer. */
export struct LayerBuffers {
	std::vector<float> state = std::vector<float>(static_cast<std::size_t>(value_heads * head_floats));
	std::vector<Bf16> window = std::vector<Bf16>(static_cast<std::size_t>(key_heads * window_rows * group_channels));
	std::vector<Bf16> ring = std::vector<Bf16>(static_cast<std::size_t>(key_heads * max_drafts) * entry_bytes / sizeof(Bf16));

	[[nodiscard]] auto view() -> LayerMemory {
		return {.state = state, .window = window, .ring = ring};
	}

	[[nodiscard]] auto operator==(LayerBuffers const& other) const -> bool {
		auto const same_bits = [](Bf16 left, Bf16 right) {
			return left.bits == right.bits;
		};
		auto const same_floats = [](float left, float right) {
			return std::bit_cast<std::uint32_t>(left) == std::bit_cast<std::uint32_t>(right);
		};
		return std::ranges::equal(state, other.state, same_floats) && std::ranges::equal(window, other.window, same_bits) &&
		       std::ranges::equal(ring, other.ring, same_bits);
	}
};

/** One deltanet layer's side data on one card. */
export struct LayerWeights {
	/** [channels][4]. */
	std::span<Bf16 const> conv;
	/** One per value head. */
	std::span<Bf16 const> a_log;
	std::span<Bf16 const> dt_bias;
	/** [128]. */
	std::span<Bf16 const> norm;
};

/** Activation rows, as `Rows` lays them out. */
export struct Activations {
	/** [rows][channels]. */
	std::span<Bf16 const> qkv;
	/** [rows][value heads]: a in the low half, b in the high. */
	std::span<std::uint32_t const> gates;
	/** [rows][value features]. */
	std::span<Bf16> outputs;
};

using Lane = std::array<float, Column::per_lane>;
/** A group's value features, and a value head's query, key and value channels. */
constexpr auto group_columns = std::size_t{group} * std::size_t{dim};
constexpr auto head_columns = std::size_t{3} * std::size_t{dim};
using Lanes = std::array<Lane, Column::lanes>;
/** A 128-element fp32 vector in natural order. */
export using Vector = std::array<float, dim>;

[[nodiscard]] auto index(int value) -> std::size_t {
	return core::narrow<std::size_t>(value);
}

/** Tree lane `r`'s elements of a natural-order vector. */
[[nodiscard]] auto slice(Vector const& vector, int r) -> Lane {
	auto lane = Lane{};
	std::ranges::copy(std::span{vector}.subspan(index(r * Column::per_lane), Column::per_lane), lane.begin());
	return lane;
}

/** The tree's total of `lanes` partial sums, as `tree_sum` adds them. */
export template<std::size_t lanes>
[[nodiscard]] auto tree_total(std::array<float, lanes> const& partials) -> float {
	float values[lanes] = {};
	std::ranges::copy(partials, std::begin(values));
	return tree_sum(values);
}

/** The tree's total of each lane's share. */
template<class Share>
[[nodiscard]] auto grouped(Share share) -> float {
	float partials[Column::lanes] = {};
	for (auto r = 0; r < Column::lanes; ++r) {
		partials[r] = share(r);
	}
	return tree_sum(partials);
}

/** A convolved bf16 query or key normalized as the device's eight-lane group does it. */
export [[nodiscard]] auto normalized_vector(std::span<Bf16 const, dim> source, Role role) -> Vector {
	auto lanes = Lanes{};
	for (auto r = 0; r < Column::lanes; ++r) {
		for (auto m = 0; m < Column::per_lane; ++m) {
			lanes.at(index(r)).at(index(m)) = widen(source[index(r * Column::per_lane + m)]);
		}
	}
	float const total = grouped([&](int r) {
		float values[Column::per_lane] = {};
		std::ranges::copy(lanes.at(index(r)), std::begin(values));
		return partial_squares(values);
	});
	float const scale = l2_scale(total);
	auto result = Vector{};
	for (auto r = 0; r < Column::lanes; ++r) {
		float values[Column::per_lane] = {};
		std::ranges::copy(lanes.at(index(r)), std::begin(values));
		normalize(values, scale, role);
		auto const first = r * Column::per_lane;
		std::ranges::copy(values, result.begin() + first);
	}
	return result;
}

/** One state column as its eight lanes hold it. */
struct ColumnState {
	float lane[Column::lanes][Column::per_lane];
};

[[nodiscard]] auto load_column(std::span<float const> head, int column) -> ColumnState {
	auto state = ColumnState{};
	for (auto row = 0; row < dim; ++row) {
		state.lane[row / Column::per_lane][row % Column::per_lane] = head[index(State::element(row, column))];
	}
	return state;
}

void store_column(std::span<float> head, int column, ColumnState const& state) {
	for (auto row = 0; row < dim; ++row) {
		head[index(State::element(row, column))] = state.lane[row / Column::per_lane][row % Column::per_lane];
	}
}

/** One token's update of one column, the device's sequence of lane steps and butterflies; returns the column's output. */
[[nodiscard]] auto step(ColumnState& state, Vector const& key, Vector const& query, float value, Gates gate) -> float {
	float keys[Column::lanes][Column::per_lane] = {};
	float queries[Column::lanes][Column::per_lane] = {};
	for (auto r = 0; r < Column::lanes; ++r) {
		std::ranges::copy(slice(key, r), std::begin(keys[r]));
		std::ranges::copy(slice(query, r), std::begin(queries[r]));
	}
	float const projected = grouped([&](int r) {
		return decay_and_project(state.lane[r], keys[r], gate.decay);
	});
	float const column_delta = delta(value, projected, gate.beta);
	for (auto r = 0; r < Column::lanes; ++r) {
		update(state.lane[r], keys[r], column_delta);
	}
	return grouped([&](int r) {
		return read(state.lane[r], queries[r]);
	});
}

/** One head's outputs for one token, normalized and weighted as the device's eight-lane group does it. */
export void normalize_head(std::span<Bf16 const, dim> outputs, std::span<Bf16 const> weight, std::span<Bf16, dim> target) {
	float const total = grouped([&](int r) {
		float values[Column::per_lane] = {};
		for (auto m = 0; m < Column::per_lane; ++m) {
			values[m] = widen(outputs[index(r * Column::per_lane + m)]);
		}
		return partial_squares(values);
	});
	float const scale = norm_scale(total);
	for (auto feature = 0; feature < dim; ++feature) {
		target[index(feature)] = normalized(outputs[index(feature)], scale, weight[index(feature)]);
	}
}

[[nodiscard]] auto taps(std::span<Bf16 const> conv, int channel) -> Four {
	auto four = Four{};
	for (auto tap = 0; tap < 4; ++tap) {
		four.element[tap] = conv[index(channel * 4 + tap)];
	}
	return four;
}

using GroupRow = std::array<Bf16, group_channels>;

[[nodiscard]] auto window_row(LayerMemory const& memory, int key_head, int row) -> std::span<Bf16, group_channels> {
	return memory.window.subspan(index((key_head * window_rows + row) * group_channels)).first<group_channels>();
}

[[nodiscard]] auto ring_entry(LayerMemory const& memory, int key_head, int entry) -> std::span<Bf16, entry_bytes / sizeof(Bf16)> {
	return memory.ring.subspan(ring_offset(0, key_head, entry) / sizeof(Bf16)).first<entry_bytes / sizeof(Bf16)>();
}

[[nodiscard]] auto projected_row(Activations const& rows, int row, int key_head) -> GroupRow {
	auto result = GroupRow{};
	for (auto local = 0; local < group_channels; ++local) {
		result.at(index(local)) = rows.qkv[index(row * channels + card_channel(key_head, local))];
	}
	return result;
}

/** a then b for each value head of a group, as a ring entry holds them. */
using GateInputs = std::array<Bf16, entry_gate_halves>;

[[nodiscard]] auto projected_gates(Activations const& rows, int row, int key_head) -> GateInputs {
	auto result = GateInputs{};
	for (auto head = 0; head < group; ++head) {
		auto const word = rows.gates[index(row * value_heads + group * key_head + head)];
		result.at(index(head)) = {.bits = static_cast<std::uint16_t>(word & 0xffffU)};
		result.at(index(group + head)) = {.bits = static_cast<std::uint16_t>(word >> 16U)};
	}
	return result;
}

[[nodiscard]] auto head_state(LayerMemory const& memory, int value_head) -> std::span<float> {
	return memory.state.subspan(index(value_head * head_floats), index(head_floats));
}

/**
 * One slot's share of a decode cycle at one layer, as `deltanet_decode_core` computes it: each key head group replays
 * its ring, runs the rows, stores the state and window after the first row, keeps the drafts in the ring, and writes
 * each row's normalized output.
 */
export void decode(LayerWeights const& weights, LayerMemory const& memory, Activations const& rows, SlotCycle const& work) {
	core::invariant(work.replays <= static_cast<unsigned>(max_replays) && work.rows >= 1 && work.rows <= static_cast<unsigned>(max_rows));
	auto const replays = core::narrow<int>(work.replays);
	auto const events = replays + core::narrow<int>(work.rows);
	auto const first_row = core::narrow<int>(work.first_row);
	for (auto key_head = 0; key_head < key_heads; ++key_head) {
		// The window and every event's raw inputs, read before anything is written.
		auto sequence = std::vector<GroupRow>{};
		auto gate_inputs = std::vector<GateInputs>{};
		for (auto row = 0; row < window_rows; ++row) {
			auto copy = GroupRow{};
			std::ranges::copy(window_row(memory, key_head, row), copy.begin());
			sequence.push_back(copy);
		}
		for (auto event = 0; event < events; ++event) {
			if (event < replays) {
				auto const entry = ring_entry(memory, key_head, event);
				auto copy = GroupRow{};
				std::ranges::copy(entry.first<group_channels>(), copy.begin());
				sequence.push_back(copy);
				auto inputs = GateInputs{};
				std::ranges::copy(entry.subspan<group_channels, entry_gate_halves>(), inputs.begin());
				gate_inputs.push_back(inputs);
			} else {
				sequence.push_back(projected_row(rows, first_row + event - replays, key_head));
				gate_inputs.push_back(projected_gates(rows, first_row + event - replays, key_head));
			}
		}
		auto convolved = std::vector<GroupRow>(index(events));
		for (auto event = 0; event < events; ++event) {
			for (auto local = 0; local < group_channels; ++local) {
				auto const inputs =
				    Four{.element = {sequence.at(index(event)).at(index(local)), sequence.at(index(event + 1)).at(index(local)),
					                 sequence.at(index(event + 2)).at(index(local)), sequence.at(index(event + 3)).at(index(local))}};
				convolved.at(index(event)).at(index(local)) = convolve(taps(weights.conv, card_channel(key_head, local)), inputs);
			}
		}
		for (auto row = 0; row < window_rows; ++row) {
			std::ranges::copy(sequence.at(index(replays + 1 + row)), window_row(memory, key_head, row).begin());
		}
		for (auto event = replays + 1; event < events; ++event) {
			auto const entry = ring_entry(memory, key_head, event - replays - 1);
			std::ranges::copy(sequence.at(index(window_rows + event)), entry.begin());
			std::ranges::copy(gate_inputs.at(index(event)), entry.begin() + group_channels);
			std::ranges::fill(entry.subspan(group_channels + entry_gate_halves), Bf16{.bits = 0});
		}
		auto queries = std::vector<Vector>{};
		auto keys = std::vector<Vector>{};
		auto event_gates = std::vector<std::array<Gates, group>>{};
		for (auto event = 0; event < events; ++event) {
			auto const& convolved_row = convolved.at(index(event));
			queries.push_back(normalized_vector(std::span{convolved_row}.first<dim>(), Role::Query));
			keys.push_back(normalized_vector(std::span{convolved_row}.subspan<dim, dim>(), Role::Key));
			auto per_head = std::array<Gates, group>{};
			auto const& inputs = gate_inputs.at(index(event));
			for (auto head = 0; head < group; ++head) {
				auto const value_head = group * key_head + head;
				per_head.at(index(head)) = gates(inputs.at(index(head)), inputs.at(index(group + head)), weights.a_log[index(value_head)],
				                                 weights.dt_bias[index(value_head)]);
			}
			event_gates.push_back(per_head);
		}
		auto outputs = std::vector<std::array<Bf16, group_columns>>(std::size_t{work.rows});
		for (auto head = 0; head < group; ++head) {
			auto const state = head_state(memory, group * key_head + head);
			for (auto column = 0; column < dim; ++column) {
				auto lanes = load_column(state, column);
				for (auto event = 0; event < events; ++event) {
					auto const value = widen(convolved.at(index(event)).at(index(2 * dim + head * dim + column)));
					auto const output =
					    step(lanes, keys.at(index(event)), queries.at(index(event)), value, event_gates.at(index(event)).at(index(head)));
					if (event >= replays) {
						outputs.at(index(event - replays)).at(index(head * dim + column)) = round_to_bf16(output);
					}
					if (event == replays) {
						store_column(state, column, lanes);
					}
				}
			}
		}
		for (auto row = 0; row < core::narrow<int>(work.rows); ++row) {
			for (auto head = 0; head < group; ++head) {
				auto const target =
				    rows.outputs.subspan(index((first_row + row) * value_features + (group * key_head + head) * dim)).first<dim>();
				normalize_head(std::span{outputs.at(index(row))}.subspan(index(head * dim)).first<dim>(), weights.norm, target);
			}
		}
	}
}

/**
 * One slot's prompt chunk at one layer, as `deltanet_prefill` and then `deltanet_prefill_window` compute it: each value
 * head runs every token from the stored state and writes the normalized outputs and the state after the chunk; then
 * each group's window moves to the chunk's end.
 */
export void prefill(LayerWeights const& weights, LayerMemory const& memory, Activations const& rows, SlotChunk const& work) {
	core::invariant(work.tokens >= 1);
	auto const tokens = core::narrow<int>(work.tokens);
	auto const first_row = core::narrow<int>(work.first_row);
	for (auto value_head = 0; value_head < value_heads; ++value_head) {
		auto const key_head = value_head / group;
		auto const head = value_head % group;
		auto const locals = std::array{0, dim, 2 * dim + head * dim};
		// The query, key and own value channels of the head, sliding through the window and the chunk.
		auto sequence = std::vector<GroupRow>{};
		for (auto row = 0; row < window_rows; ++row) {
			auto copy = GroupRow{};
			std::ranges::copy(window_row(memory, key_head, row), copy.begin());
			sequence.push_back(copy);
		}
		for (auto token = 0; token < tokens; ++token) {
			sequence.push_back(projected_row(rows, first_row + token, key_head));
		}
		auto const state = head_state(memory, value_head);
		auto columns = std::vector<ColumnState>{};
		for (auto column = 0; column < dim; ++column) {
			columns.push_back(load_column(state, column));
		}
		for (auto token = 0; token < tokens; ++token) {
			auto convolved = std::array<Bf16, head_columns>{};
			for (auto part = 0; part < 3; ++part) {
				for (auto offset = 0; offset < dim; ++offset) {
					auto const local = locals.at(index(part)) + offset;
					auto const inputs =
					    Four{.element = {sequence.at(index(token)).at(index(local)), sequence.at(index(token + 1)).at(index(local)),
						                 sequence.at(index(token + 2)).at(index(local)), sequence.at(index(token + 3)).at(index(local))}};
					convolved.at(index(part * dim + offset)) = convolve(taps(weights.conv, card_channel(key_head, local)), inputs);
				}
			}
			auto const query = normalized_vector(std::span{convolved}.first<dim>(), Role::Query);
			auto const key = normalized_vector(std::span{convolved}.subspan<dim, dim>(), Role::Key);
			auto const inputs = projected_gates(rows, first_row + token, key_head);
			auto const gate = gates(inputs.at(index(head)), inputs.at(index(group + head)), weights.a_log[index(value_head)],
			                        weights.dt_bias[index(value_head)]);
			auto outputs = std::array<Bf16, dim>{};
			for (auto column = 0; column < dim; ++column) {
				auto const value = widen(convolved.at(index(2 * dim + column)));
				outputs.at(index(column)) = round_to_bf16(step(columns.at(index(column)), key, query, value, gate));
			}
			normalize_head(outputs, weights.norm,
			               rows.outputs.subspan(index((first_row + token) * value_features + value_head * dim)).first<dim>());
		}
		for (auto column = 0; column < dim; ++column) {
			store_column(state, column, columns.at(index(column)));
		}
	}
	for (auto key_head = 0; key_head < key_heads; ++key_head) {
		auto next = std::array<GroupRow, window_rows>{};
		for (auto row = 0; row < window_rows; ++row) {
			auto const position = tokens + row;
			if (position >= window_rows) {
				next.at(index(row)) = projected_row(rows, first_row + position - window_rows, key_head);
			} else {
				std::ranges::copy(window_row(memory, key_head, position), next.at(index(row)).begin());
			}
		}
		for (auto row = 0; row < window_rows; ++row) {
			std::ranges::copy(next.at(index(row)), window_row(memory, key_head, row).begin());
		}
	}
}

/** The chain probe's inputs, as `ProbeParams` lays them out for tree `T`. */
export struct ProbeInputs {
	std::span<float const> keys;
	std::span<float const> queries;
	std::span<float const> values;
	std::span<Gates const> gates;
	std::uint32_t tokens;
};

/** `deltanet_probe_chain_<lanes>`: the delta rule alone over every probe head, under tree `T`. */
export template<class T>
void probe(ProbeInputs const& inputs, std::span<float> state, std::span<float> outputs) {
	auto const tokens = core::narrow<int>(inputs.tokens);
	for (auto head = 0; head < probe_heads; ++head) {
		auto const head_floats_view = state.subspan(index(head * head_floats), index(head_floats));
		for (auto column = 0; column < dim; ++column) {
			float lanes[T::lanes][T::per_lane] = {};
			for (auto row = 0; row < dim; ++row) {
				lanes[row / T::per_lane][row % T::per_lane] = head_floats_view[index(StateLayout<T>::element(row, column))];
			}
			for (auto token = 0; token < tokens; ++token) {
				auto const row_index = head * tokens + token;
				float keys[T::lanes][T::per_lane] = {};
				float queries[T::lanes][T::per_lane] = {};
				for (auto element = 0; element < dim; ++element) {
					auto const at = index(row_index * dim + vector_element<T>(element));
					keys[element / T::per_lane][element % T::per_lane] = inputs.keys[at];
					queries[element / T::per_lane][element % T::per_lane] = inputs.queries[at];
				}
				auto const gate = inputs.gates[index(row_index)];
				float partials[T::lanes] = {};
				for (auto r = 0; r < T::lanes; ++r) {
					partials[r] = decay_and_project(lanes[r], keys[r], gate.decay);
				}
				float const column_delta = delta(inputs.values[index(row_index * dim + column)], tree_sum(partials), gate.beta);
				for (auto r = 0; r < T::lanes; ++r) {
					update(lanes[r], keys[r], column_delta);
					partials[r] = read(lanes[r], queries[r]);
				}
				outputs[index(row_index * dim + column)] = tree_sum(partials);
			}
			for (auto row = 0; row < dim; ++row) {
				head_floats_view[index(StateLayout<T>::element(row, column))] = lanes[row / T::per_lane][row % T::per_lane];
			}
		}
	}
}

/** Writes one layer's state and windows into a slot's state image, where the image keeps layer `layer`. */
export void store_layer(std::span<std::byte> image, int layer, LayerBuffers const& buffers) {
	core::invariant(image.size() == slot_image_bytes);
	std::ranges::copy(std::as_bytes(std::span{buffers.state}), image.begin() + core::narrow<std::ptrdiff_t>(state_offset(layer, 0)));
	std::ranges::copy(std::as_bytes(std::span{buffers.window}), image.begin() + core::narrow<std::ptrdiff_t>(window_offset(layer, 0)));
}

/** One layer's state and windows read from a slot's state image; the ring, which no image holds, reads zero. */
export [[nodiscard]] auto load_layer(std::span<std::byte const> image, int layer) -> LayerBuffers {
	core::invariant(image.size() == slot_image_bytes);
	auto buffers = LayerBuffers{};
	auto const state = std::as_writable_bytes(std::span{buffers.state});
	auto const window = std::as_writable_bytes(std::span{buffers.window});
	std::ranges::copy(image.subspan(state_offset(layer, 0), state.size()), state.begin());
	std::ranges::copy(image.subspan(window_offset(layer, 0), window.size()), window.begin());
	return buffers;
}

}
