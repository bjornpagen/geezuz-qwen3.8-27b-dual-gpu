// SPDX-License-Identifier: AGPL-3.0-only
//
// The deltanet's arithmetic, layouts and kernel geometry re-exported to host code, so the emulation, the launch
// declarations and the tests compute with the very definitions the kernels compile. Device code cannot import
// modules, so these are device headers; this unit gives their declarations a module surface unchanged.
module;

#include "deltanet/geometry.h"
#include "deltanet/layout.h"
#include "deltanet/math.h"

export module engine.deltanet:arithmetic;

namespace engine::deltanet {

namespace device = engine::device::deltanet;

export using device::Address;
export using device::Bf16;
export using device::CheckParams;
export using device::Column;
export using device::DecodeParams;
export using device::DecodeShared;
export using device::Events;
export using device::Extent;
export using device::Four;
export using device::Gates;
export using device::PrefillParams;
export using device::PrefillShared;
export using device::ProbeParams;
export using device::Role;
export using device::RowOutputs;
export using device::Rows;
export using device::SlotChunk;
export using device::SlotCycle;
export using device::State;
export using device::StateLayout;
export using device::TokenBlock;
export using device::Tree;
export using device::Weights;

export using device::bf16_per_chunk;
export using device::channels;
export using device::check_blocks;
export using device::check_budget;
export using device::check_rounds;
export using device::decode_budget;
export using device::decode_items;
export using device::dim;
export using device::entry_bytes;
export using device::entry_chunks;
export using device::entry_gate_halves;
export using device::gate_row_bytes;
export using device::group;
export using device::group_channels;
export using device::group_chunks;
export using device::head_floats;
export using device::head_state_bytes;
export using device::key_heads;
export using device::l2_norm_eps;
export using device::layers;
export using device::max_drafts;
export using device::max_events;
export using device::max_replays;
export using device::max_rows;
export using device::output_row_bytes;
export using device::prefill_block_tokens;
export using device::prefill_budget;
export using device::prefill_items;
export using device::probe_blocks;
export using device::probe_budget;
export using device::probe_heads;
export using device::qkv_row_bytes;
export using device::query_channels;
export using device::query_divisor;
export using device::ring_bytes_per_slot;
export using device::slot_image_bytes;
export using device::state_image_bytes;
export using device::value_features;
export using device::value_heads;
export using device::window_budget;
export using device::window_group_bytes;
export using device::window_image_bytes;
export using device::window_items;
export using device::window_rows;

export using device::card_channel;
export using device::check_value;
export using device::combine;
export using device::convolve;
export using device::decay_and_project;
export using device::delta;
export using device::deltanet_layer;
export using device::gated;
export using device::gates;
export using device::l2_scale;
export using device::norm_scale;
export using device::normalize;
export using device::normalized;
export using device::partial_dot;
export using device::partial_squares;
export using device::read;
export using device::round_to_bf16;
export using device::ring_offset;
export using device::snapshot_part;
export using device::state_offset;
export using device::tree_sum;
export using device::update;
export using device::vector_element;
export using device::widen;
export using device::window_offset;

}
