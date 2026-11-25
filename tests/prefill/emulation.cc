// SPDX-License-Identifier: AGPL-3.0-only
//
// The equality prefill owes decode, emulated on the host at the model's real dimensions. Two emulated cards share a
// test double of the model: every weight and embedding a deterministic function of where the image (or the table)
// holds it, and every weight matmul a cheap row-wise double. A prompt is then run two ways:
//
// - prefill: the very launch sequences `engine.prefill` builds, executed launch by launch on emulated cards whose
//   memories are addressed as the kernels address them (the image's offsets, the workspace's layout, the exchange's
//   items, slots and flags, the kv pool's block tables), each kernel done the way its device code does it, through the
//   building blocks' own host emulations (attention's rows, the deltanet's prefill) and the model's rounding section;
// - decode: the same tokens one per cycle, written independently from the decode loop's tasks (one verify row a
//   cycle, the deltanet's decode with nothing to replay, attention appending then attending, the mtp catch-up of the
//   entry each new token completes), into its own emulated cards.
//
// The weight matmul is a double because its bits are the gemm's own business: the prefill gemm walks the decode's
// segments, and the gemv's tests prove both equal to one row-wise reference, which is all this equality relies on.
export module engine.prefill_emulation;

import std;
import engine.attn;
import engine.core;
import engine.deltanet;
import engine.mma_model;
import engine.model;
import engine.prefill;
import engine.prefill_device;
import engine.prims;
import engine.schedule;

namespace engine::prefill_emulation {

namespace pd = engine::prefill_device;
namespace attn_device = engine::device::attn;
namespace prims = engine::device::prims;

using model::Count;
using schedule::Op;
using schedule::Part;

export using prefill::Address;
export using prims::Bf16;

// ---------------------------------------------------------------------------------------------------------------------
// The emulated address space: a region per kind of memory and card, addressed as the kernels address them.

/** A kind of memory the launches name. */
export enum class Region : std::uint8_t {
	Image,
	Embedding,
	Tokens,
	Pool,
	Tables,
	States,
	Rotations,
	Prompt,
	Workspace,
};

inline constexpr std::uint64_t region_shift = 40;
inline constexpr std::uint64_t card_shift = 36;

export [[nodiscard]] constexpr auto base(Region region, model::Card card) -> Address {
	return (std::uint64_t{std::to_underlying(region)} + 1) << region_shift | std::uint64_t{std::to_underlying(card)} << card_shift;
}

struct Located {
	Region region;
	model::Card card;
	std::uint64_t offset;
};

[[nodiscard]] constexpr auto locate(Address address) -> Located {
	auto const region = (address >> region_shift) - 1;
	core::invariant(region <= std::to_underlying(Region::Workspace));
	return {.region = static_cast<Region>(region),
	        .card = static_cast<model::Card>((address >> card_shift) & 1U),
	        .offset = address & ((std::uint64_t{1} << card_shift) - 1)};
}

/** Uniform in [-1, 1) from a 64-bit hash. */
[[nodiscard]] constexpr auto unit(std::uint64_t hash) -> float {
	return static_cast<float>(static_cast<double>(hash >> 11U) * 0x1p-52 - 1.0);
}

[[nodiscard]] constexpr auto mix(std::uint64_t value) -> std::uint64_t {
	value += 0x9e3779b97f4a7c15ULL;
	value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
	value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
	return value ^ (value >> 31U);
}

template<class... Parts>
[[nodiscard]] constexpr auto hash(std::uint64_t seed, Parts... parts) -> std::uint64_t {
	auto value = mix(seed);
	((value = mix(value ^ static_cast<std::uint64_t>(parts))), ...);
	return value;
}

inline constexpr schedule::CardImage layout = schedule::image(0);

/** The model double: every weight, embedding and matmul output a function of a seed and where it is held. */
export class Model {
public:
	explicit Model(std::uint64_t seed) : seed_{seed} {
		for (auto const& placed : std::span{layout.placements}.first(schedule::placements)) {
			if (placed.read.part == Part::Whole) {
				sides_.push_back({.offset = placed.offset, .tensor = placed.read.tensor[0]});
			}
		}
		std::ranges::sort(sides_, {}, &Side::offset);
	}

	/** The bf16 at `offset` of card `card`'s image, where side data is held: in a range its tensor plausibly takes. */
	[[nodiscard]] auto side(model::Card card, std::uint64_t offset) const -> Bf16 {
		auto const found = std::ranges::upper_bound(sides_, offset, {}, &Side::offset);
		core::invariant(found != sides_.begin());
		auto const tensor = std::prev(found)->tensor;
		auto const& t = schedule::tensors;
		// A replicated tensor is the same on both cards; a split one holds each card's own part.
		auto const shard = model::inventory[tensor].split == model::Split::Replicated ? 0U : 1U + std::to_underlying(card);
		auto const value = unit(hash(seed_, 1, shard, offset));
		if (tensor == t.deltanet_norm) {
			return prims::round_to_bf16(1.0F + 0.25F * value);
		}
		if (tensor == t.a || tensor == t.b) {
			return prims::round_to_bf16(0.02F * value);
		}
		if (tensor == t.a_log || tensor == t.dt_bias) {
			return prims::round_to_bf16(value);
		}
		return prims::round_to_bf16(0.25F * value);
	}

	[[nodiscard]] auto embedding(std::uint64_t token, Count feature) const -> Bf16 {
		return prims::round_to_bf16(unit(hash(seed_, 2, token, feature)));
	}

	/**
	 * Output `output` of the matmul whose tiles start at `tiles` on card `card` (tensor `tensor` of an interleaved
	 * pair), for one activation row: four of its k, each weighted, folded by fma from +0. A row-wise function of the
	 * row alone, as every weight product is.
	 */
	[[nodiscard]] auto project(model::Card card, std::uint64_t tiles, Count tensor, Count output, std::span<Bf16 const> row) const
	    -> float {
		auto sum = 0.0F;
		for (auto term = std::uint64_t{0}; term < 4; ++term) {
			auto const at = hash(seed_, 3, std::to_underlying(card), tiles, tensor, output, term);
			auto const weight = 0.25F * unit(mix(at));
			sum = prims::math::fma(weight, prims::widen(row[at % row.size()]), sum);
		}
		return sum;
	}

private:
	struct Side {
		std::uint64_t offset;
		Count tensor;
	};

	std::uint64_t seed_;
	std::vector<Side> sides_;
};

/** Both emulated cards' memories: pages made on first write, every unwritten byte zero (as fresh kv blocks are). */
export class Memory {
public:
	static constexpr std::uint64_t page_bytes = std::uint64_t{64} * 1024;

	explicit Memory(Model const& model) : model_{&model} {}

	auto read(Address address, std::span<std::byte> out) const -> void {
		auto const at = locate(address);
		if (at.region == Region::Image || at.region == Region::Embedding) {
			// Generated regions hold bf16 values only; every read starts on one.
			core::invariant(at.offset % 2 == 0 && out.size() % 2 == 0);
			for (auto index = std::size_t{0}; index < out.size() / 2; ++index) {
				auto const value = generated(at, at.offset + 2 * index);
				out[2 * index] = static_cast<std::byte>(value.bits & 0xffU);
				out[2 * index + 1] = static_cast<std::byte>(value.bits >> 8U);
			}
			return;
		}
		for (auto done = std::size_t{0}; done < out.size();) {
			auto const position = address + done;
			auto const page = position / page_bytes;
			auto const within = position % page_bytes;
			auto const take = std::min<std::size_t>(out.size() - done, page_bytes - within);
			if (auto const found = pages_.find(page); found != pages_.end()) {
				std::ranges::copy(std::span{found->second}.subspan(within, take), out.begin() + core::narrow<std::ptrdiff_t>(done));
			} else {
				std::ranges::fill(out.subspan(done, take), std::byte{0});
			}
			done += take;
		}
	}

	auto write(Address address, std::span<std::byte const> in) -> void {
		auto const region = locate(address).region;
		core::invariant(region != Region::Image && region != Region::Embedding);
		for (auto done = std::size_t{0}; done < in.size();) {
			auto const position = address + done;
			auto const page = position / page_bytes;
			auto const within = position % page_bytes;
			auto const take = std::min<std::size_t>(in.size() - done, page_bytes - within);
			auto& bytes = pages_.try_emplace(page, std::vector<std::byte>(page_bytes)).first->second;
			std::ranges::copy(in.subspan(done, take), bytes.begin() + core::narrow<std::ptrdiff_t>(within));
			done += take;
		}
	}

	template<class T>
	[[nodiscard]] auto values(Address address, std::size_t count) const -> std::vector<T> {
		auto out = std::vector<T>(count);
		read(address, std::as_writable_bytes(std::span{out}));
		return out;
	}

	template<class T>
	[[nodiscard]] auto value(Address address) const -> T {
		auto out = std::array<T, 1>{};
		read(address, std::as_writable_bytes(std::span{out}));
		return out[0];
	}

	template<class T>
	auto store(Address address, std::span<T const> values) -> void {
		write(address, std::as_bytes(values));
	}

	template<class T>
	auto store_one(Address address, T value) -> void {
		write(address, std::as_bytes(std::span{&value, 1}));
	}

	/** Every written page of one region of one card, by address; pages written only with zeros count as unwritten. */
	[[nodiscard]] auto same(Memory const& other, Region region, model::Card card) const -> bool {
		auto const low = base(region, card) / page_bytes;
		auto const high = (base(region, card) + (std::uint64_t{1} << card_shift)) / page_bytes;
		auto const zero = std::vector<std::byte>(page_bytes);
		auto const page_of = [&](Memory const& memory, std::uint64_t page) -> std::vector<std::byte> const& {
			auto const found = memory.pages_.find(page);
			return found == memory.pages_.end() ? zero : found->second;
		};
		auto pages = std::set<std::uint64_t>{};
		for (auto const* memory : {this, &other}) {
			for (auto const& [page, bytes] : memory->pages_) {
				if (page >= low && page < high) {
					pages.insert(page);
				}
			}
		}
		return std::ranges::all_of(pages, [&](std::uint64_t page) {
			return page_of(*this, page) == page_of(other, page);
		});
	}

private:
	[[nodiscard]] auto generated(Located const& at, std::uint64_t offset) const -> Bf16 {
		if (at.region == Region::Image) {
			return model_->side(at.card, offset);
		}
		auto const row_bytes = model::hidden * sizeof(Bf16);
		return model_->embedding(offset / row_bytes, (offset % row_bytes) / sizeof(Bf16));
	}

	/* SAFETY: the model outlives every memory the tests build on it. */
	Model const* model_;
	std::map<std::uint64_t, std::vector<std::byte>> pages_;
};

// ---------------------------------------------------------------------------------------------------------------------
// Shared helpers: the norms and the attention rows, as both runs compute a row.

[[nodiscard]] auto widened(std::span<Bf16 const> values) -> std::vector<float> {
	auto out = std::vector<float>(values.size());
	std::ranges::transform(values, out.begin(), prims::widen);
	return out;
}

[[nodiscard]] auto rounded(std::span<float const> values) -> std::vector<Bf16> {
	auto out = std::vector<Bf16>(values.size());
	std::ranges::transform(values, out.begin(), prims::round_to_bf16);
	return out;
}

/** x * factor * (1 + w), rounded once, from a row's sum of squares. */
[[nodiscard]] auto normalized(std::span<float const> x, float squares, std::span<Bf16 const> weight) -> std::vector<Bf16> {
	auto const factor = model::rms_factor(squares, model::hidden);
	auto out = std::vector<Bf16>(model::hidden);
	for (auto index = Count{0}; index < model::hidden; ++index) {
		out[index] = prims::round_to_bf16(model::one_plus_scaled(x[index], factor, prims::widen(weight[index])));
	}
	return out;
}

/** Where the kv row of `position` starts in the slot's blocks: its 32 chunks of 8 features follow, swizzled. */
[[nodiscard]] auto kv_row(Memory const& memory, Address pool, Address tables, std::uint32_t slot, Count layer, Count head,
                          attn_device::Kind kind, Count position) -> Address {
	auto const block = memory.value<std::uint32_t>(tables + (Count{slot} * model::kv_table_blocks + position / model::kv_block_tokens) * 4);
	return pool + Count{block} * model::kv_block_bytes + attn_device::row_offset(layer, head, kind, position);
}

auto store_kv(Memory& memory, Address pool, Address tables, std::uint32_t slot, Count layer, Count head, attn_device::Kind kind,
              Count position, attn::Row const& row) -> void {
	auto stored = attn::Row{};
	for (auto feature = Count{0}; feature < attn_device::head_dim; ++feature) {
		auto const chunk = attn_device::stored_chunk(position, feature / attn_device::features_per_chunk);
		stored[chunk * attn_device::features_per_chunk + feature % attn_device::features_per_chunk] = row[feature];
	}
	memory.store<Bf16>(kv_row(memory, pool, tables, slot, layer, head, kind, position), stored);
}

[[nodiscard]] auto load_row(Memory const& memory, Address pool, Address tables, std::uint32_t slot, Count layer, Count head,
                            attn_device::Kind kind, Count position) -> attn::Row {
	auto stored = attn::Row{};
	memory.read(kv_row(memory, pool, tables, slot, layer, head, kind, position), std::as_writable_bytes(std::span{stored}));
	auto row = attn::Row{};
	for (auto feature = Count{0}; feature < attn_device::head_dim; ++feature) {
		auto const chunk = attn_device::stored_chunk(position, feature / attn_device::features_per_chunk);
		row[feature] = stored[chunk * attn_device::features_per_chunk + feature % attn_device::features_per_chunk];
	}
	return row;
}

/** Keys and values of positions [0, end) of one kv head; positions never appended read zero. */
struct KvRows {
	std::vector<attn::Row> keys;
	std::vector<attn::Row> values;
};

[[nodiscard]] auto load_kv(Memory const& memory, Address pool, Address tables, std::uint32_t slot, Count layer, Count head, Count end)
    -> KvRows {
	auto rows = KvRows{.keys = std::vector<attn::Row>(end), .values = std::vector<attn::Row>(end)};
	for (auto position = Count{0}; position < end; ++position) {
		rows.keys[position] = load_row(memory, pool, tables, slot, layer, head, attn_device::Kind::Key, position);
		rows.values[position] = load_row(memory, pool, tables, slot, layer, head, attn_device::Kind::Value, position);
	}
	return rows;
}

/** One row's gated attention output over its query heads, its query and gate rows fp32 and its kv in the pool. */
[[nodiscard]] auto attend_row(Memory const& memory, Address pool, Address tables, std::uint32_t slot, Count layer,
                              std::span<float const> query, std::span<float const> gate, std::span<Bf16 const> q_norm,
                              std::uint32_t position) -> std::vector<Bf16> {
	auto out = std::vector<Bf16>(attn_device::output_row_values);
	auto const end = (Count{position} / attn_device::chunk_positions + 1) * attn_device::chunk_positions;
	for (auto head = Count{0}; head < attn_device::kv_heads; ++head) {
		auto const kv = load_kv(memory, pool, tables, slot, layer, head, end);
		for (auto member = Count{0}; member < attn_device::group_heads; ++member) {
			auto const q = head * attn_device::group_heads + member;
			auto const features = std::span<float const>{query}.subspan(q * attn_device::head_dim).first<attn_device::head_dim>();
			auto const rotated = attn::prepare_rotated(
			    features, std::span<Bf16 const, attn_device::head_dim>{q_norm.first(attn_device::head_dim)}, position);
			auto const state = attn::attend(rotated, {.keys = kv.keys, .values = kv.values}, position, attn_device::PvParts::Three);
			auto const gated =
			    attn::finish(state, std::span<float const>{gate}.subspan(q * attn_device::head_dim).first<attn_device::head_dim>());
			std::ranges::copy(gated, out.begin() + core::narrow<std::ptrdiff_t>(q * attn_device::head_dim));
		}
	}
	return out;
}

/** One row's keys and values appended at `position`: k_norm and the rope, then each rounded once. */
auto append_row(Memory& memory, Address pool, Address tables, std::uint32_t slot, Count layer, std::span<float const> key,
                std::span<float const> value, std::span<Bf16 const> k_norm, std::uint32_t position) -> void {
	for (auto head = Count{0}; head < attn_device::kv_heads; ++head) {
		auto const k = std::span<float const>{key}.subspan(head * attn_device::head_dim).first<attn_device::head_dim>();
		auto const v = std::span<float const>{value}.subspan(head * attn_device::head_dim).first<attn_device::head_dim>();
		store_kv(memory, pool, tables, slot, layer, head, attn_device::Kind::Key, position,
		         attn::prepare_rotated(k, std::span<Bf16 const, attn_device::head_dim>{k_norm.first(attn_device::head_dim)}, position));
		store_kv(memory, pool, tables, slot, layer, head, attn_device::Kind::Value, position, attn::prepare_value(v));
	}
}

/** The deltanet layer's side data on one card, owned. */
struct DeltanetSides {
	std::vector<Bf16> conv;
	std::vector<Bf16> a_log;
	std::vector<Bf16> dt_bias;
	std::vector<Bf16> norm;

	[[nodiscard]] auto view() const -> deltanet::LayerWeights {
		return {.conv = conv, .a_log = a_log, .dt_bias = dt_bias, .norm = norm};
	}
};

[[nodiscard]] auto deltanet_sides(Memory const& memory, deltanet::Weights const& weights) -> DeltanetSides {
	auto const at = [](auto address) {
		return std::bit_cast<Address>(address.value);
	};
	return {.conv = memory.values<Bf16>(at(weights.conv), static_cast<std::size_t>(deltanet::channels) * 4),
	        .a_log = memory.values<Bf16>(at(weights.a_log), static_cast<std::size_t>(deltanet::value_heads)),
	        .dt_bias = memory.values<Bf16>(at(weights.dt_bias), static_cast<std::size_t>(deltanet::value_heads)),
	        .norm = memory.values<Bf16>(at(weights.norm), static_cast<std::size_t>(deltanet::dim))};
}

/** A slot's deltanet memory of one layer, read out of (and written back into) its state image. */
[[nodiscard]] auto load_layer(Memory const& memory, Address image, int layer) -> deltanet::LayerBuffers {
	auto buffers = deltanet::LayerBuffers{};
	memory.read(image + deltanet::state_offset(layer, 0), std::as_writable_bytes(std::span{buffers.state}));
	memory.read(image + deltanet::window_offset(layer, 0), std::as_writable_bytes(std::span{buffers.window}));
	return buffers;
}

auto store_layer(Memory& memory, Address image, int layer, deltanet::LayerBuffers const& buffers) -> void {
	memory.write(image + deltanet::state_offset(layer, 0), std::as_bytes(std::span{buffers.state}));
	memory.write(image + deltanet::window_offset(layer, 0), std::as_bytes(std::span{buffers.window}));
}

/** in_proj_a and in_proj_b of one row: one mma chain over the whole hidden each, fp32 a then b per value head. */
[[nodiscard]] auto decay_inputs(Memory const& memory, Address a, Address b, std::span<Bf16 const> row) -> std::vector<float> {
	auto out = std::vector<float>(static_cast<std::size_t>(deltanet::value_heads) * 2);
	for (auto head = 0; head < deltanet::value_heads; ++head) {
		for (auto const [which, matrix] : std::views::enumerate(std::array{a, b})) {
			auto const weights = memory.values<Bf16>(matrix + static_cast<Count>(head) * model::hidden * 2, model::hidden);
			out[static_cast<std::size_t>(head) * 2 + static_cast<std::size_t>(which)] = mma_model::chain<model::hidden>(
			    0.0F, std::span<Bf16 const, model::hidden>{weights}, std::span<Bf16 const, model::hidden>{row.first(model::hidden)});
		}
	}
	return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// The prefill run: launches executed on emulated cards.

/** Where the emulated cards' memories are, as the decode side would hand them to a run. */
export [[nodiscard]] constexpr auto memories() -> prefill::Memories {
	auto const card = [](model::Card which) {
		return prefill::CardMemories{
		    .image = base(Region::Image, which),
		    .pool = base(Region::Pool, which),
		    .tables = base(Region::Tables, which),
		    .states = base(Region::States, which),
		    .rotations = base(Region::Rotations, which),
		    .prompt = base(Region::Prompt, which),
		};
	};
	return {.embedding = base(Region::Embedding, model::Card::Zero), .cards = {card(model::Card::Zero), card(model::Card::One)}};
}

export [[nodiscard]] constexpr auto view(model::Card card) -> prefill::CardView {
	auto const index = std::to_underlying(card);
	return {
	    .card = card,
	    .memories = memories().cards[index],
	    .workspace = base(Region::Workspace, card),
	    .peer_workspace = base(Region::Workspace, card == model::Card::Zero ? model::Card::One : model::Card::Zero),
	    .embedding = memories().embedding,
	    .tokens = base(Region::Tokens, model::Card::Zero),
	};
}

/** One emulated card executing launches: each kernel as its device code computes it, over the rows it names. */
struct Executor {
	Model const& model;
	Memory& memory;
	model::Card card;

	[[nodiscard]] auto row_bf16(Address buffer, Count width, Count row) const -> std::vector<Bf16> {
		return memory.values<Bf16>(buffer + row * width * 2, width);
	}

	[[nodiscard]] auto row_f32(Address buffer, Count width, Count row) const -> std::vector<float> {
		return memory.values<float>(buffer + row * width * 4, width);
	}

	/** The squares of a row's 128-feature tiles, stored [tile][max_rows]. */
	auto store_squares(Address squares, Count row, std::span<float const> values) -> void {
		for (auto tile = Count{0}; tile < pd::hidden_tiles; ++tile) {
			memory.store_one(squares + (tile * pd::max_rows + row) * 4,
			                 model::lane_sum<model::hidden_tile>(values, values, tile * schedule::tile_rows));
		}
	}

	[[nodiscard]] auto row_squares(Address squares, Count row) const -> float {
		auto total = memory.value<float>(squares + row * 4);
		for (auto tile = Count{1}; tile < pd::hidden_tiles; ++tile) {
			total = total + memory.value<float>(squares + (tile * pd::max_rows + row) * 4);
		}
		return total;
	}

	auto operator()(prefill::Launch<pd::Zero> const& step) -> void {
		auto const zeros = std::vector<std::byte>(std::size_t{16} * 4096);
		for (auto done = std::uint64_t{0}; done < step.params.chunks * 16; done += zeros.size()) {
			memory.write(step.params.target + done,
			             std::span{zeros}.first(std::min<std::size_t>(zeros.size(), step.params.chunks * 16 - done)));
		}
	}

	auto operator()(prefill::Launch<pd::Embed> const& step) -> void {
		auto const& p = step.params;
		for (auto row = Count{0}; row < p.rows.computed; ++row) {
			auto values = std::vector<float>(model::hidden);
			if (row < p.rows.count) {
				auto const token = memory.value<std::uint32_t>(p.tokens + row * 4);
				values = widened(memory.values<Bf16>(p.embedding + Count{token} * model::hidden * 2, model::hidden));
			}
			memory.store<float>(p.workspace + pd::workspace.residual + row * model::hidden * 4, values);
			store_squares(p.workspace + pd::workspace.squares, row, values);
		}
	}

	auto operator()(prefill::Launch<pd::Norm> const& step) -> void {
		auto const& p = step.params;
		rows = p.rows.count;
		auto const weight = memory.values<Bf16>(p.weight, model::hidden);
		for (auto row = Count{0}; row < p.rows.computed; ++row) {
			auto out = std::vector<Bf16>(model::hidden);
			if (row < p.rows.count) {
				out = normalized(row_f32(p.residual, model::hidden, row), row_squares(p.squares, row), weight);
			}
			memory.store<Bf16>(p.out + row * model::hidden * 2, out);
		}
	}

	auto operator()(prefill::Launch<pd::Decay> const& step) -> void {
		auto const& p = step.params;
		for (auto row = Count{0}; row < p.rows.count; ++row) {
			auto const inputs = decay_inputs(memory, p.a, p.b, row_bf16(p.workspace + pd::workspace.normed, model::hidden, row));
			memory.store<float>(p.workspace + pd::workspace.gates + row * deltanet::gate_row_bytes, inputs);
		}
	}

	/** A gemm's outputs for the chunk's rows: per row, `width` outputs of tensor `tensor` of the placement. */
	[[nodiscard]] auto project(pd::Weights const& weights, Count tensor, Address activations, Count k, Count width, Count row) const
	    -> std::vector<float> {
		auto const in = row_bf16(activations, k, row);
		auto out = std::vector<float>(width);
		for (auto output = Count{0}; output < width; ++output) {
			out[output] = model.project(card, locate(weights.tiles).offset, tensor, output, in);
		}
		return out;
	}

	/** Rows of a gemm launch: the computed rows hold the chunk's, and padding rows feed only padding rows. */
	[[nodiscard]] static auto live_rows(unsigned row_blocks, Count count) -> Count {
		core::invariant(count <= Count{row_blocks} * pd::block_rows);
		return count;
	}

	/** The rows of the gemms that follow: the last norm's, whose rows they read. */
	Count rows = 0;

	auto store_f32(pd::GemmParams const& p, Count offset, Count width) -> void {
		for (auto row = Count{0}; row < live_rows(p.row_blocks, rows); ++row) {
			memory.store<float>(p.workspace + offset + row * width * 4,
			                    project(p.weights, 0, p.workspace + pd::workspace.normed, model::hidden, width, row));
		}
	}

	auto operator()(prefill::Launch<pd::DeltanetIn> const& step) -> void {
		auto const& p = step.params;
		for (auto row = Count{0}; row < live_rows(p.row_blocks, rows); ++row) {
			auto const out =
			    project(p.weights, 0, p.workspace + pd::workspace.normed, model::hidden, static_cast<Count>(deltanet::channels), row);
			memory.store<Bf16>(p.workspace + pd::workspace.qkv + row * deltanet::qkv_row_bytes, rounded(out));
		}
	}

	auto operator()(prefill::Launch<pd::DeltanetZ> const& step) -> void {
		auto const& p = step.params;
		auto const width = static_cast<Count>(deltanet::value_features);
		for (auto row = Count{0}; row < live_rows(p.row_blocks, rows); ++row) {
			auto const z = project(p.weights, 0, p.workspace + pd::workspace.normed, model::hidden, width, row);
			auto const core = row_f32(p.workspace + pd::workspace.core, width, row);
			auto gated = std::vector<Bf16>(width);
			for (auto index = Count{0}; index < width; ++index) {
				gated[index] = prims::round_to_bf16(model::deltanet::gated(core[index], z[index]));
			}
			memory.store<Bf16>(p.workspace + pd::workspace.gated + row * width * 2, gated);
		}
	}

	auto operator()(prefill::Launch<pd::AttentionQuery> const& step) -> void {
		store_f32(step.params, pd::workspace.query, attn_device::query_row_floats);
	}

	auto operator()(prefill::Launch<pd::AttentionKey> const& step) -> void {
		store_f32(step.params, pd::workspace.key, attn_device::kv_row_floats);
	}

	auto operator()(prefill::Launch<pd::AttentionValue> const& step) -> void {
		store_f32(step.params, pd::workspace.value, attn_device::kv_row_floats);
	}

	auto operator()(prefill::Launch<pd::AttentionGate> const& step) -> void {
		store_f32(step.params, pd::workspace.attention_gate, attn_device::gate_row_floats);
	}

	auto operator()(prefill::Launch<pd::MtpAttentionQuery> const& step) -> void {
		store_f32(step.params, pd::workspace.query, attn_device::query_row_floats);
	}

	auto operator()(prefill::Launch<pd::MtpAttentionKey> const& step) -> void {
		store_f32(step.params, pd::workspace.key, attn_device::kv_row_floats);
	}

	auto operator()(prefill::Launch<pd::MtpAttentionValue> const& step) -> void {
		store_f32(step.params, pd::workspace.value, attn_device::kv_row_floats);
	}

	auto operator()(prefill::Launch<pd::MtpAttentionGate> const& step) -> void {
		store_f32(step.params, pd::workspace.attention_gate, attn_device::gate_row_floats);
	}

	auto mlp_in(pd::GemmParams const& p) -> void {
		for (auto row = Count{0}; row < live_rows(p.row_blocks, rows); ++row) {
			auto const gate = project(p.weights, 0, p.workspace + pd::workspace.normed, model::hidden, model::ffn_per_card, row);
			auto const up = project(p.weights, 1, p.workspace + pd::workspace.normed, model::hidden, model::ffn_per_card, row);
			auto product = std::vector<Bf16>(model::ffn_per_card);
			for (auto index = Count{0}; index < model::ffn_per_card; ++index) {
				product[index] = prims::round_to_bf16(model::mlp_product(gate[index], up[index]));
			}
			memory.store<Bf16>(p.workspace + pd::workspace.product + row * model::ffn_per_card * 2, product);
		}
	}

	auto operator()(prefill::Launch<pd::MlpIn> const& step) -> void {
		mlp_in(step.params);
	}

	auto operator()(prefill::Launch<pd::MtpMlpIn> const& step) -> void {
		mlp_in(step.params);
	}

	/** A row-parallel gemm: each item as this card's partial and in the peer's receive slot, its flag released. */
	auto push(pd::PushParams const& p, Address activations, Count k) -> void {
		auto const slot = pd::exchange_slot(p.epoch);
		auto const peer_receive = p.peer_workspace + pd::workspace.receive + slot * pd::exchange_floats * 4;
		auto const peer_flags = p.peer_workspace + pd::workspace.flags + slot * pd::exchange_items * schedule::flag_bytes;
		for (auto row = Count{0}; row < live_rows(p.row_blocks, rows); ++row) {
			auto const out = project(p.weights, 0, activations, k, model::hidden, row);
			for (auto tile = Count{0}; tile < pd::hidden_tiles; ++tile) {
				auto const part = std::span{out}.subspan(tile * schedule::tile_rows, schedule::tile_rows);
				auto const at = pd::exchange_offset(tile, row) * 4;
				memory.store<float>(p.workspace + pd::workspace.own + at, part);
				memory.store<float>(peer_receive + at, part);
			}
		}
		for (auto item = Count{0}; item < pd::hidden_tiles * pd::max_row_blocks; ++item) {
			if (item % pd::max_row_blocks < p.row_blocks) {
				memory.store_one(peer_flags + item * schedule::flag_bytes, std::uint64_t{p.epoch});
			}
		}
	}

	auto operator()(prefill::Launch<pd::DeltanetOut> const& step) -> void {
		push(step.params, step.params.workspace + pd::workspace.gated, static_cast<Count>(deltanet::value_features));
	}

	auto operator()(prefill::Launch<pd::AttentionOut> const& step) -> void {
		push(step.params, step.params.workspace + pd::workspace.attention_output, attn_device::output_row_values);
	}

	auto operator()(prefill::Launch<pd::MtpAttentionOut> const& step) -> void {
		push(step.params, step.params.workspace + pd::workspace.attention_output, attn_device::output_row_values);
	}

	auto operator()(prefill::Launch<pd::MlpOut> const& step) -> void {
		push(step.params, step.params.workspace + pd::workspace.product, model::ffn_per_card);
	}

	auto operator()(prefill::Launch<pd::MtpMlpOut> const& step) -> void {
		push(step.params, step.params.workspace + pd::workspace.product, model::ffn_per_card);
	}

	auto operator()(prefill::Launch<pd::MtpFc> const& step) -> void {
		push(step.params, step.params.workspace + pd::workspace.mtp_input, model::hidden);
	}

	/** Residual + (own + peer), or own + peer starting it, once the peer's item arrived; with the new squares. */
	auto meet(pd::ResidualParams const& p, bool start) -> void {
		auto const slot = pd::exchange_slot(p.epoch);
		for (auto row = Count{0}; row < p.rows.count; ++row) {
			auto residual = row_f32(p.workspace + pd::workspace.residual, model::hidden, row);
			for (auto tile = Count{0}; tile < pd::hidden_tiles; ++tile) {
				auto const flag = memory.value<std::uint64_t>(p.workspace + pd::workspace.flags +
				                                              (slot * pd::exchange_items + pd::item_index(tile, row / pd::block_rows)) *
				                                                  schedule::flag_bytes);
				// The peer's push of this exchange ran before this card's residual: the flag must already say so.
				core::invariant(flag == p.epoch);
				auto const at = pd::exchange_offset(tile, row) * 4;
				auto const own = memory.values<float>(p.workspace + pd::workspace.own + at, schedule::tile_rows);
				auto const peer =
				    memory.values<float>(p.workspace + pd::workspace.receive + slot * pd::exchange_floats * 4 + at, schedule::tile_rows);
				for (auto output = Count{0}; output < schedule::tile_rows; ++output) {
					auto const sum = own[output] + peer[output];
					auto& value = residual[tile * schedule::tile_rows + output];
					value = start ? sum : value + sum;
				}
			}
			memory.store<float>(p.workspace + pd::workspace.residual + row * model::hidden * 4, residual);
			store_squares(p.workspace + pd::workspace.squares, row, residual);
		}
	}

	auto operator()(prefill::Launch<pd::ResidualAdd> const& step) -> void {
		meet(step.params, false);
	}

	auto operator()(prefill::Launch<pd::ResidualStart> const& step) -> void {
		meet(step.params, true);
	}

	auto operator()(prefill::Launch<pd::MtpGather> const& step) -> void {
		auto const& p = step.params;
		for (auto row = Count{0}; row < p.rows.computed; ++row) {
			auto embedded = std::vector<float>(model::hidden);
			auto hidden = std::vector<Bf16>(model::hidden);
			if (row < p.rows.count) {
				auto const entry = p.first + row;
				auto const token = memory.value<std::uint32_t>(p.tokens + (entry + 1 - p.from) * 4);
				embedded = widened(memory.values<Bf16>(p.embedding + Count{token} * model::hidden * 2, model::hidden));
				auto const source = entry < p.from ? p.workspace + pd::workspace.carried + Count{p.slot} * model::hidden * 2
				                                   : p.workspace + pd::workspace.final_normed + Count{entry - p.from} * model::hidden * 2;
				hidden = memory.values<Bf16>(source, model::hidden);
			}
			memory.store<float>(p.workspace + pd::workspace.mtp_embedding + row * model::hidden * 4, embedded);
			memory.store<Bf16>(p.workspace + pd::workspace.mtp_hidden + row * model::hidden * 2, hidden);
			store_squares(p.workspace + pd::workspace.embedding_squares, row, embedded);
			store_squares(p.workspace + pd::workspace.hidden_squares, row, widened(hidden));
		}
	}

	auto operator()(prefill::Launch<pd::MtpNorm> const& step) -> void {
		auto const& p = step.params;
		rows = p.rows.count;
		auto const half = model::hidden / model::cards;
		auto const norms =
		    std::array{memory.values<Bf16>(p.embedding_norm, model::hidden), memory.values<Bf16>(p.hidden_norm, model::hidden)};
		for (auto row = Count{0}; row < p.rows.computed; ++row) {
			auto out = std::vector<Bf16>(model::hidden);
			if (row < p.rows.count) {
				auto const inputs = std::array{row_f32(p.workspace + pd::workspace.mtp_embedding, model::hidden, row),
				                               widened(row_bf16(p.workspace + pd::workspace.mtp_hidden, model::hidden, row))};
				auto const squares = std::array{row_squares(p.workspace + pd::workspace.embedding_squares, row),
				                                row_squares(p.workspace + pd::workspace.hidden_squares, row)};
				for (auto input = std::size_t{0}; input < 2; ++input) {
					auto const full = normalized(inputs.at(input), squares.at(input), norms.at(input));
					std::ranges::copy(std::span{full}.subspan(Count{p.card} * half, half),
					                  out.begin() + core::narrow<std::ptrdiff_t>(input * half));
				}
			}
			memory.store<Bf16>(p.workspace + pd::workspace.mtp_input + row * model::hidden * 2, out);
		}
	}

	auto operator()(prefill::Launch<pd::Keep> const& step) -> void {
		auto const& p = step.params;
		auto const row = row_bf16(p.workspace + pd::workspace.final_normed, model::hidden, p.row);
		memory.store<Bf16>(p.workspace + pd::workspace.carried + Count{p.slot} * model::hidden * 2, row);
		memory.store<Bf16>(p.prompt, row);
	}

	auto operator()(prefill::Launch<attn::Append> const& step) -> void {
		auto const& p = step.params;
		auto const k_norm = memory.values<Bf16>(p.preparation.key, attn_device::head_dim);
		for (auto row = Count{0}; row < p.count; ++row) {
			append_row(memory, p.kv.blocks, p.kv.tables, p.slot, p.kv.layer, row_f32(p.rows.key, attn_device::kv_row_floats, row),
			           row_f32(p.rows.value, attn_device::kv_row_floats, row), k_norm, core::narrow<std::uint32_t>(p.start + row));
		}
	}

	auto operator()(prefill::Launch<attn::Prefill> const& step) -> void {
		auto const& p = step.params;
		auto const q_norm = memory.values<Bf16>(p.preparation.query, attn_device::head_dim);
		for (auto row = Count{0}; row < p.count; ++row) {
			auto const out =
			    attend_row(memory, p.kv.blocks, p.kv.tables, p.slot, p.kv.layer, row_f32(p.rows.query, attn_device::query_row_floats, row),
				           row_f32(p.rows.gate, attn_device::gate_row_floats, row), q_norm, core::narrow<std::uint32_t>(p.start + row));
			memory.store<Bf16>(p.output + row * attn_device::output_row_values * 2, out);
		}
	}

	/** The deltanet's prefill core and window move, as its emulation runs them together. */
	auto operator()(prefill::Launch<deltanet::Prefill> const& step) -> void {
		auto const& p = step.params;
		auto const work = p.slots[0];
		core::invariant(p.live == 1 && work.first_row == 0);
		auto const image = std::bit_cast<Address>(p.images.value) + Count{work.slot} * deltanet::slot_image_bytes;
		auto buffers = load_layer(memory, image, static_cast<int>(p.layer));
		auto const sides = deltanet_sides(memory, p.weights);
		auto const qkv =
		    memory.values<Bf16>(std::bit_cast<Address>(p.rows.qkv.value), work.tokens * static_cast<std::size_t>(deltanet::channels));
		auto const gates = memory.values<float>(std::bit_cast<Address>(p.rows.gates.value),
		                                        work.tokens * static_cast<std::size_t>(deltanet::value_heads) * 2);
		auto outputs = std::vector<float>(work.tokens * static_cast<std::size_t>(deltanet::value_features));
		deltanet::prefill(sides.view(), buffers.view(), {.qkv = qkv, .gates = gates, .outputs = outputs}, work);
		store_layer(memory, image, static_cast<int>(p.layer), buffers);
		memory.store<float>(std::bit_cast<Address>(p.rows.outputs.value), outputs);
	}

	auto operator()(prefill::Launch<deltanet::PrefillWindow> const& /*step*/) -> void {}
};

/** Runs one launch on one card. */
struct Run {
	Executor& executor;

	auto operator()(prefill::PrefillLaunch const& step) -> void {
		std::visit(executor, step);
	}

	auto operator()(prefill::AttentionLaunch const& step) -> void {
		std::visit(executor, step);
	}

	auto operator()(prefill::DeltanetLaunch const& step) -> void {
		std::visit(executor, step);
	}
};

/** Both cards' sequences of one chunk, launch by launch in lockstep, as both streams order them. */
export auto execute(Model const& model, Memory& memory, std::array<std::vector<prefill::Step>, model::cards> const& steps) -> void {
	core::invariant(steps[0].size() == steps[1].size());
	auto executors = std::array{Executor{.model = model, .memory = memory, .card = model::Card::Zero, .rows = 0},
	                            Executor{.model = model, .memory = memory, .card = model::Card::One, .rows = 0}};
	for (auto index = std::size_t{0}; index < steps[0].size(); ++index) {
		for (auto card = std::size_t{0}; card < model::cards; ++card) {
			// The mtp pass computes fewer rows than the chunk: each launch names its own.
			auto& executor = executors.at(card);
			std::visit(Run{.executor = executor}, steps.at(card).at(index));
		}
	}
}

// ---------------------------------------------------------------------------------------------------------------------
// The decode run: one token per cycle, written from the decode loop's tasks.

/** One card's projection of one row through the placement `ring` of image phase `phase`. */
[[nodiscard]] auto project(Model const& model, model::Card card, Count phase, Count ring, Count tensor, std::span<Bf16 const> row,
                           Count width) -> std::vector<float> {
	auto out = std::vector<float>(width);
	for (auto output = Count{0}; output < width; ++output) {
		out[output] = model.project(card, pd::image_table.phase[phase].ring[ring], tensor, output, row);
	}
	return out;
}

[[nodiscard]] auto side_values(Memory const& memory, model::Card card, Count phase, Count index, Count count) -> std::vector<Bf16> {
	return memory.values<Bf16>(base(Region::Image, card) + pd::image_table.phase[phase].side[index], count);
}

/** residual + (card 0 + card 1), element by element: the decode's `add_to_residual`. */
auto add_cards(std::vector<float>& residual, std::array<std::vector<float>, model::cards> const& sums) -> void {
	for (auto index = Count{0}; index < model::hidden; ++index) {
		residual[index] = residual[index] + (sums[0][index] + sums[1][index]);
	}
}

/** The decode side's per-slot facts: its tokens so far and each layer's deltanet memory, ring included. */
export class Decoder {
public:
	Decoder(Model const& model, Memory& memory, std::uint32_t slot, std::span<Count const> layers)
	    : model_{&model}, memory_{&memory}, slot_{slot}, layers_{layers.begin(), layers.end()} {}

	/** One cycle: the token at `position` through the layers and the final norm; the mtp entry it completes. */
	auto step(std::uint32_t token, std::uint32_t position) -> void {
		auto residual =
		    widened(memory_->values<Bf16>(base(Region::Embedding, model::Card::Zero) + Count{token} * model::hidden * 2, model::hidden));
		for (auto const layer : layers_) {
			auto const& phases = pd::layers.trunk[layer];
			switch (phases.mixer) {
			case model::Mixer::LinearAttention:
				deltanet_layer(residual, layer, phases);
				break;
			case model::Mixer::FullAttention:
				attention_layer(residual, phases, prefill_kv_layer(layer), position);
				break;
			}
			mlp(residual, phases);
		}
		auto const normed = normalized(residual, model::hidden_squares(residual),
		                               side_values(*memory_, model::Card::Zero, pd::head_phase,
		                                           pd::side_index(Op::Head, schedule::tensors.final_norm, Part::Whole), model::hidden));
		// The entry before this token's position waits for this token: a prompt's first token completes none.
		if (position > 0) {
			mtp_entry(last_, token, position - 1);
		}
		last_ = normed;
	}

	/** The final norm of the last token stepped: what the prompt's end reads. */
	[[nodiscard]] auto last() const -> std::vector<Bf16> const& {
		return last_;
	}

private:
	[[nodiscard]] static auto prefill_kv_layer(Count layer) -> unsigned {
		auto ordinal = 0U;
		for (auto index = Count{0}; index < layer; ++index) {
			ordinal += model::trunk_mixers[index] == model::Mixer::FullAttention ? 1U : 0U;
		}
		return ordinal;
	}

	[[nodiscard]] auto norm_of(std::span<float const> residual, Count phase, Count index) const -> std::vector<Bf16> {
		return normalized(residual, model::hidden_squares(residual), side_values(*memory_, model::Card::Zero, phase, index, model::hidden));
	}

	auto deltanet_layer(std::vector<float>& residual, Count layer, pd::LayerPhases const& phases) -> void {
		auto const& t = schedule::tensors;
		auto const normed = norm_of(residual, phases.mixer_in, pd::side_index(Op::DeltanetIn, t.input_norm, Part::Whole));
		auto sums = std::array<std::vector<float>, model::cards>{};
		for (auto const card : {model::Card::Zero, model::Card::One}) {
			auto const qkv = rounded(project(*model_, card, phases.mixer_in, 0, 0, normed, static_cast<Count>(deltanet::channels)));
			auto const image = base(Region::Image, card);
			auto const gates = decay_inputs(
			    *memory_, image + pd::image_table.phase[phases.mixer_in].side[pd::side_index(Op::DeltanetIn, t.a, Part::Whole)],
			    image + pd::image_table.phase[phases.mixer_in].side[pd::side_index(Op::DeltanetIn, t.b, Part::Whole)], normed);
			auto const at = [&](Count tensor) {
				return deltanet::address<deltanet::Bf16 const>(image +
				                                               pd::image_table.phase[phases.deltanet_core].side[side_of_core(tensor)]);
			};
			auto const sides =
			    deltanet_sides(*memory_, {.conv = at(t.conv), .a_log = at(t.a_log), .dt_bias = at(t.dt_bias), .norm = at(t.deltanet_norm)});
			auto& buffers = layer_memory(card, layer);
			auto core = std::vector<float>(static_cast<std::size_t>(deltanet::value_features));
			deltanet::decode(sides.view(), buffers.view(), {.qkv = qkv, .gates = gates, .outputs = core},
			                 {.slot = slot_, .replays = 0, .rows = 1, .first_row = 0});
			store_layer(*memory_, base(Region::States, card) + Count{slot_} * deltanet::slot_image_bytes, deltanet::deltanet_layer(layer),
			            buffers);
			auto const z = project(*model_, card, phases.deltanet_z, 0, 0, normed, static_cast<Count>(deltanet::value_features));
			auto gated = std::vector<Bf16>(core.size());
			for (auto index = std::size_t{0}; index < core.size(); ++index) {
				gated[index] = prims::round_to_bf16(model::deltanet::gated(core[index], z[index]));
			}
			sums.at(std::to_underlying(card)) = project(*model_, card, phases.mixer_out, 0, 0, gated, model::hidden);
		}
		add_cards(residual, sums);
	}

	[[nodiscard]] static auto side_of_core(Count tensor) -> Count {
		auto const& t = schedule::tensors;
		if (tensor == t.conv) {
			return pd::side_index(Op::DeltanetCore, t.conv, Part::Whole);
		}
		if (tensor == t.a_log) {
			return pd::side_index(Op::DeltanetCore, t.a_log, Part::Whole);
		}
		if (tensor == t.dt_bias) {
			return pd::side_index(Op::DeltanetCore, t.dt_bias, Part::Whole);
		}
		return pd::side_index(Op::DeltanetCore, t.deltanet_norm, Part::Whole);
	}

	[[nodiscard]] auto layer_memory(model::Card card, Count layer) -> deltanet::LayerBuffers& {
		auto& buffers = deltanet_[std::pair{std::to_underlying(card), layer}];
		return buffers;
	}

	/** Attention for one row at `position`: append its kv, attend over [0, position], gate, then o_proj on both cards. */
	auto attention_layer(std::vector<float>& residual, pd::LayerPhases const& phases, unsigned kv, std::uint32_t position) -> void {
		auto const& t = schedule::tensors;
		auto const normed = norm_of(residual, phases.mixer_in, pd::side_index(Op::AttentionIn, t.input_norm, Part::Whole));
		auto sums = std::array<std::vector<float>, model::cards>{};
		for (auto const card : {model::Card::Zero, model::Card::One}) {
			auto const query = project(*model_, card, phases.mixer_in, 0, 0, normed, attn_device::query_row_floats);
			auto const key = project(*model_, card, phases.mixer_in, 1, 0, normed, attn_device::kv_row_floats);
			auto const value = project(*model_, card, phases.mixer_in, 2, 0, normed, attn_device::kv_row_floats);
			auto const gate = project(*model_, card, phases.attention_gate, 0, 0, normed, attn_device::gate_row_floats);
			auto const q_norm =
			    side_values(*memory_, card, phases.mixer_in, pd::side_index(Op::AttentionIn, t.q_norm, Part::Whole), attn_device::head_dim);
			auto const k_norm =
			    side_values(*memory_, card, phases.mixer_in, pd::side_index(Op::AttentionIn, t.k_norm, Part::Whole), attn_device::head_dim);
			auto const pool = base(Region::Pool, card);
			auto const tables = base(Region::Tables, card);
			append_row(*memory_, pool, tables, slot_, kv, key, value, k_norm, position);
			auto const out = attend_row(*memory_, pool, tables, slot_, kv, query, gate, q_norm, position);
			sums.at(std::to_underlying(card)) = project(*model_, card, phases.mixer_out, 0, 0, out, model::hidden);
		}
		add_cards(residual, sums);
	}

	auto mlp(std::vector<float>& residual, pd::LayerPhases const& phases) -> void {
		auto const normed = norm_of(residual, phases.mlp_in, pd::side_index(Op::MlpIn, schedule::tensors.post_norm, Part::Whole));
		auto sums = std::array<std::vector<float>, model::cards>{};
		for (auto const card : {model::Card::Zero, model::Card::One}) {
			auto const gate = project(*model_, card, phases.mlp_in, 0, 0, normed, model::ffn_per_card);
			auto const up = project(*model_, card, phases.mlp_in, 0, 1, normed, model::ffn_per_card);
			auto product = std::vector<Bf16>(model::ffn_per_card);
			for (auto index = Count{0}; index < model::ffn_per_card; ++index) {
				product[index] = prims::round_to_bf16(model::mlp_product(gate[index], up[index]));
			}
			sums.at(std::to_underlying(card)) = project(*model_, card, phases.mlp_out, 0, 0, product, model::hidden);
		}
		add_cards(residual, sums);
	}

	/** The mtp entry at `position`: its hidden input, the embedding of the token after it, its decoder layer. */
	auto mtp_entry(std::vector<Bf16> const& hidden, std::uint32_t next, std::uint32_t position) -> void {
		auto const& t = schedule::tensors;
		auto const embedded =
		    widened(memory_->values<Bf16>(base(Region::Embedding, model::Card::Zero) + Count{next} * model::hidden * 2, model::hidden));
		auto const hidden_wide = widened(hidden);
		auto const half = model::hidden / model::cards;
		auto residual = std::vector<float>(model::hidden);
		auto sums = std::array<std::vector<float>, model::cards>{};
		for (auto const card : {model::Card::Zero, model::Card::One}) {
			auto const embedding_norm = side_values(*memory_, card, pd::layers.mtp_in,
			                                        pd::side_index(Op::MtpIn, t.pre_fc_norm_embedding, Part::Whole), model::hidden);
			auto const hidden_norm =
			    side_values(*memory_, card, pd::layers.mtp_in, pd::side_index(Op::MtpIn, t.pre_fc_norm_hidden, Part::Whole), model::hidden);
			auto const e = normalized(embedded, model::hidden_squares(embedded), embedding_norm);
			auto const h = normalized(hidden_wide, model::hidden_squares(hidden_wide), hidden_norm);
			auto input = std::vector<Bf16>{};
			auto const own = Count{std::to_underlying(card)} * half;
			input.insert(input.end(), e.begin() + core::narrow<std::ptrdiff_t>(own), e.begin() + core::narrow<std::ptrdiff_t>(own + half));
			input.insert(input.end(), h.begin() + core::narrow<std::ptrdiff_t>(own), h.begin() + core::narrow<std::ptrdiff_t>(own + half));
			sums.at(std::to_underlying(card)) = project(*model_, card, pd::layers.mtp_in, 0, 0, input, model::hidden);
		}
		for (auto index = Count{0}; index < model::hidden; ++index) {
			residual[index] = sums[0][index] + sums[1][index];
		}
		attention_layer(residual, pd::layers.mtp, prefill::mtp_kv_layer, position);
		mlp(residual, pd::layers.mtp);
	}

	/* SAFETY: the model and the memory outlive the decoder; the tests hold all three in one scope. */
	Model const* model_;
	Memory* memory_;
	std::uint32_t slot_;
	std::vector<Count> layers_;
	std::map<std::pair<std::uint8_t, Count>, deltanet::LayerBuffers> deltanet_;
	/** The final norm of the last token stepped. */
	std::vector<Bf16> last_;
};

}
