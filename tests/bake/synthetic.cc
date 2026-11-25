// SPDX-License-Identifier: AGPL-3.0-only
//
// Test doubles for the bake: a tiny plan with every split, part, selection and interleave the model's plan uses (two
// trunk layers and an mtp layer, a permuted vocabulary, a host tensor), its three catalogs, and a writer of synthetic
// safetensors checkpoints for it, with the faults a real checkpoint can have.
export module engine.bake_synthetic;

import std;
import engine.bake;
import engine.core;
import engine.file_write;
import engine.image;
import engine.model;
import engine.schedule;
import engine.sha256;
import engine.testing;

namespace engine::bake_synthetic {

using model::Format;
using model::Scope;
using model::Split;
using schedule::Part;
using schedule::Select;

inline constexpr std::size_t hidden = 256;
inline constexpr std::size_t vocab = 1024;

export inline constexpr std::array<model::Tensor, 13> inventory = {{
    {.name = "model.language_model.embed_tokens.weight", .shape = model::matrix(vocab, hidden), .scope = Scope::Model, .format = Format::Bf16, .split = Split::Host},
    {.name = "model.language_model.norm.weight", .shape = model::vector(hidden), .scope = Scope::Model, .format = Format::Bf16, .split = Split::Replicated},
    {.name = "lm_head.weight", .shape = model::matrix(vocab, hidden), .scope = Scope::Model, .format = Format::Bf16, .split = Split::Vocabulary},
    {.name = "fc.weight", .shape = model::matrix(hidden, 2 * hidden), .scope = Scope::Mtp, .format = Format::Bf16, .split = Split::Columns},
    {.name = "input_layernorm.weight", .shape = model::vector(hidden), .scope = Scope::DecoderLayer, .format = Format::Bf16, .split = Split::Replicated},
    {.name = "self_attn.q_proj.weight", .shape = model::matrix(1024, hidden), .scope = Scope::DecoderLayer, .format = Format::E4m3, .split = Split::Rows},
    {.name = "self_attn.v_proj.weight", .shape = model::matrix(512, hidden), .scope = Scope::DecoderLayer, .format = Format::E4m3, .split = Split::Rows},
    {.name = "mlp.gate_proj.weight", .shape = model::matrix(512, hidden), .scope = Scope::DecoderLayer, .format = Format::E4m3, .split = Split::Rows},
    {.name = "mlp.up_proj.weight", .shape = model::matrix(512, hidden), .scope = Scope::DecoderLayer, .format = Format::E4m3, .split = Split::Rows},
    {.name = "mlp.down_proj.weight", .shape = model::matrix(hidden, 512), .scope = Scope::DecoderLayer, .format = Format::E4m3, .split = Split::Columns},
    {.name = "linear_attn.A_log", .shape = model::vector(64), .scope = Scope::DecoderLayer, .format = Format::Bf16, .split = Split::Rows},
    {.name = "linear_attn.in_proj_a.weight", .shape = model::matrix(8, 512), .scope = Scope::DecoderLayer, .format = Format::Bf16, .split = Split::Columns},
    {.name = "linear_attn.conv1d.weight", .shape = model::depthwise(64, 4), .scope = Scope::DecoderLayer, .format = Format::Bf16, .split = Split::Rows},
}};

namespace index {
inline constexpr std::size_t embed = 0;
inline constexpr std::size_t final_norm = 1;
inline constexpr std::size_t lm_head = 2;
inline constexpr std::size_t fc = 3;
inline constexpr std::size_t input_norm = 4;
inline constexpr std::size_t q = 5;
inline constexpr std::size_t v = 6;
inline constexpr std::size_t gate = 7;
inline constexpr std::size_t up = 8;
inline constexpr std::size_t down = 9;
inline constexpr std::size_t a_log = 10;
inline constexpr std::size_t in_proj_a = 11;
inline constexpr std::size_t conv = 12;
}

/** The decoder layers the tiny model has: two in the trunk, one in the mtp head. */
export inline constexpr std::array<schedule::Instance, 3> layers = {{
    {.stream = model::Stream::Body, .layer = 0},
    {.stream = model::Stream::Body, .layer = 1},
    {.stream = model::Stream::Mtp, .layer = 0},
}};

inline constexpr schedule::Instance head = {.stream = model::Stream::Head, .layer = 0};
inline constexpr schedule::Instance mtp = {.stream = model::Stream::Mtp, .layer = 0};

[[nodiscard]] constexpr auto one(std::size_t tensor, Part part, Select select) -> schedule::Read {
	return {.tensor = {tensor, tensor}, .tensor_count = 1, .part = part, .select = select};
}

[[nodiscard]] constexpr auto rank_of(model::Card card) -> std::size_t {
	return std::to_underlying(card);
}

/** The tiny plan's selections: a q_proj head is two blocks, query then gate; each card's draft rows are one block. */
export [[nodiscard]] constexpr auto tiny_blocks(Select select, std::size_t tensor, model::Card card) -> schedule::Blocks {
	auto const total = model::card_shape(std::span{inventory}[tensor], rank_of(card)).extents[0] / schedule::tile_rows;
	switch (select) {
	case Select::All:
		return {.first = 0, .take = total, .stride = total, .groups = 1};
	case Select::Query:
		return {.first = 0, .take = 1, .stride = 2, .groups = total / 2};
	case Select::Gate:
		return {.first = 1, .take = 1, .stride = 2, .groups = total / 2};
	case Select::DraftVocabulary:
		return {.first = 0, .take = 1, .stride = total, .groups = 1};
	case Select::OtherVocabulary:
		return {.first = 1, .take = total - 1, .stride = total, .groups = 1};
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto bytes_of(schedule::Read const& read, std::size_t tensor, model::Card card) -> std::size_t {
	auto const& entry = std::span{inventory}[tensor];
	auto const shape = model::card_shape(entry, rank_of(card));
	auto const blocks = tiny_blocks(read.select, tensor, card);
	auto const chosen = blocks.take * blocks.groups;
	switch (read.part) {
	case Part::Tiles:
		return chosen * schedule::tile_rows * shape.extents[1] * model::element_bytes(entry.format);
	case Part::Scales:
		return chosen * model::divide_up(shape.extents[1], model::scale_block) * 2;
	case Part::Whole:
		return model::card_bytes(entry, rank_of(card));
	}
	std::unreachable();
}

inline constexpr std::size_t placement_count = 49;

/** The tiny image of a card: every layer's tiles, the head's and the mtp head's, then the side data. */
[[nodiscard]] consteval auto place(model::Card card) -> std::array<schedule::Placement, placement_count> {
	auto reads = std::vector<std::pair<schedule::Read, schedule::Instance>>{};
	for (auto const& at : layers) {
		reads.emplace_back(one(index::q, Part::Tiles, Select::Query), at);
		reads.emplace_back(one(index::v, Part::Tiles, Select::All), at);
		reads.emplace_back(schedule::Read{.tensor = {index::gate, index::up}, .tensor_count = 2, .part = Part::Tiles, .select = Select::All}, at);
		reads.emplace_back(one(index::q, Part::Tiles, Select::Gate), at);
		reads.emplace_back(one(index::down, Part::Tiles, Select::All), at);
	}
	reads.emplace_back(one(index::lm_head, Part::Tiles, Select::DraftVocabulary), head);
	reads.emplace_back(one(index::lm_head, Part::Tiles, Select::OtherVocabulary), head);
	reads.emplace_back(one(index::fc, Part::Tiles, Select::All), mtp);
	for (auto const& at : layers) {
		reads.emplace_back(one(index::input_norm, Part::Whole, Select::All), at);
		reads.emplace_back(one(index::q, Part::Scales, Select::Query), at);
		reads.emplace_back(one(index::q, Part::Scales, Select::Gate), at);
		reads.emplace_back(one(index::v, Part::Scales, Select::All), at);
		reads.emplace_back(one(index::gate, Part::Scales, Select::All), at);
		reads.emplace_back(one(index::up, Part::Scales, Select::All), at);
		reads.emplace_back(one(index::down, Part::Scales, Select::All), at);
		reads.emplace_back(one(index::a_log, Part::Whole, Select::All), at);
		reads.emplace_back(one(index::in_proj_a, Part::Whole, Select::All), at);
		reads.emplace_back(one(index::conv, Part::Whole, Select::All), at);
	}
	reads.emplace_back(one(index::final_norm, Part::Whole, Select::All), head);
	auto placements = std::array<schedule::Placement, placement_count>{};
	auto extents = std::vector<image::Extent>{};
	for (auto index = std::size_t{0}; index < reads.size(); ++index) {
		auto const& [read, at] = reads.at(index);
		auto bytes = std::size_t{0};
		for (auto const tensor : std::span{read.tensor}.first(read.tensor_count)) {
			bytes += bytes_of(read, tensor, card);
		}
		placements.at(index) = {.read = read, .instance = at, .offset = 0, .bytes = bytes};
		extents.push_back({.identity = index + 1, .bytes = bytes, .part = read.part});
	}
	auto const offsets = image::place(extents);
	for (auto index = std::size_t{0}; index < placements.size(); ++index) {
		placements.at(index).offset = offsets.at(index).offset;
	}
	return placements;
}

inline constexpr std::array<std::array<schedule::Placement, placement_count>, 2> card_placements = {place(model::Card::Zero), place(model::Card::One)};

template<model::Card card>
[[nodiscard]] consteval auto extents() -> std::array<image::Extent, placement_count> {
	auto result = std::array<image::Extent, placement_count>{};
	for (auto index = std::size_t{0}; index < placement_count; ++index) {
		auto const& placement = card_placements.at(rank_of(card)).at(index);
		result.at(index) = {.identity = index + 1, .bytes = placement.bytes, .part = placement.read.part};
	}
	return result;
}

inline constexpr sha256::Digest tiny_description = sha256::digest("tiny bake plan: a test double");

template<model::Card card>
struct TinyCard {
	static constexpr image::Target target = image::card_target(card);
	static constexpr std::array<image::Extent, placement_count> extents = bake_synthetic::extents<card>();
	static constexpr sha256::Digest description = tiny_description;
};

struct TinyHost {
	static constexpr image::Target target = image::Target::Host;
	static constexpr std::array<image::Extent, 1> extents = {{
	    {.identity = 1, .bytes = vocab * hidden * 2, .part = Part::Whole},
	}};
	static constexpr sha256::Digest description = tiny_description;
};

/** The tiny plan: a test double of the model's own. */
export struct Tiny {
	static constexpr auto const& inventory = bake_synthetic::inventory;

	using CardZero = TinyCard<model::Card::Zero>;
	using CardOne = TinyCard<model::Card::One>;
	using Host = TinyHost;

	static constexpr std::array<std::size_t, 1> host_tensors = {index::embed};

	[[nodiscard]] static constexpr auto placements(model::Card card) -> std::span<schedule::Placement const> {
		return std::span{card_placements.at(rank_of(card))};
	}

	[[nodiscard]] static constexpr auto blocks(Select select, std::size_t tensor, model::Card card) -> schedule::Blocks {
		return tiny_blocks(select, tensor, card);
	}

	/** A permutation of the vocabulary, interleaving the cards: 389 is odd, so it is a bijection modulo 1024. */
	[[nodiscard]] static constexpr auto token(model::Card card, std::size_t row) -> std::size_t {
		return (((2 * row) + rank_of(card)) * 389) % vocab;
	}

	static constexpr std::size_t v_proj = index::v;
	static constexpr std::size_t input_norm = index::input_norm;
};

static_assert(bake::Consistent<Tiny>);

// ---------------------------------------------------------------------------------------------------------------------
// Synthetic checkpoints.

/** A clean checkpoint. */
export struct Clean {};

/** One e4m3 code of a tensor replaced by the nan code 0x7f. */
export struct NanAt {
	std::string tensor;
	std::size_t row;
	std::size_t column;
};

/** A tensor that belongs to no part of the model, outside the vision tower. */
export struct Extra {};

/** A tensor stored with its rows and columns swapped. */
export struct Reshaped {
	std::string tensor;
};

/** A tensor left out. */
export struct Omitted {
	std::string tensor;
};

/** Every v_proj code and scale one value, and every input norm weight zero: the attention bound is exact arithmetic. */
export struct Uniform {
	std::uint8_t code;
	std::uint16_t scale;
};

export using Fault = std::variant<Clean, NanAt, Extra, Reshaped, Omitted, Uniform>;

struct Stored {
	std::string name;
	std::string_view dtype;
	std::vector<std::size_t> shape;
	std::vector<std::byte> bytes;
};

/** A random finite e4m3 code. */
[[nodiscard]] auto finite_code(std::mt19937_64& random) -> std::byte {
	while (true) {
		auto const code = static_cast<std::uint8_t>(random() & 0xffU);
		if ((code & 0x7fU) != 0x7fU) {
			return std::byte{code};
		}
	}
}

/** A random bf16 of moderate magnitude, either sign. */
[[nodiscard]] auto bf16_value(std::mt19937_64& random, unsigned low_exponent, unsigned high_exponent) -> std::uint16_t {
	auto const exponent = std::uniform_int_distribution<unsigned>{low_exponent, high_exponent}(random);
	auto const mantissa = static_cast<unsigned>(random() & 0x7fU);
	auto const sign = static_cast<unsigned>(random() & 1U) << 15U;
	return static_cast<std::uint16_t>(sign | (exponent << 7U) | mantissa);
}

auto push_bf16(std::vector<std::byte>& bytes, std::uint16_t value) -> void {
	bytes.push_back(std::byte{static_cast<std::uint8_t>(value & 0xffU)});
	bytes.push_back(std::byte{static_cast<std::uint8_t>(value >> 8U)});
}

[[nodiscard]] auto shape_of(model::Shape const& shape) -> std::vector<std::size_t> {
	auto const all = std::span{shape.extents}.first(shape.rank);
	return {all.begin(), all.end()};
}

/** Every tensor copy of the tiny model with its name, the e4m3 ones followed by their scales, before any fault. */
[[nodiscard]] auto tensors(std::uint64_t seed, Fault const& fault) -> std::vector<Stored> {
	auto random = engine::testing::generator(seed);
	auto const uniform = core::alternative<Uniform>(fault);
	auto stored = std::vector<Stored>{};
	auto const add = [&](std::size_t tensor, schedule::Instance const& at) {
		auto const& entry = std::span{inventory}[tensor];
		auto const name = bake::checkpoint_name(entry, at);
		auto const elements = model::elements(entry.shape);
		auto bytes = std::vector<std::byte>{};
		auto const is_v = tensor == index::v && uniform;
		auto const is_norm = tensor == index::input_norm && uniform;
		switch (entry.format) {
		case Format::E4m3: {
			for (auto element = std::size_t{0}; element < elements; ++element) {
				bytes.push_back(is_v ? std::byte{uniform->code} : finite_code(random));
			}
			stored.push_back({.name = name, .dtype = "F8_E4M3", .shape = shape_of(entry.shape), .bytes = std::move(bytes)});
			auto const scales = model::scale_shape(entry.shape);
			auto scale_bytes = std::vector<std::byte>{};
			for (auto element = std::size_t{0}; element < model::elements(scales); ++element) {
				push_bf16(scale_bytes, is_v ? uniform->scale : static_cast<std::uint16_t>(bf16_value(random, 117, 123) & 0x7fffU));
			}
			stored.push_back({.name = bake::scale_name(name), .dtype = "BF16", .shape = shape_of(scales), .bytes = std::move(scale_bytes)});
			break;
		}
		case Format::Bf16:
			for (auto element = std::size_t{0}; element < elements; ++element) {
				push_bf16(bytes, is_norm ? std::uint16_t{0} : bf16_value(random, 112, 130));
			}
			stored.push_back({.name = name, .dtype = "BF16", .shape = shape_of(entry.shape), .bytes = std::move(bytes)});
			break;
		}
	};
	for (auto const tensor : {index::embed, index::final_norm, index::lm_head}) {
		add(tensor, head);
	}
	add(index::fc, mtp);
	for (auto const& at : layers) {
		for (auto tensor = index::input_norm; tensor < inventory.size(); ++tensor) {
			add(tensor, at);
		}
	}
	return stored;
}

/** Applies the faults that change stored tensors in place. */
auto apply(std::vector<Stored>& stored, Fault const& fault) -> void {
	if (auto const nan = core::alternative<NanAt>(fault)) {
		auto const found = std::ranges::find(stored, nan->tensor, &Stored::name);
		found->bytes.at((nan->row * found->shape.at(1)) + nan->column) = std::byte{0x7f};
	}
	if (auto const reshaped = core::alternative<Reshaped>(fault)) {
		auto const found = std::ranges::find(stored, reshaped->tensor, &Stored::name);
		std::ranges::reverse(found->shape);
	}
	if (auto const omitted = core::alternative<Omitted>(fault)) {
		std::erase_if(stored, [&omitted](Stored const& each) { return each.name == omitted->tensor; });
	}
	if (std::holds_alternative<Extra>(fault)) {
		stored.push_back({.name = "model.language_model.layers.0.extra.weight", .dtype = "BF16", .shape = {2}, .bytes = std::vector<std::byte>(4)});
	}
}

/** The safetensors encoding of `stored`: header length, json header, data in order. */
[[nodiscard]] auto encode(std::span<Stored const> stored) -> std::vector<std::byte> {
	auto header = std::string{R"({"__metadata__":{"format":"pt"})"};
	auto offset = std::size_t{0};
	for (auto const& each : stored) {
		auto shape = std::string{};
		for (auto const extent : each.shape) {
			shape += std::format("{}{}", shape.empty() ? "" : ",", extent);
		}
		header += std::format(R"(,"{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{},{}]}})", each.name, each.dtype, shape, offset,
		                      offset + each.bytes.size());
		offset += each.bytes.size();
	}
	header += '}';
	auto file = std::vector<std::byte>{};
	for (auto byte = std::size_t{0}; byte < 8; ++byte) {
		file.push_back(std::byte{static_cast<std::uint8_t>((header.size() >> (8 * byte)) & 0xffU)});
	}
	std::ranges::transform(header, std::back_inserter(file), [](char character) { return std::bit_cast<std::byte>(character); });
	for (auto const& each : stored) {
		file.insert(file.end(), each.bytes.begin(), each.bytes.end());
	}
	return file;
}

/** A synthetic checkpoint on disk: its files and the digests a publisher would list for them. */
export struct Written {
	std::vector<std::string> names;
	std::vector<sha256::Digest> digests;

	/** The published list, borrowing the names. */
	[[nodiscard]] auto pinned() const -> std::vector<bake::Pinned> {
		auto list = std::vector<bake::Pinned>{};
		for (auto index = std::size_t{0}; index < names.size(); ++index) {
			list.push_back({.name = names[index], .digest = digests[index]});
		}
		return list;
	}
};

auto write_file(std::filesystem::path const& path, std::span<std::byte const> bytes) -> bool {
	auto output = file_write::Output::create(path, bytes.size());
	return output && output->write(0, bytes) && std::move(*output).commit();
}

/**
 * Writes the tiny model's checkpoint with `fault` into `directory`: the model-wide and mtp tensors and a vision tower
 * tensor in one file, the layers in another, as the real checkpoint splits them across files.
 */
export [[nodiscard]] auto write_checkpoint(std::filesystem::path const& directory, std::uint64_t seed, Fault const& fault) -> std::optional<Written> {
	auto stored = tensors(seed, fault);
	apply(stored, fault);
	auto outside = std::vector<Stored>{};
	auto layered = std::vector<Stored>{};
	for (auto& each : stored) {
		(each.name.starts_with(model::trunk_layer_prefix) ? layered : outside).push_back(std::move(each));
	}
	outside.push_back({.name = "model.visual.patch_embed.weight", .dtype = "BF16", .shape = {3, 2}, .bytes = std::vector<std::byte>(12, std::byte{1})});
	auto written = Written{};
	for (auto const& [name, contents] : {std::pair{"outside.safetensors", encode(outside)}, std::pair{"layers.safetensors", encode(layered)}}) {
		if (!write_file(directory / name, contents)) {
			return std::nullopt;
		}
		written.names.emplace_back(name);
		written.digests.push_back(sha256::digest(contents));
	}
	return written;
}

/** A fresh, empty directory of the test's own under the directory ctest runs it in. */
export [[nodiscard]] auto scratch(std::string_view name) -> std::filesystem::path {
	auto const directory = std::filesystem::current_path() / "bake.scratch" / name;
	std::filesystem::remove_all(directory);
	std::filesystem::create_directories(directory / "checkpoint");
	std::filesystem::create_directories(directory / "images");
	return directory;
}

}
