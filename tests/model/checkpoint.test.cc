// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.file;
import engine.json;
import engine.model;
import engine.safetensors;
import engine.testing;

namespace {

namespace model = engine::model;
namespace safetensors = engine::safetensors;
namespace json = engine::json;

using model::Count;

/** A tensor as the checkpoint's headers state it, and the file holding it. */
struct Stored {
	safetensors::Dtype dtype;
	std::vector<std::size_t> shape;
	std::string file;
	Count copies;
};

/** A tensor the model description says the checkpoint holds, and the inventory entry it comes from. */
struct Wanted {
	std::string name;
	safetensors::Dtype dtype;
	std::vector<std::size_t> shape;
	std::size_t entry;
};

using Catalog = std::map<std::string, Stored, std::less<>>;

[[nodiscard]] auto extents(model::Shape const& shape) -> std::vector<std::size_t> {
	auto const all = std::span{shape.extents};
	return {all.begin(), all.begin() + static_cast<std::ptrdiff_t>(shape.rank)};
}

[[nodiscard]] auto stored_as(model::Format format) -> safetensors::Dtype {
	switch (format) {
	case model::Format::E4m3:
		return safetensors::Dtype::F8E4M3;
	case model::Format::Bf16:
		return safetensors::Dtype::Bf16;
	}
	std::unreachable();
}

/** The full names one inventory entry stands for, one per layer or head instance. */
[[nodiscard]] auto names_of(model::Tensor const& tensor) -> std::vector<std::string> {
	switch (tensor.scope) {
	case model::Scope::Model:
		return {std::string{tensor.name}};
	case model::Scope::Mtp:
		return {std::format("{}{}", model::mtp_prefix, tensor.name)};
	case model::Scope::DecoderLayer:
	case model::Scope::LinearAttentionLayer:
	case model::Scope::FullAttentionLayer:
		break;
	}
	auto names = std::vector<std::string>{};
	for (auto const [layer, mixer] : std::views::enumerate(model::trunk_mixers)) {
		if (model::in_layer(tensor.scope, mixer)) {
			names.push_back(std::format("{}{}.{}", model::trunk_layer_prefix, layer, tensor.name));
		}
	}
	for (auto layer = Count{0}; layer < model::mtp_layers; ++layer) {
		if (model::in_layer(tensor.scope, model::mtp_mixer)) {
			names.push_back(std::format("{}{}.{}", model::mtp_layer_prefix, layer, tensor.name));
		}
	}
	return names;
}

/** Every tensor the model description expects: weights, and the block scales of the e4m3 ones. */
[[nodiscard]] auto wanted() -> std::vector<Wanted> {
	auto all = std::vector<Wanted>{};
	for (auto const [entry, tensor] : std::views::enumerate(model::inventory)) {
		auto const index = static_cast<std::size_t>(entry);
		for (auto const& name : names_of(tensor)) {
			all.push_back({.name = name, .dtype = stored_as(tensor.format), .shape = extents(tensor.shape), .entry = index});
			if (tensor.format == model::Format::E4m3) {
				all.push_back({
				    .name = name + model::scale_suffix,
				    .dtype = safetensors::Dtype::Bf16,
				    .shape = extents(model::scale_shape(tensor.shape)),
				    .entry = index,
				});
			}
		}
	}
	return all;
}

[[nodiscard]] auto excluded(std::string_view name) -> bool {
	return name.starts_with(model::excluded_prefix);
}

struct Checkpoint {
	Catalog tensors;
	Count files;
};

/** Every tensor of every safetensors file in the checkpoint directory, or nothing if a file fails to decode. */
[[nodiscard]] auto read_checkpoint(std::filesystem::path const& directory) -> std::optional<Checkpoint> {
	auto paths = std::vector<std::filesystem::path>{};
	for (auto const& item : std::filesystem::directory_iterator{directory}) {
		if (item.path().extension() == ".safetensors") {
			paths.push_back(item.path());
		}
	}
	auto checkpoint = Checkpoint{.tensors = {}, .files = paths.size()};
	for (auto const& path : paths) {
		auto const mapping = engine::file::Mapping::open(path);
		if (!mapping) {
			std::println("cannot map {}", path.string());
			return std::nullopt;
		}
		auto const file = safetensors::decode(mapping->bytes());
		if (!file) {
			std::println("cannot decode {}", path.string());
			return std::nullopt;
		}
		for (auto const& tensor : file->tensors) {
			auto const [place, fresh] =
			    checkpoint.tensors.try_emplace(tensor.name, tensor.dtype, tensor.shape, path.filename().string(), 0);
			place->second.copies += 1;
		}
	}
	return checkpoint;
}

[[nodiscard]] auto every_stored_tensor_is_wanted_or_vision(Catalog const& stored, std::vector<Wanted> const& want) -> bool {
	auto names = std::set<std::string_view>{};
	for (auto const& each : want) {
		names.insert(each.name);
	}
	return std::ranges::all_of(stored, [&names](auto const& item) {
		return excluded(item.first) || names.contains(item.first);
	});
}

[[nodiscard]] auto every_wanted_tensor_is_stored_once_as_described(Catalog const& stored, std::vector<Wanted> const& want) -> bool {
	return std::ranges::all_of(want, [&stored](Wanted const& each) {
		auto const found = stored.find(each.name);
		return found != stored.end() && found->second.copies == 1 && found->second.dtype == each.dtype && found->second.shape == each.shape;
	});
}

/**
 * Splitting each checkpoint tensor by its card shards gives back its stored extent: the parts of a sharded axis
 * add up to it, a replicated tensor is whole on each card, and an e4m3 part cuts the stored scale tensor on block
 * boundaries.
 */
[[nodiscard]] auto shards_reassemble_the_stored_shapes(Catalog const& stored, std::vector<Wanted> const& want) -> bool {
	return std::ranges::all_of(want, [&stored](Wanted const& each) {
		auto const& tensor = model::inventory[each.entry];
		auto const found = stored.find(each.name);
		if (found == stored.end() || each.dtype != stored_as(tensor.format)) {
			return true;
		}
		auto const axis = model::split_axis(tensor.split);
		auto const whole = found->second.shape.at(axis);
		auto total = Count{0};
		for (auto card = Count{0}; card < model::cards; ++card) {
			auto const part = model::card_shape(tensor, card).extents[axis];
			auto const blocked = tensor.format != model::Format::E4m3 || part % model::scale_block == 0;
			if (!blocked) {
				return false;
			}
			total += part;
		}
		switch (tensor.split) {
		case model::Split::Host:
			return total == 0;
		case model::Split::Replicated:
			return total == model::cards * whole;
		case model::Split::Rows:
		case model::Split::Columns:
		case model::Split::DeltanetChannels:
		case model::Split::HiddenPair:
		case model::Split::Vocabulary:
			return total == whole;
		}
		std::unreachable();
	});
}

/** The bytes of a small text file of the checkpoint, or nothing if it cannot be mapped. */
[[nodiscard]] auto text_of(std::filesystem::path const& path) -> std::optional<std::string> {
	auto const mapping = engine::file::Mapping::open(path);
	if (!mapping) {
		std::println("cannot map {}", path.string());
		return std::nullopt;
	}
	auto text = std::string(mapping->bytes().size(), '\0');
	std::ranges::transform(mapping->bytes(), text.begin(), [](std::byte byte) {
		return static_cast<char>(byte);
	});
	return text;
}

/** config.json: nested four deep (text_config, rope_parameters, mrope_section), its longest array 882 module names. */
inline constexpr json::Bounds config_bounds{
    .document_bytes = std::size_t{256} * 1024,
    .depth = 8,
    .string_bytes = 256,
    .members = 4096,
    .number_chars = 32,
};

using ConfigObject = json::Object<config_bounds>;

[[nodiscard]] auto integer_is(ConfigObject& object, std::string_view name, Count expected) -> bool {
	auto value = object.member(name);
	if (!value) {
		return false;
	}
	auto const integer = value->as_integer();
	return integer && std::cmp_equal(*integer, expected);
}

[[nodiscard]] auto real_of(ConfigObject& object, std::string_view name) -> std::optional<double> {
	auto value = object.member(name);
	if (!value) {
		return std::nullopt;
	}
	auto const real = value->as_real();
	return real ? std::optional{*real} : std::nullopt;
}

[[nodiscard]] auto boolean_is(ConfigObject& object, std::string_view name, bool expected) -> bool {
	auto value = object.member(name);
	if (!value) {
		return false;
	}
	auto const boolean = value->as_boolean();
	return boolean && *boolean == expected;
}

[[nodiscard]] auto string_is(ConfigObject& object, std::string_view name, std::string_view expected) -> bool {
	auto value = object.member(name);
	if (!value) {
		return false;
	}
	auto const text = value->as_string();
	return text && *text == expected;
}

/** The integers of an array member, or nothing if it is not one. */
[[nodiscard]] auto integers_of(ConfigObject& object, std::string_view name) -> std::optional<std::vector<std::int64_t>> {
	auto value = object.member(name);
	if (!value) {
		return std::nullopt;
	}
	auto array = value->as_array();
	if (!array) {
		return std::nullopt;
	}
	auto result = std::vector<std::int64_t>{};
	while (true) {
		auto element = array->next();
		if (!element) {
			return std::nullopt;
		}
		if (!*element) {
			return result;
		}
		auto const integer = (*element)->as_integer();
		if (!integer) {
			return std::nullopt;
		}
		result.push_back(*integer);
	}
}

/** The strings of an array member, or nothing if it is not one. */
[[nodiscard]] auto strings_of(ConfigObject& object, std::string_view name) -> std::optional<std::vector<std::string>> {
	auto value = object.member(name);
	if (!value) {
		return std::nullopt;
	}
	auto array = value->as_array();
	if (!array) {
		return std::nullopt;
	}
	auto result = std::vector<std::string>{};
	while (true) {
		auto element = array->next();
		if (!element) {
			return std::nullopt;
		}
		if (!*element) {
			return result;
		}
		auto const text = (*element)->as_string();
		if (!text) {
			return std::nullopt;
		}
		result.emplace_back(*text);
	}
}

template<class T, std::size_t n>
[[nodiscard]] auto counts_are(std::optional<std::vector<std::int64_t>> const& read, std::array<T, n> const& expected) -> bool {
	return read && std::ranges::equal(*read, expected, [](std::int64_t left, T right) {
		       return std::cmp_equal(left, right);
	       });
}

/** What config.json's text_config says, each judged against the model description. */
struct TextConfig {
	bool dimensions;
	bool layer_types;
	bool rope;
	bool norms_and_gates;
	bool mtp_and_embeddings;
};

struct Config {
	TextConfig text;
	bool quantization;
	std::vector<std::string> unconverted;
};

[[nodiscard]] auto read_text_config(ConfigObject& text) -> TextConfig {
	auto config = TextConfig{};
	config.dimensions =
	    integer_is(text, "hidden_size", model::hidden) && integer_is(text, "num_hidden_layers", model::trunk_layers) &&
	    integer_is(text, "vocab_size", model::vocab) && integer_is(text, "intermediate_size", model::ffn) &&
	    integer_is(text, "max_position_embeddings", model::max_positions) && integer_is(text, "num_attention_heads", model::query_heads) &&
	    integer_is(text, "num_key_value_heads", model::kv_heads) && integer_is(text, "head_dim", model::head_dim) &&
	    integer_is(text, "linear_num_key_heads", model::key_heads) && integer_is(text, "linear_num_value_heads", model::value_heads) &&
	    integer_is(text, "linear_key_head_dim", model::key_dim) && integer_is(text, "linear_value_head_dim", model::value_dim) &&
	    integer_is(text, "linear_conv_kernel_dim", model::conv_kernel);
	auto const kinds = strings_of(text, "layer_types");
	config.layer_types = kinds && std::ranges::equal(*kinds, model::trunk_mixers, [](std::string const& kind, model::Mixer mixer) {
		                     return kind == (mixer == model::Mixer::LinearAttention ? "linear_attention" : "full_attention");
	                     });
	auto const eps = real_of(text, "rms_norm_eps");
	config.norms_and_gates = eps && static_cast<float>(*eps) == model::rms_norm_eps && boolean_is(text, "attn_output_gate", true) &&
	                         string_is(text, "hidden_act", "silu") && model::mlp_gate == model::Gate::Silu &&
	                         string_is(text, "output_gate_type", "swish") && model::deltanet_output_gate == model::Gate::Silu;
	config.mtp_and_embeddings = integer_is(text, "mtp_num_hidden_layers", model::mtp_layers) &&
	                            boolean_is(text, "mtp_use_dedicated_embeddings", false) && boolean_is(text, "tie_word_embeddings", false);
	auto rope_value = text.member("rope_parameters");
	auto rope = rope_value ? rope_value->as_object() : std::unexpected(rope_value.error());
	if (!rope) {
		config.rope = false;
		return config;
	}
	auto const theta = real_of(*rope, "rope_theta");
	auto const factor = real_of(*rope, "partial_rotary_factor");
	auto const sections = std::to_array(model::mrope_sections);
	config.rope =
	    theta && static_cast<float>(*theta) == model::rope_theta && factor &&
	    *factor * static_cast<double>(model::partial_rotary.denominator) == static_cast<double>(model::partial_rotary.numerator) &&
	    counts_are(integers_of(*rope, "mrope_section"), sections) && boolean_is(*rope, "mrope_interleaved", true) &&
	    string_is(*rope, "rope_type", "default");
	return config;
}

[[nodiscard]] auto read_config(std::filesystem::path const& directory) -> std::optional<Config> {
	auto const text = text_of(directory / "config.json");
	auto reader = json::Reader<config_bounds>::make();
	if (!text || !reader || !json::check_text(*reader, *text)) {
		return std::nullopt;
	}
	auto root_value = reader->read_copy(*text);
	auto root = root_value ? root_value->as_object() : std::unexpected(root_value.error());
	if (!root) {
		return std::nullopt;
	}
	auto text_value = root->member("text_config");
	auto text_config = text_value ? text_value->as_object() : std::unexpected(text_value.error());
	if (!text_config) {
		return std::nullopt;
	}
	auto const text_checks = read_text_config(*text_config);
	auto quantization_value = root->member("quantization_config");
	auto quantization = quantization_value ? quantization_value->as_object() : std::unexpected(quantization_value.error());
	if (!quantization) {
		return std::nullopt;
	}
	auto const block = std::array{model::scale_block, model::scale_block};
	auto const quantized = string_is(*quantization, "quant_method", "fp8") && string_is(*quantization, "fmt", "e4m3") &&
	                       counts_are(integers_of(*quantization, "weight_block_size"), block);
	auto unconverted = strings_of(*quantization, "modules_to_not_convert");
	if (!unconverted) {
		return std::nullopt;
	}
	return Config{.text = text_checks, .quantization = quantized, .unconverted = std::move(*unconverted)};
}

// The list of unconverted modules is not reliable in what it includes (it names modules the checkpoint lacks), but
// no module it names may be stored as e4m3.
[[nodiscard]] auto no_e4m3_module_is_listed_unconverted(std::vector<std::string> const& unconverted, std::vector<Wanted> const& want)
    -> bool {
	auto const listed = std::set<std::string_view>{unconverted.begin(), unconverted.end()};
	return std::ranges::none_of(want, [&listed](Wanted const& each) {
		auto const module = std::string_view{each.name}.substr(0, each.name.rfind(".weight"));
		return each.dtype == safetensors::Dtype::F8E4M3 && each.name.ends_with(".weight") && listed.contains(module);
	});
}

/** model.safetensors.index.json: one map of about 1,600 tensor names to file names. */
inline constexpr json::Bounds index_bounds{
    .document_bytes = std::size_t{1024} * 1024,
    .depth = 2,
    .string_bytes = 256,
    .members = 4096,
    .number_chars = 32,
};

/** The index's map of tensor name to file, or nothing if it is not one. */
[[nodiscard]] auto read_index(std::filesystem::path const& directory) -> std::optional<std::map<std::string, std::string, std::less<>>> {
	auto const text = text_of(directory / "model.safetensors.index.json");
	auto reader = json::Reader<index_bounds>::make();
	if (!text || !reader || !json::check_text(*reader, *text)) {
		return std::nullopt;
	}
	auto root_value = reader->read_copy(*text);
	auto root = root_value ? root_value->as_object() : std::unexpected(root_value.error());
	auto map_value = root ? root->member("weight_map") : std::unexpected(root.error());
	auto map = map_value ? map_value->as_object() : std::unexpected(map_value.error());
	if (!map) {
		return std::nullopt;
	}
	auto index = std::map<std::string, std::string, std::less<>>{};
	while (true) {
		auto member = map->next();
		if (!member) {
			return std::nullopt;
		}
		if (!*member) {
			return index;
		}
		auto const file = (*member)->value.as_string();
		if (!file) {
			return std::nullopt;
		}
		index.emplace((*member)->name, *file);
	}
}

/** The index names exactly the stored tensors, each with the file that holds it. */
[[nodiscard]] auto index_matches_the_headers(std::map<std::string, std::string, std::less<>> const& index, Catalog const& stored) -> bool {
	return index.size() == stored.size() && std::ranges::all_of(stored, [&index](auto const& item) {
		       auto const found = index.find(item.first);
		       return found != index.end() && found->second == item.second.file;
	       });
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	if (arguments.size() != 2) {
		std::println("usage: {} <checkpoint directory>", arguments.empty() ? "checkpoint" : arguments[0]);
		return 1;
	}
	auto const directory = std::filesystem::path{arguments[1]};
	auto const checkpoint = read_checkpoint(directory);
	if (!checkpoint) {
		std::println("FAIL: every safetensors file of the checkpoint decodes");
		return 1;
	}
	auto const want = wanted();
	auto const config = read_config(directory);
	auto const index = read_index(directory);
	if (!config || !index) {
		std::println("FAIL: config.json and the index read as json of the expected shape");
		return 1;
	}
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "the checkpoint holds tensors in more than one safetensors file",
	        .holds = checkpoint->files > 1 && !checkpoint->tensors.empty(),
	    },
	    engine::testing::Case{
	        .name = "every checkpoint tensor is in the inventory, or is the vision tower's",
	        .holds = every_stored_tensor_is_wanted_or_vision(checkpoint->tensors, want),
	    },
	    engine::testing::Case{
	        .name = "every inventory tensor and block scale is stored exactly once, with its dtype and shape",
	        .holds = every_wanted_tensor_is_stored_once_as_described(checkpoint->tensors, want),
	    },
	    engine::testing::Case{
	        .name = "each tensor's card shards reassemble its stored shape, e4m3 parts on scale-block boundaries",
	        .holds = shards_reassemble_the_stored_shapes(checkpoint->tensors, want),
	    },
	    engine::testing::Case{
	        .name = "the index maps exactly the stored tensors, each to the file holding it",
	        .holds = index_matches_the_headers(*index, checkpoint->tensors),
	    },
	    engine::testing::Case{
	        .name = "config.json's text dimensions are the model description's",
	        .holds = config->text.dimensions,
	    },
	    engine::testing::Case{
	        .name = "config.json's layer_types are the trunk's mixers in order",
	        .holds = config->text.layer_types,
	    },
	    engine::testing::Case{
	        .name = "config.json's rope is theta 1e7, a quarter of each head, interleaved mrope sections 11, 11, 10",
	        .holds = config->text.rope,
	    },
	    engine::testing::Case{
	        .name = "config.json's norm epsilon, attention output gate and gate activations are the model description's",
	        .holds = config->text.norms_and_gates,
	    },
	    engine::testing::Case{
	        .name = "config.json has one mtp layer on the trunk's embeddings, and untied embeddings",
	        .holds = config->text.mtp_and_embeddings,
	    },
	    engine::testing::Case{
	        .name = "config.json quantizes to fp8 e4m3 in blocks of the scale block on each side",
	        .holds = config->quantization,
	    },
	    engine::testing::Case{
	        .name = "no module modules_to_not_convert names is stored as e4m3",
	        .holds = no_e4m3_module_is_listed_unconverted(config->unconverted, want),
	    },
	};
	return engine::testing::report(cases);
}
