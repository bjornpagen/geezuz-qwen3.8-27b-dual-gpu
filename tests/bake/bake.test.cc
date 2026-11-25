// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.bake;
import engine.bake_inverse;
import engine.bake_synthetic;
import engine.core;
import engine.file;
import engine.file_write;
import engine.image;
import engine.model;
import engine.testing;

namespace {

namespace bake = engine::bake;
namespace synthetic = engine::bake_synthetic;
namespace image = engine::image;

using synthetic::Tiny;

/** A synthetic checkpoint with `fault`, verified against its own published list. */
struct Prepared {
	std::filesystem::path directory;
	synthetic::Written written;
};

[[nodiscard]] auto prepare(std::string_view name, std::uint64_t seed, synthetic::Fault const& fault) -> std::optional<Prepared> {
	auto const directory = synthetic::scratch(name);
	auto written = synthetic::write_checkpoint(directory / "checkpoint", seed, fault);
	if (!written) {
		return std::nullopt;
	}
	return Prepared{.directory = directory, .written = std::move(*written)};
}

[[nodiscard]] auto bake_prepared(Prepared const& prepared) -> engine::core::Result<bake::Report, bake::BakeError> {
	auto verified = bake::verify(prepared.directory / "checkpoint", prepared.written.pinned(), image::checkpoint_commit);
	if (!verified) {
		return std::unexpected(verified.error());
	}
	return bake::bake_verified<Tiny>(*std::move(verified), prepared.directory / "images");
}

[[nodiscard]] auto source_of(Prepared const& prepared) -> std::optional<bake::Source> {
	auto verified = bake::verify(prepared.directory / "checkpoint", prepared.written.pinned(), image::checkpoint_commit);
	if (!verified) {
		return std::nullopt;
	}
	auto source = bake::Source::open<Tiny>(*std::move(verified));
	return source ? std::optional{*std::move(source)} : std::nullopt;
}

[[nodiscard]] auto entries(std::filesystem::path const& directory) -> std::size_t {
	return static_cast<std::size_t>(std::distance(std::filesystem::directory_iterator{directory}, std::filesystem::directory_iterator{}));
}

/**
 * The bake's images open as the catalogs say, and inverting them (split, tile encoding, vocabulary permutation) gives
 * back every checkpoint tensor bit for bit, each element from exactly the cards its split gives it to.
 */
[[nodiscard]] auto images_invert_to_the_checkpoint(std::uint64_t seed) -> bool {
	auto const prepared = prepare(std::format("round-trip-{}", seed), seed, synthetic::Clean{});
	if (!prepared) {
		return false;
	}
	auto const report = bake_prepared(*prepared);
	auto const source = source_of(*prepared);
	if (!report || !source || entries(prepared->directory / "images") != 3) {
		return false;
	}
	auto const images = prepared->directory / "images";
	auto const zero = image::Image<Tiny::CardZero>::open(images / bake::image_name(image::Target::CardZero), report->checkpoint);
	auto const one = image::Image<Tiny::CardOne>::open(images / bake::image_name(image::Target::CardOne), report->checkpoint);
	auto const host = image::Image<Tiny::Host>::open(images / bake::image_name(image::Target::Host), report->checkpoint);
	if (!zero || !one || !host) {
		return false;
	}
	auto const verdict = engine::bake_inverse::invert<Tiny>(
	    *source, {.cards = {engine::bake_inverse::extents_of(*zero), engine::bake_inverse::extents_of(*one)}, .host = engine::bake_inverse::extents_of(*host)});
	for (auto const& name : verdict.mismatched) {
		std::println("mismatched: {}", name);
	}
	return verdict.mismatched.empty() && verdict.matched == verdict.copies && verdict.copies > 0 &&
	       report->checkpoint == source->verified().checkpoint;
}

/** Baking the same checkpoint twice writes the same images: the bake depends on nothing but its input. */
[[nodiscard]] auto bake_is_deterministic() -> bool {
	auto const prepared = prepare("deterministic", 7, synthetic::Clean{});
	if (!prepared) {
		return false;
	}
	auto const first = bake_prepared(*prepared);
	auto const second = bake_prepared(*prepared);
	if (!first || !second) {
		return false;
	}
	return std::ranges::equal(first->images, second->images, [](bake::ImageReport const& left, bake::ImageReport const& right) {
		return left.payload == right.payload && left.bytes == right.bytes;
	});
}

/** The bake's refusal for a checkpoint with `fault`, and whether it left the output directory empty. */
[[nodiscard]] auto refusal(std::string_view name, synthetic::Fault const& fault) -> std::optional<bake::BakeError> {
	auto const prepared = prepare(name, 11, fault);
	if (!prepared) {
		return std::nullopt;
	}
	auto const report = bake_prepared(*prepared);
	if (report || entries(prepared->directory / "images") != 0) {
		return std::nullopt;
	}
	return report.error();
}

[[nodiscard]] auto refuses_with(std::string_view name, synthetic::Fault const& fault, bake::BakeError const& expected) -> bool {
	auto const refused = refusal(name, fault);
	if (refused && !(*refused == expected)) {
		std::println("refused instead with: {}", refused->describe());
	}
	return refused == expected;
}

[[nodiscard]] auto nan_codes_are_refused_where_they_are() -> bool {
	auto holds = true;
	auto random = engine::testing::generator(19);
	// Codes in every tiled e4m3 tensor, at random positions, each refused naming the tensor and the element.
	for (std::string_view const tensor : {"model.language_model.layers.1.mlp.gate_proj.weight", "model.language_model.layers.0.mlp.up_proj.weight",
	                                 "model.language_model.layers.1.self_attn.q_proj.weight", "mtp.layers.0.mlp.down_proj.weight",
	                                 "model.language_model.layers.0.self_attn.v_proj.weight"}) {
		auto const name = std::string{tensor};
		auto const& entry = *std::ranges::find_if(synthetic::inventory, [&name](engine::model::Tensor const& each) { return name.ends_with(each.name); });
		auto const row = std::uniform_int_distribution<std::size_t>{0, entry.shape.extents[0] - 1}(random);
		auto const column = std::uniform_int_distribution<std::size_t>{0, entry.shape.extents[1] - 1}(random);
		holds = holds && refuses_with(std::format("nan-{}", name), synthetic::NanAt{.tensor = name, .row = row, .column = column},
		                              bake::BakeError{.cause = bake::NanCode{.tensor = name, .row = row, .column = column}});
	}
	return holds;
}

[[nodiscard]] auto a_changed_file_is_refused_by_name() -> bool {
	auto const prepared = prepare("hash", 23, synthetic::Clean{});
	if (!prepared) {
		return false;
	}
	auto const path = prepared->directory / "checkpoint" / "layers.safetensors";
	auto bytes = std::vector<std::byte>{};
	{
		auto const mapping = engine::file::Mapping::open(path);
		if (!mapping) {
			return false;
		}
		bytes.assign(mapping->bytes().begin(), mapping->bytes().end());
	}
	bytes.at(bytes.size() / 2) ^= std::byte{0x01};
	auto output = engine::file_write::Output::create(path, bytes.size());
	if (!output || !output->write(0, bytes) || !std::move(*output).commit()) {
		return false;
	}
	auto const report = bake_prepared(*prepared);
	if (report) {
		return false;
	}
	auto const mismatch = engine::core::alternative<bake::DigestMismatch>(report.error().cause);
	return mismatch && mismatch->file == "layers.safetensors" && mismatch->expected == prepared->written.digests.at(1) &&
	       mismatch->actual != mismatch->expected && entries(prepared->directory / "images") == 0;
}

[[nodiscard]] auto a_missing_file_is_refused_by_name() -> bool {
	auto const prepared = prepare("missing-file", 29, synthetic::Clean{});
	if (!prepared) {
		return false;
	}
	std::filesystem::remove(prepared->directory / "checkpoint" / "outside.safetensors");
	auto const report = bake_prepared(*prepared);
	return !report && report.error() == bake::BakeError{.cause = bake::Unreadable{.file = "outside.safetensors",
	                                                                               .error = {.failure = engine::file::Failure::NotFound}}};
}

[[nodiscard]] auto disagreements_with_the_model_are_refused() -> bool {
	auto const reshaped = std::string{"model.language_model.layers.1.linear_attn.in_proj_a.weight"};
	auto const omitted = std::string{"mtp.layers.0.self_attn.v_proj.weight_scale_inv"};
	return refuses_with("reshaped", synthetic::Reshaped{.tensor = reshaped}, bake::BakeError{.cause = bake::WrongTensor{.tensor = reshaped}}) &&
	       refuses_with("omitted", synthetic::Omitted{.tensor = omitted}, bake::BakeError{.cause = bake::MissingTensor{.tensor = omitted}}) &&
	       refuses_with("extra", synthetic::Extra{},
	                    bake::BakeError{.cause = bake::UnexpectedTensor{.tensor = "model.language_model.layers.0.extra.weight"}});
}

/** The bound of a checkpoint whose attention weights make it exact arithmetic: codes, scales and norms all one value. */
[[nodiscard]] auto bound_of_uniform(std::string_view name, std::uint8_t code, std::uint16_t scale) -> std::optional<engine::core::Result<bake::Bound, bake::BakeError>> {
	auto const prepared = prepare(name, 31, synthetic::Uniform{.code = code, .scale = scale});
	if (!prepared) {
		return std::nullopt;
	}
	auto const source = source_of(*prepared);
	if (!source) {
		return std::nullopt;
	}
	return bake::attention_bound<Tiny>(*source);
}

// With every code 1.0 (0x38), every scale 1.0 and every norm weight zero, each row sums 256 ones and sqrt(256) = 16:
// the bound is 4096, which upward rounding may exceed by a few ulps but never undershoots.
[[nodiscard]] auto bound_is_exact_on_uniform_weights() -> bool {
	auto const bound = bound_of_uniform("bound-exact", 0x38, 0x3f80);
	return bound && *bound && (*bound)->value >= 4096.0 && (*bound)->value <= 4096.0 * (1.0 + 0x1p-40);
}

// Codes of 448 (0x7e) at scale 2^62 (0x5e80) give 16 * 256 * 448 * 2^62, past 2^70.
[[nodiscard]] auto bound_past_the_limit_is_refused() -> bool {
	auto const bound = bound_of_uniform("bound-overflow", 0x7e, 0x5e80);
	if (!bound || *bound) {
		return false;
	}
	auto const overflow = engine::core::alternative<bake::AttentionOverflow>((*bound).error().cause);
	return overflow && overflow->bound >= 16.0 * 256.0 * 448.0 * 0x1p62;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the images invert to the checkpoint bit for bit (seed 1)", .holds = images_invert_to_the_checkpoint(1)},
	    engine::testing::Case{.name = "the images invert to the checkpoint bit for bit (seed 2)", .holds = images_invert_to_the_checkpoint(2)},
	    engine::testing::Case{.name = "baking one checkpoint twice writes the same images", .holds = bake_is_deterministic()},
	    engine::testing::Case{.name = "an e4m3 nan code is refused naming its tensor, row and column, leaving no image",
	                          .holds = nan_codes_are_refused_where_they_are()},
	    engine::testing::Case{.name = "a file whose SHA-256 is not the published one is refused by name, leaving no image",
	                          .holds = a_changed_file_is_refused_by_name()},
	    engine::testing::Case{.name = "a missing checkpoint file is refused by name", .holds = a_missing_file_is_refused_by_name()},
	    engine::testing::Case{.name = "a wrong shape, a missing tensor and a foreign tensor are each refused by name",
	                          .holds = disagreements_with_the_model_are_refused()},
	    engine::testing::Case{.name = "the attention bound is the exact value, rounded up, when the arithmetic is exact",
	                          .holds = bound_is_exact_on_uniform_weights()},
	    engine::testing::Case{.name = "a bound at or past 2^70 is refused", .holds = bound_past_the_limit_is_refused()},
	};
	return engine::testing::report(cases);
}
