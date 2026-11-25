// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.image;
import engine.model;
import engine.sha256;
import engine.testing;

namespace {

namespace image = engine::image;
namespace model = engine::model;
namespace sha256 = engine::sha256;

using image::ModelCatalog;
using image::Part;

[[nodiscard]] auto tensor_of(image::Site const& site) -> model::Tensor const& {
	return std::span{model::inventory}[site.tensor];
}

/** The catalog holds exactly the bytes the model description budgets: each card's weights, and the host table whole. */
[[nodiscard]] auto catalog_matches_the_budget() -> bool {
	auto card = std::uint64_t{0};
	auto host = std::uint64_t{0};
	for (auto index = std::size_t{0}; index < ModelCatalog::sites.size(); ++index) {
		auto const bytes = ModelCatalog::extents.at(index).bytes;
		(tensor_of(ModelCatalog::sites.at(index)).split == model::Split::Host ? host : card) += bytes;
	}
	return card == model::weight_bytes_per_card && host == model::embedding_host_bytes;
}

// How many copies of each inventory tensor the description gives the model: one for a tensor outside the decoder
// layers, else one per layer holding its scope.
constexpr auto instances_of = [] {
	auto counts = std::array<model::Count, std::size(model::inventory)>{};
	for (auto tensor = std::size_t{0}; tensor < counts.size(); ++tensor) {
		auto const scope = std::span{model::inventory}[tensor].scope;
		auto const layers = model::layer_instances(scope);
		counts.at(tensor) = scope == model::Scope::Model || scope == model::Scope::Mtp ? 1 : layers.trunk + layers.mtp;
	}
	return counts;
}();

/** Each inventory tensor appears once per instance the description gives it, with scales exactly for e4m3 tensors. */
[[nodiscard]] auto catalog_covers_every_instance() -> bool {
	for (auto tensor = std::size_t{0}; tensor < std::size(model::inventory); ++tensor) {
		auto const& entry = std::span{model::inventory}[tensor];
		auto const instances = instances_of.at(tensor);
		auto const count = [tensor](Part part) {
			return std::ranges::count_if(ModelCatalog::sites, [tensor, part](image::Site const& site) {
				return site.tensor == tensor && site.part == part;
			});
		};
		auto const scales = entry.format == model::Format::E4m3 ? instances : 0;
		if (std::cmp_not_equal(count(Part::Values), instances) || std::cmp_not_equal(count(Part::Scales), scales)) {
			return false;
		}
	}
	return true;
}

/** Every key is found again from its tensor's name, layer and part, so no two sites coincide. */
[[nodiscard]] auto every_key_is_found_by_its_site() -> bool {
	for (auto index = std::size_t{0}; index < ModelCatalog::sites.size(); ++index) {
		auto const& site = ModelCatalog::sites.at(index);
		auto const layer = site.layer.transform([](std::uint8_t value) { return std::size_t{value}; });
		auto const key = image::Key{.index = engine::core::narrow<std::uint16_t>(index)};
		if (image::find(tensor_of(site).name, layer, site.part) != key || image::site(key) != site) {
			return false;
		}
	}
	return true;
}

/** Layer instances follow the mixer pattern: attention tensors only at attention layers, the mtp layer last. */
[[nodiscard]] auto find_refuses_absent_extents() -> bool {
	auto const trunk = model::trunk_layers;
	return !image::find("model.language_model.norm.weight", std::nullopt, Part::Scales) &&
	       !image::find("model.language_model.norm.weight", std::size_t{0}, Part::Values) &&
	       !image::find("mlp.gate_proj.weight", std::nullopt, Part::Values) &&
	       !image::find("mlp.gate_proj.weight", image::decoder_layers, Part::Values) &&
	       !image::find("self_attn.q_proj.weight", std::size_t{0}, Part::Values) &&
	       !image::find("linear_attn.A_log", trunk, Part::Values) && !image::find("no.such.weight", std::nullopt, Part::Values) &&
	       image::find("self_attn.q_proj.weight", std::size_t{3}, Part::Scales) && image::find("self_attn.q_proj.weight", trunk, Part::Values) &&
	       image::find("linear_attn.A_log", std::size_t{0}, Part::Values) && image::find("mlp.down_proj.weight", trunk, Part::Scales);
}

/** The digest the build computed with the portable rounds, in stages, is the run-time digest of the same text. */
[[nodiscard]] auto description_digest_is_its_text() -> bool {
	return sha256::digest(image::description_text) == image::description_digest && ModelCatalog::description == image::description_digest;
}

// Four files whose digests and manifest digest came from `shasum -a 256 <files> | LC_ALL=C sort -k2 | shasum -a 256`.
constexpr auto manifest_files = std::array{
    image::FileDigest{.name = "layers-2.safetensors", .digest = sha256::pinned("3fc4ccfe745870e2c0d99f71f30ff0656c8dedd41cc1d7d3d376b0dbe685e2f3")},
    image::FileDigest{.name = "B.json", .digest = sha256::pinned("df7e70e5021544f4834bbee64a9e3789febc4be81470df629cad6ddb03320a5c")},
    image::FileDigest{.name = "layers-10.safetensors", .digest = sha256::pinned("e4432baa90819aaef51d2a7f8e148bf7e679610f3173752fabb4dcb2d0f418d3")},
    image::FileDigest{.name = "a.safetensors", .digest = sha256::pinned("ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb")},
};
constexpr auto manifest_from_coreutils = sha256::pinned("b15f6af616bec0591237240194ad5d3259d9cace42ff6550ae6c090b39a5df6f");

static_assert(image::manifest_digest(manifest_files) == manifest_from_coreutils);

/** The manifest digest is the coreutils pipeline's, whatever order the files are listed in. */
[[nodiscard]] auto manifest_matches_coreutils_in_any_order(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto files = std::vector<image::FileDigest>{manifest_files.begin(), manifest_files.end()};
	for (auto trial = 0; trial < 50; ++trial) {
		std::ranges::shuffle(files, random);
		if (image::manifest_digest(files) != manifest_from_coreutils) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto manifest_refuses(std::span<image::FileDigest const> files, image::ManifestFailure failure) -> bool {
	auto const digest = image::manifest_digest(files);
	return !digest && digest.error().failure == failure;
}

[[nodiscard]] auto manifest_refuses_bad_lists() -> bool {
	auto const some = manifest_files[0].digest;
	auto ok = manifest_refuses({}, image::ManifestFailure::Empty) &&
	          manifest_refuses(std::array{image::FileDigest{.name = "", .digest = some}}, image::ManifestFailure::BadName) &&
	          manifest_refuses(std::array{manifest_files[0], manifest_files[1], manifest_files[0]}, image::ManifestFailure::DuplicateName);
	for (auto code = 0; code < 256; ++code) {
		auto const character = std::bit_cast<char>(static_cast<std::uint8_t>(code));
		auto const name = std::string{"config"} + character + ".json";
		auto const files = std::array{image::FileDigest{.name = name, .digest = some}};
		auto const control = code < 0x20 || code == 0x7f;
		ok = ok && (control ? manifest_refuses(files, image::ManifestFailure::BadName) : image::manifest_digest(files).has_value());
	}
	return ok;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "the model catalog holds each card's weight bytes and the host table", .holds = catalog_matches_the_budget()},
	    engine::testing::Case{.name = "the model catalog covers every tensor instance, scales exactly for e4m3",
	                          .holds = catalog_covers_every_instance()},
	    engine::testing::Case{.name = "every model key is found by its site", .holds = every_key_is_found_by_its_site()},
	    engine::testing::Case{.name = "extents the model lacks are not found", .holds = find_refuses_absent_extents()},
	    engine::testing::Case{.name = "the build's description digest is the run-time digest of its text", .holds = description_digest_is_its_text()},
	    engine::testing::Case{.name = "the manifest digest is coreutils' in any listing order (seed 1)",
	                          .holds = manifest_matches_coreutils_in_any_order(1)},
	    engine::testing::Case{.name = "empty lists, control characters and duplicates have no manifest", .holds = manifest_refuses_bad_lists()},
	};
	return engine::testing::report(cases);
}
