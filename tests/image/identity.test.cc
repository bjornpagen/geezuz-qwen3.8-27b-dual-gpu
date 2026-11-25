// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.image;
import engine.model;
import engine.sha256;
import engine.testing;

namespace {

namespace image = engine::image;
namespace sha256 = engine::sha256;

/** The digest the build computed with the portable rounds, in stages, is the run-time digest of the same text. */
[[nodiscard]] auto description_digest_is_its_text() -> bool {
	return sha256::digest(image::description_text) == image::description_digest && image::CardCatalog<engine::model::Card::One>::description == image::description_digest &&
	       image::HostCatalog::description == image::description_digest;
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
	    engine::testing::Case{.name = "every catalog carries the description digest, the run-time digest of its text", .holds = description_digest_is_its_text()},
	    engine::testing::Case{.name = "the manifest digest is coreutils' in any listing order (seed 1)",
	                          .holds = manifest_matches_coreutils_in_any_order(1)},
	    engine::testing::Case{.name = "empty lists, control characters and duplicates have no manifest", .holds = manifest_refuses_bad_lists()},
	};
	return engine::testing::report(cases);
}
