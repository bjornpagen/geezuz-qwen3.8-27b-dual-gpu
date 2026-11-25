// SPDX-License-Identifier: AGPL-3.0-only
export module engine.bake:verify;

import std;
import engine.core;
import engine.file;
import engine.image;
import engine.sha256;
import :error;
import :plan;

namespace engine::bake {

/** A checkpoint file whose bytes hashed to their published SHA-256, kept mapped so no other bytes are ever read. */
export struct VerifiedFile {
	std::string name;
	file::Mapping mapping;
};

/** Every file a plan reads, verified, and the checkpoint identity their digests give. */
export struct Verified {
	std::vector<VerifiedFile> files;
	image::Checkpoint checkpoint;
	std::uint64_t bytes;
};

// Hashing in pieces keeps the pages of one piece hot for the hasher while the kernel reads the next; the size is
// otherwise immaterial to the digest.
inline constexpr std::size_t hash_piece_bytes = std::size_t{64} * 1024 * 1024;

[[nodiscard]] auto digest_of(std::span<std::byte const> bytes) -> sha256::Digest {
	auto hasher = sha256::Hasher{};
	while (!bytes.empty()) {
		auto const piece = bytes.first(std::min(bytes.size(), hash_piece_bytes));
		hasher.update(piece);
		bytes = bytes.subspan(piece.size());
	}
	return std::move(hasher).finish();
}

/**
 * Maps every file of `files` from `directory` and hashes it whole, before any of its bytes are used. A file that cannot
 * be mapped or whose SHA-256 is not its published one is refused by name. The identity is `commit` with the manifest
 * of the files' digests.
 */
export [[nodiscard]] auto verify(std::filesystem::path const& directory, std::span<Pinned const> files, image::Commit const& commit)
    -> core::Result<Verified, BakeError> {
	auto verified = Verified{.files = {}, .checkpoint = {}, .bytes = 0};
	auto digests = std::vector<image::FileDigest>{};
	for (auto const& pinned : files) {
		auto mapping = file::Mapping::open(directory / pinned.name);
		if (!mapping) {
			return refuse(Unreadable{.file = std::string{pinned.name}, .error = mapping.error()});
		}
		auto const actual = digest_of(mapping->bytes());
		if (actual != pinned.digest) {
			return refuse(DigestMismatch{.file = std::string{pinned.name}, .expected = pinned.digest, .actual = actual});
		}
		verified.bytes += mapping->bytes().size();
		digests.push_back({.name = pinned.name, .digest = actual});
		verified.files.push_back({.name = std::string{pinned.name}, .mapping = *std::move(mapping)});
	}
	// A published list names each file once, printably, so its manifest exists.
	auto const manifest = image::manifest_digest(digests);
	core::invariant(manifest.has_value());
	verified.checkpoint = {.commit = commit, .manifest = *manifest};
	return verified;
}

}
