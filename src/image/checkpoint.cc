// SPDX-License-Identifier: AGPL-3.0-only
export module engine.image:checkpoint;

import std;
import engine.core;
import engine.sha256;

namespace engine::image {

export inline constexpr std::size_t commit_bytes = 20;

/**
 * A git commit id. Git names a commit by its SHA-1, so this names the checkpoint's revision but vouches for no bytes;
 * the manifest's SHA-256 does that.
 */
export struct Commit {
	std::array<std::byte, commit_bytes> bytes;

	[[nodiscard]] constexpr auto operator==(Commit const&) const -> bool = default;
};

/** A commit id written in the source as 40 hex digits: malformed text does not compile. */
export [[nodiscard]] consteval auto pinned_commit(std::string_view text) -> Commit {
	auto const read = sha256::from_hex<commit_bytes>(text);
	core::invariant(read.has_value());
	return Commit{.bytes = *read};
}

/** The Hugging Face revision of `Qwen/Qwen3.8-27B-FP8` the engine is built for. */
export inline constexpr Commit checkpoint_commit = pinned_commit("c3a462cc3d38cf351f3b26cbd763469eeb74ef81");

/** Which checkpoint an image was baked from: the revision, and the digest of its files' manifest. */
export struct Checkpoint {
	Commit commit;
	sha256::Digest manifest;

	[[nodiscard]] constexpr auto operator==(Checkpoint const&) const -> bool = default;
};

/** One checkpoint file: its path relative to the checkpoint's root, and the SHA-256 of its bytes. */
export struct FileDigest {
	std::string_view name;
	sha256::Digest digest;
};

/** Why a list of file digests has no canonical manifest. */
export enum class ManifestFailure : std::uint8_t {
	/** No files: no checkpoint is empty. */
	Empty,
	/** An empty name, or one with a control character, which would break the manifest's one line per file. */
	BadName,
	/** Two entries name the same file. */
	DuplicateName,
};

export struct [[nodiscard]] ManifestError {
	ManifestFailure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case ManifestFailure::Empty:
		case ManifestFailure::BadName:
		case ManifestFailure::DuplicateName:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(ManifestError const&) const -> bool = default;
};

[[nodiscard]] constexpr auto printable(std::string_view name) -> bool {
	return !name.empty() && std::ranges::none_of(name, [](char character) {
		auto const code = std::bit_cast<unsigned char>(character);
		return code < 0x20U || code == 0x7fU;
	});
}

/**
 * The canonical manifest of a checkpoint's files: one line per file, `<64 lowercase hex digits><two spaces><name>\n`,
 * sorted by name byte by byte. It is exactly what `sha256sum` prints for the files, sorted in the C locale, so
 * `sha256sum <files> | LC_ALL=C sort -k2 | sha256sum` reproduces its digest with standard tools.
 */
export [[nodiscard]] constexpr auto manifest(std::span<FileDigest const> files) -> core::Result<std::string, ManifestError> {
	if (files.empty()) {
		return std::unexpected(ManifestError{.failure = ManifestFailure::Empty});
	}
	if (!std::ranges::all_of(files, printable, &FileDigest::name)) {
		return std::unexpected(ManifestError{.failure = ManifestFailure::BadName});
	}
	auto sorted = std::vector<FileDigest>{files.begin(), files.end()};
	std::ranges::sort(sorted, std::less{}, &FileDigest::name);
	if (std::ranges::adjacent_find(sorted, std::equal_to{}, &FileDigest::name) != sorted.end()) {
		return std::unexpected(ManifestError{.failure = ManifestFailure::DuplicateName});
	}
	auto text = std::string{};
	for (auto const& file : sorted) {
		auto const digits = sha256::hex(file.digest.bytes);
		text.append(digits.begin(), digits.end());
		text += "  ";
		text += file.name;
		text += '\n';
	}
	return text;
}

/** The SHA-256 of the canonical manifest: the checkpoint's identity as an image records it. */
export [[nodiscard]] constexpr auto manifest_digest(std::span<FileDigest const> files) -> core::Result<sha256::Digest, ManifestError> {
	return manifest(files).transform([](std::string const& text) { return sha256::digest(text); });
}

}
