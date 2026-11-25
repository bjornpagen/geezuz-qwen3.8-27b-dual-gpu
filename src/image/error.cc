// SPDX-License-Identifier: AGPL-3.0-only
export module engine.image:error;

import std;
import engine.core;
import engine.file;

namespace engine::image {

/**
 * Why bytes are not the image this engine expects. Every kind is a property of the file against the engine that reads
 * it, and every one is refused whole: nothing of a refused image is handed out.
 */
export enum class Failure : std::uint8_t {
	/** The file ends before its header, its section table or the payload the table lays out: an incomplete copy. */
	Truncated,
	/** The file goes on past the payload the table lays out. */
	Overlong,
	/** The file does not begin with the image magic: it is not a baked image. */
	BadMagic,
	/** A format version this engine does not read: rebake with this engine's baker. */
	UnsupportedVersion,
	/** Baked from another model description than the one this engine was built from. */
	ModelMismatch,
	/** Baked from another checkpoint than the one expected. */
	CheckpointMismatch,
	/** Baked for another memory: the other card, or a card where the host was expected, or the reverse. */
	WrongTarget,
	/** The section table is not this engine's catalog: another schedule's layout, or a corrupted entry. */
	ScheduleMismatch,
	/** The bytes between the section table and the payload are not all zero. */
	DirtyPadding,
	/** The payload is not the bytes the baker hashed: corrupted on disk or in transit. */
	PayloadMismatch,
};

export struct [[nodiscard]] Rejection {
	Failure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case Failure::Truncated:
		case Failure::Overlong:
		case Failure::BadMagic:
		case Failure::UnsupportedVersion:
		case Failure::ModelMismatch:
		case Failure::CheckpointMismatch:
		case Failure::WrongTarget:
		case Failure::ScheduleMismatch:
		case Failure::DirtyPadding:
		case Failure::PayloadMismatch:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(Rejection const&) const -> bool = default;
};

/** Why an image could not be opened: the file could not be mapped, or its bytes were refused. */
export struct [[nodiscard]] ImageError {
	std::variant<file::FileError, Rejection> cause;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return std::visit([](auto const& error) { return error.retry(); }, cause);
	}

	[[nodiscard]] constexpr auto operator==(ImageError const&) const -> bool = default;
};

}
