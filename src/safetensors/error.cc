// SPDX-License-Identifier: AGPL-3.0-only
export module engine.safetensors:error;

import std;
import engine.core;

namespace engine::safetensors {

/** Why bytes are not a safetensors file the engine reads. Every kind is a property of the file, never of the moment. */
export enum class Failure : std::uint8_t {
	/** Fewer bytes than the 8-byte header length, or than the header it announces. */
	Truncated,
	/** The announced header is longer than any header the engine accepts. */
	HeaderTooLarge,
	/** The header is not one json document within the engine's json bounds. */
	HeaderNotJson,
	/** The header is not an object of tensor entries each with a dtype, a shape and two data offsets. */
	MalformedEntry,
	/** A dtype outside those the checkpoint holds. */
	UnknownDtype,
	/** A name repeats within one object of the header: two tensors of one name, or a field stated twice. */
	DuplicateName,
	/** A shape whose element count, or byte count, does not fit a size. */
	ShapeOverflow,
	/** An entry's data range does not hold exactly its shape's bytes. */
	SizeMismatch,
	/** The entries' data ranges do not tile the data section: a gap, an overlap, or bytes left over. */
	Untiled,
	/** `__metadata__` is not an object of strings. */
	MalformedMetadata,
};

export struct [[nodiscard]] Error {
	Failure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case Failure::Truncated:
		case Failure::HeaderTooLarge:
		case Failure::HeaderNotJson:
		case Failure::MalformedEntry:
		case Failure::UnknownDtype:
		case Failure::DuplicateName:
		case Failure::ShapeOverflow:
		case Failure::SizeMismatch:
		case Failure::Untiled:
		case Failure::MalformedMetadata:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(Error const&) const -> bool = default;
};

}
