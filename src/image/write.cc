// SPDX-License-Identifier: AGPL-3.0-only
export module engine.image:write;

import std;
import engine.core;
import engine.sha256;
import :catalog;
import :checkpoint;
import :layout;

namespace engine::image {

/** A complete image's header and digest. */
export struct Finished {
	/** The bytes from offset 0 to the payload: fields, section table and zero padding. */
	std::vector<std::byte> header;
	/** The digest the header records for the payload. */
	sha256::Digest payload;
};

/**
 * Lays out and hashes an image of catalog `C` as the baker produces its extents, without touching a file: it says
 * where each piece of bytes goes, and the caller writes it there. Extents arrive in catalog order, each in pieces of
 * any size; the zero padding between extents is hashed here and must read back as zeros, which a file created empty
 * and written only at the given offsets does. Once every extent is complete, `finish` returns the header for offset 0;
 * the file is then `file_bytes<C>` long.
 *
 * Feeding an extent out of order, or more bytes than it holds, is a baker bug and aborts.
 */
export template<Catalog C>
class Writer {
public:
	explicit Writer(Checkpoint checkpoint) : checkpoint_{checkpoint} {}

	/** Hashes the next `bytes` of extent `index` and returns the file offset they belong at. */
	[[nodiscard]] auto write(std::size_t index, std::span<std::byte const> bytes) -> std::uint64_t {
		core::invariant(current_ < std::size(C::extents) && index == current_);
		auto const placement = layout<C>[current_];
		core::invariant(bytes.size() <= placement.bytes - filled_);
		auto const at = placement.offset + filled_;
		hash_zeros_to(at);
		hasher_.update(bytes);
		hashed_ = at + bytes.size();
		filled_ += bytes.size();
		if (filled_ == placement.bytes) {
			++current_;
			filled_ = 0;
		}
		return payload_start<C> + at;
	}

	/** The header of the complete image, every extent having been written. */
	[[nodiscard]] auto finish() && -> Finished {
		core::invariant(current_ == std::size(C::extents));
		auto const payload = std::move(hasher_).finish();
		return {.header = encode_header<C>(checkpoint_, payload), .payload = payload};
	}

private:
	auto hash_zeros_to(std::uint64_t offset) -> void {
		static constexpr auto zeros = std::array<std::byte, 256>{};
		while (hashed_ < offset) {
			auto const length = std::min<std::uint64_t>(offset - hashed_, zeros.size());
			hasher_.update(std::span{zeros}.first(length));
			hashed_ += length;
		}
	}

	Checkpoint checkpoint_;
	sha256::Hasher hasher_;
	/** The extent being filled, and how many of its bytes have arrived. */
	std::size_t current_ = 0;
	std::uint64_t filled_ = 0;
	/** Payload bytes hashed so far, padding included. */
	std::uint64_t hashed_ = 0;
};

}
