// SPDX-License-Identifier: AGPL-3.0-only
export module engine.image:write;

import std;
import engine.core;
import engine.sha256;
import :catalog;
import :checkpoint;
import :layout;

namespace engine::image {

/** What the baker stamps on an image besides its order: the card it is for and the checkpoint it comes from. */
export struct Stamp {
	Card card;
	Checkpoint checkpoint;
};

/** A complete image's header and length. */
export struct Finished {
	/** The bytes from offset 0 to the payload: fields, section table and zero padding. */
	std::vector<std::byte> header;
	/** The file's length: the payload's end. */
	std::uint64_t file_bytes;
	/** The digest the header records for the payload. */
	sha256::Digest payload;
};

/**
 * Lays out and hashes an image as the baker produces its extents, without touching a file: it says where each piece
 * of bytes goes, and the caller writes it there. Extents arrive in the order's sequence, each in pieces of any size;
 * the zero padding between extents is hashed here and must read back as zeros, which a file created empty and written
 * only at the given offsets does. When every extent is complete, `finish` returns the header for offset 0.
 *
 * Feeding the wrong extent, or more bytes than an extent holds, is a baker bug and aborts.
 */
export template<Catalog C>
class Writer {
public:
	Writer(Stamp stamp, Order<C> order)
	    : stamp_{stamp},
	      sections_{place(order)},
	      payload_offset_{payload_offset(sections_.size())} {}

	/** Hashes the next `bytes` of extent `key` and returns the file offset they belong at. */
	[[nodiscard]] auto write(Key key, std::span<std::byte const> bytes) -> std::uint64_t {
		core::invariant(current_ < sections_.size() && sections_[current_].key == key);
		auto const placement = sections_[current_].placement;
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
		return payload_offset_ + at;
	}

	/** The header of the complete image, every extent having been written. */
	[[nodiscard]] auto finish() && -> Finished {
		core::invariant(current_ == sections_.size());
		auto const payload = std::move(hasher_).finish();
		auto const stamped = Stamped{.card = stamp_.card, .checkpoint = stamp_.checkpoint, .description = C::description, .payload = payload};
		return {.header = encode_header(stamped, sections_), .file_bytes = payload_offset_ + payload_bytes(sections_), .payload = payload};
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

	Stamp stamp_;
	std::vector<Section> sections_;
	std::uint64_t payload_offset_;
	sha256::Hasher hasher_;
	/** The section being filled, and how many of its bytes have arrived. */
	std::size_t current_ = 0;
	std::uint64_t filled_ = 0;
	/** Payload bytes hashed so far, padding included. */
	std::uint64_t hashed_ = 0;
};

}
