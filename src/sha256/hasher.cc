// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sha256:hasher;

import std;
import engine.core;
import engine.sha256_block;
import :digest;
import :portable;

namespace engine::sha256 {

/** The standard bounds a message below 2^64 bits. */
inline constexpr std::uint64_t max_message_bytes = std::uint64_t{1} << 61U;

/**
 * Streaming SHA-256: feed a message in pieces of any size, in order, then finish. The digest depends only on the
 * concatenated bytes, never on where the pieces were cut. Usable in constant evaluation, where it hashes data the
 * build knows (the model description), and at run time, where it runs on the machine's SHA instructions.
 */
export class Hasher {
public:
	constexpr Hasher() = default;

	/** Appends `bytes` to the message. Whole blocks pass straight from `bytes` to the compression function. */
	constexpr auto update(std::span<std::byte const> bytes) -> void {
		core::invariant(bytes.size() < max_message_bytes - total_);
		total_ += bytes.size();
		if (pending_size_ > 0) {
			auto const taken = std::min(bytes.size(), block_bytes - pending_size_);
			std::ranges::copy(bytes.first(taken), std::span{pending_}.subspan(pending_size_).begin());
			pending_size_ += taken;
			bytes = bytes.subspan(taken);
			// The message ends inside this block for now; the next update or the padding completes it.
			if (pending_size_ < block_bytes) {
				return;
			}
			state_ = compress(state_, pending_);
			pending_size_ = 0;
		}
		auto const whole = bytes.size() - (bytes.size() % block_bytes);
		state_ = compress(state_, bytes.first(whole));
		std::ranges::copy(bytes.subspan(whole), pending_.begin());
		pending_size_ = bytes.size() - whole;
	}

	/** Appends the bytes of `text`. */
	constexpr auto update(std::string_view text) -> void {
		auto chunk = std::array<std::byte, block_bytes>{};
		while (!text.empty()) {
			auto const piece = text.substr(0, block_bytes);
			std::ranges::transform(piece, chunk.begin(), [](char character) { return std::bit_cast<std::byte>(character); });
			update(std::span<std::byte const>{chunk}.first(piece.size()));
			text.remove_prefix(piece.size());
		}
	}

	/** Pads the message and returns its digest; the hasher is spent. */
	[[nodiscard]] constexpr auto finish() && -> Digest {
		auto block = pending_;
		std::ranges::fill(std::span{block}.subspan(pending_size_), std::byte{0});
		block.at(pending_size_) = std::byte{0x80};
		// The 8-byte length must follow the 0x80 marker inside one block; when it does not fit, the marker's block is
		// compressed alone and the length goes into a block of zeros.
		if (pending_size_ + 1 > block_bytes - length_bytes) {
			state_ = compress(state_, block);
			block.fill(std::byte{0});
		}
		auto const bits = total_ * 8;
		for (auto index = std::size_t{0}; index < length_bytes; ++index) {
			block.at(block_bytes - 1 - index) = std::byte{core::narrow<std::uint8_t>((bits >> (8 * index)) & 0xffU)};
		}
		state_ = compress(state_, block);
		auto result = Digest{};
		for (auto index = std::size_t{0}; index < result.bytes.size(); ++index) {
			auto const word = state_.at(index / 4);
			result.bytes.at(index) = std::byte{core::narrow<std::uint8_t>((word >> (24 - (8 * (index % 4)))) & 0xffU)};
		}
		return result;
	}

private:
	using State = sha256_block::State;

	static constexpr std::size_t block_bytes = sha256_block::block_bytes;
	static constexpr std::size_t length_bytes = 8;

	[[nodiscard]] static constexpr auto compress(State state, std::span<std::byte const> blocks) -> State {
		// The SHA instructions have no constant evaluation, so a digest the build computes takes the portable rounds.
		if consteval {
			return portable_compress(state, blocks);
		}
		return sha256_block::compress(state, blocks);
	}

	State state_ = sha256_block::initial_state;
	std::array<std::byte, block_bytes> pending_ = {};
	std::size_t pending_size_ = 0;
	std::uint64_t total_ = 0;
};

/** The digest of `bytes`. */
export [[nodiscard]] constexpr auto digest(std::span<std::byte const> bytes) -> Digest {
	auto hasher = Hasher{};
	hasher.update(bytes);
	return std::move(hasher).finish();
}

/** The digest of the bytes of `text`. */
export [[nodiscard]] constexpr auto digest(std::string_view text) -> Digest {
	auto hasher = Hasher{};
	hasher.update(text);
	return std::move(hasher).finish();
}

}

namespace engine::sha256 {

// Bytes one constant evaluation hashes: 16 blocks of portable rounds stay well inside clang's per-evaluation step
// budget, which a few kilobytes in one evaluation exceed.
inline constexpr std::size_t stage_bytes = 1024;

template<auto const& text, std::size_t stage>
constexpr Hasher hashed_through = [] {
	auto hasher = Hasher{};
	if constexpr (stage > 0) {
		hasher = hashed_through<text, stage - 1>;
	}
	hasher.update(std::string_view{text.data(), text.size()}.substr(stage * stage_bytes, stage_bytes));
	return hasher;
}();

/**
 * The digest of text the build holds in a program-lifetime character array, computed by the build. The text is hashed
 * a stage per constant evaluation, so its length is not bounded by one evaluation's step budget.
 */
export template<auto const& text>
inline constexpr Digest constant_digest = [] {
	auto hasher = hashed_through<text, std::size(text) / stage_bytes>;
	return std::move(hasher).finish();
}();

}
