// SPDX-License-Identifier: AGPL-3.0-only
export module engine.bake:error;

import std;
import engine.core;
import engine.file;
import engine.file_write;
import engine.safetensors;
import engine.sha256;

namespace engine::bake {

/** A checkpoint file the pinned list names could not be mapped. */
export struct Unreadable {
	std::string file;
	file::FileError error;

	[[nodiscard]] auto retry() const -> core::Retry {
		return error.retry();
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("cannot map checkpoint file {}", file);
	}

	[[nodiscard]] auto operator==(Unreadable const&) const -> bool = default;
};

/** A checkpoint file's bytes are not the ones published for it: a re-upload, a partial download, or corruption. */
export struct DigestMismatch {
	std::string file;
	sha256::Digest expected;
	sha256::Digest actual;

	[[nodiscard]] auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("checkpoint file {} has SHA-256 {}, not the published {}", file, actual, expected);
	}

	[[nodiscard]] auto operator==(DigestMismatch const&) const -> bool = default;
};

/** A verified checkpoint file is not a safetensors file the engine reads. */
export struct NotSafetensors {
	std::string file;
	safetensors::Error error;

	[[nodiscard]] auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("checkpoint file {} is not a readable safetensors file", file);
	}

	[[nodiscard]] auto operator==(NotSafetensors const&) const -> bool = default;
};

/** A tensor the model description needs is in no checkpoint file. */
export struct MissingTensor {
	std::string tensor;

	[[nodiscard]] auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("the checkpoint has no tensor {}", tensor);
	}

	[[nodiscard]] auto operator==(MissingTensor const&) const -> bool = default;
};

/** A tensor's stored dtype or shape is not the model description's. */
export struct WrongTensor {
	std::string tensor;

	[[nodiscard]] auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("tensor {} is not stored with the model description's dtype and shape", tensor);
	}

	[[nodiscard]] auto operator==(WrongTensor const&) const -> bool = default;
};

/** A tensor name appears in two checkpoint files. */
export struct DuplicateTensor {
	std::string tensor;

	[[nodiscard]] auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("tensor {} is stored twice", tensor);
	}

	[[nodiscard]] auto operator==(DuplicateTensor const&) const -> bool = default;
};

/** A tensor that is neither the model description's nor the vision tower's: a checkpoint of another model. */
export struct UnexpectedTensor {
	std::string tensor;

	[[nodiscard]] auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("tensor {} belongs to no part of this model", tensor);
	}

	[[nodiscard]] auto operator==(UnexpectedTensor const&) const -> bool = default;
};

/** An e4m3 weight holds a nan code (0x7f or 0xff), which the device has no path for; row and column are the tensor's. */
export struct NanCode {
	std::string tensor;
	std::size_t row;
	std::size_t column;

	[[nodiscard]] auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("tensor {} holds an e4m3 nan code at row {}, column {}", tensor, row, column);
	}

	[[nodiscard]] auto operator==(NanCode const&) const -> bool = default;
};

/**
 * The proven bound on |v| reaches the limit under which attention's scaled fp32 accumulators cannot overflow: an
 * engine on this checkpoint could lose its exactness, so it is not baked.
 */
export struct AttentionOverflow {
	double bound;
	std::string tensor;

	[[nodiscard]] auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("the bound {:e} on |v| from {} reaches 2^70: attention could overflow", bound, tensor);
	}

	[[nodiscard]] auto operator==(AttentionOverflow const&) const -> bool = default;
};

/** An image could not be written into place. */
export struct WriteFailed {
	std::string image;
	file_write::WriteError error;

	[[nodiscard]] auto retry() const -> core::Retry {
		return error.retry();
	}

	[[nodiscard]] auto describe() const -> std::string {
		return std::format("cannot write the {} image into place", image);
	}

	[[nodiscard]] auto operator==(WriteFailed const&) const -> bool = default;
};

/** Why a checkpoint was not baked. Every refusal leaves no image under a final name that was not there before. */
export struct [[nodiscard]] BakeError {
	std::variant<Unreadable, DigestMismatch, NotSafetensors, MissingTensor, WrongTensor, DuplicateTensor, UnexpectedTensor, NanCode,
	             AttentionOverflow, WriteFailed>
	    cause;

	[[nodiscard]] auto retry() const -> core::Retry {
		return std::visit([](auto const& each) { return each.retry(); }, cause);
	}

	/** One line saying what was refused, for a person. */
	[[nodiscard]] auto describe() const -> std::string {
		return std::visit([](auto const& each) { return each.describe(); }, cause);
	}

	[[nodiscard]] auto operator==(BakeError const&) const -> bool = default;
};

[[nodiscard]] auto refuse(auto cause) -> std::unexpected<BakeError> {
	return std::unexpected(BakeError{.cause = std::move(cause)});
}

}
