// SPDX-License-Identifier: AGPL-3.0-only
module;

// The posix calls a positional writer needs: create beside the destination, size, write at offsets, flush, rename.
#include <cerrno>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

export module engine.file_write;

import std;
import engine.core;

namespace engine::file_write {

/** Why a file could not be written into place. Each kind is a different decision for the caller. */
export enum class Failure : std::uint8_t {
	/** The destination's directory does not exist, or a path component is not a directory. */
	NotFound,
	/** The process may not create or replace files in the destination's directory. */
	Permission,
	/** The file system is full or the process's file size limit is reached: freeing space and retrying may succeed. */
	NoSpace,
	/** The process ran out of descriptors: retrying later may succeed. */
	Exhausted,
	/** The destination exists and is a directory, which a rename cannot replace. */
	IsDirectory,
	/** Any other refusal by the file system (an i/o error, a read-only file system, an overlong name). */
	Io,
};

export struct [[nodiscard]] WriteError {
	Failure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case Failure::NoSpace:
		case Failure::Exhausted:
			return core::Retry::Transient;
		case Failure::NotFound:
		case Failure::Permission:
		case Failure::IsDirectory:
		case Failure::Io:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(WriteError const&) const -> bool = default;
};

[[nodiscard]] auto failure_of(int error) -> Failure {
	switch (error) {
	case ENOENT:
	case ENOTDIR:
		return Failure::NotFound;
	case EACCES:
	case EPERM:
		return Failure::Permission;
	case ENOSPC:
	case EDQUOT:
	case EFBIG:
		return Failure::NoSpace;
	case EMFILE:
	case ENFILE:
		return Failure::Exhausted;
	case EISDIR:
		return Failure::IsDirectory;
	default:
		return Failure::Io;
	}
}

[[nodiscard]] auto fail() -> std::unexpected<WriteError> {
	return std::unexpected(WriteError{.failure = failure_of(errno)});
}

/** Flushes the directory at `path`, so a rename into it survives a crash. */
[[nodiscard]] auto sync_directory(std::filesystem::path const& path) -> core::Result<void, WriteError> {
	auto const fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
	if (fd < 0) {
		return fail();
	}
	auto const synced = ::fsync(fd);
	auto const error = errno;
	std::ignore = ::close(fd);
	if (synced != 0) {
		return std::unexpected(WriteError{.failure = failure_of(error)});
	}
	return {};
}

/**
 * A file being written at offsets, then put in place whole. It is created beside its destination under a unique
 * temporary name, already its final size and reading as zeros wherever nothing has been written (sparse where the file
 * system allows); `commit` flushes it and renames it over the destination in one step. Until then the destination is
 * untouched, and a file dropped uncommitted, on any path, is removed: nothing half-written is ever under the final
 * name.
 */
export class Output {
public:
	/** Creates the temporary file for `destination`, `bytes` long. On failure nothing is left behind. */
	[[nodiscard]] static auto create(std::filesystem::path const& destination, std::uint64_t bytes) -> core::Result<Output, WriteError> {
		auto const length = core::try_narrow<off_t>(bytes);
		if (!length) {
			return std::unexpected(WriteError{.failure = Failure::NoSpace});
		}
		auto pattern = destination.string() + ".partial-XXXXXX";
		auto const fd = ::mkstemp(pattern.data());
		if (fd < 0) {
			return fail();
		}
		auto output = Output{fd, std::filesystem::path{pattern}, destination, bytes};
		if (::fchmod(fd, 0644) != 0 || ::ftruncate(fd, *length) != 0) {
			return fail();
		}
		return output;
	}

	Output(Output const&) = delete;
	auto operator=(Output const&) -> Output& = delete;

	Output(Output&& other) noexcept
	    : fd_{std::exchange(other.fd_, -1)},
	      temporary_{std::move(other.temporary_)},
	      destination_{std::move(other.destination_)},
	      bytes_{other.bytes_} {}

	auto operator=(Output&& other) noexcept -> Output& {
		if (this != &other) {
			discard();
			fd_ = std::exchange(other.fd_, -1);
			temporary_ = std::move(other.temporary_);
			destination_ = std::move(other.destination_);
			bytes_ = other.bytes_;
		}
		return *this;
	}

	~Output() {
		discard();
	}

	/** Writes all of `bytes` at `offset`; the range must lie inside the file's size, or the caller has a bug. */
	[[nodiscard]] auto write(std::uint64_t offset, std::span<std::byte const> bytes) -> core::Result<void, WriteError> {
		core::invariant(fd_ >= 0 && offset <= bytes_ && bytes.size() <= bytes_ - offset);
		while (!bytes.empty()) {
			/* SAFETY: the span is live for the call and pwrite reads at most its size from it. */
			auto const written = ::pwrite(fd_, bytes.data(), bytes.size(), core::narrow<off_t>(offset));
			if (written < 0) {
				if (errno == EINTR) {
					continue;
				}
				return fail();
			}
			auto const count = core::narrow<std::size_t>(written);
			bytes = bytes.subspan(count);
			offset += count;
		}
		return {};
	}

	/**
	 * Flushes the file, renames it over the destination and flushes the directory. On failure the file is removed and
	 * the destination is as it was, except when only the final directory flush failed: then the destination holds the
	 * complete file, which a crash before the next flush could still lose.
	 */
	[[nodiscard]] auto commit() && -> core::Result<void, WriteError> {
		core::invariant(fd_ >= 0);
		if (::fsync(fd_) != 0 || ::rename(temporary_.c_str(), destination_.c_str()) != 0) {
			auto const error = errno;
			discard();
			return std::unexpected(WriteError{.failure = failure_of(error)});
		}
		std::ignore = ::close(std::exchange(fd_, -1));
		return sync_directory(destination_.has_parent_path() ? destination_.parent_path() : std::filesystem::path{"."});
	}

private:
	Output(int fd, std::filesystem::path temporary, std::filesystem::path destination, std::uint64_t bytes)
	    : fd_{fd}, temporary_{std::move(temporary)}, destination_{std::move(destination)}, bytes_{bytes} {}

	auto discard() -> void {
		if (fd_ >= 0) {
			std::ignore = ::close(std::exchange(fd_, -1));
			std::ignore = ::unlink(temporary_.c_str());
		}
	}

	int fd_;
	std::filesystem::path temporary_;
	std::filesystem::path destination_;
	std::uint64_t bytes_;
};

}
