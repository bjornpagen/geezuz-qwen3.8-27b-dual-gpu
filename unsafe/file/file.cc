// SPDX-License-Identifier: AGPL-3.0-only
module;

// The posix file and mapping calls the one read path needs.
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

export module engine.file;

import std;
import engine.core;

namespace engine::file {

/** Why a file could not be mapped. Each kind is a different decision for the caller. */
export enum class Failure : std::uint8_t {
	/** No file at the path, or a path component is not a directory. */
	NotFound,
	/** The process may not read the file. */
	Permission,
	/** The path names a directory, device, fifo or socket: only regular files have stable contents to map. */
	NotRegular,
	/** The file has no bytes: there is nothing to map, and an empty input is never a valid file for its readers. */
	Empty,
	/** The process ran out of descriptors or address space: retrying later may succeed. */
	Exhausted,
	/** The kernel refused to map a regular, readable file. */
	MapFailed,
	/** Any other refusal to open or inspect the file (an i/o error, a symlink loop, an overlong name). */
	Unreadable,
};

export struct [[nodiscard]] FileError {
	Failure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case Failure::Exhausted:
			return core::Retry::Transient;
		case Failure::NotFound:
		case Failure::Permission:
		case Failure::NotRegular:
		case Failure::Empty:
		case Failure::MapFailed:
		case Failure::Unreadable:
			return core::Retry::Permanent;
		}
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(FileError const&) const -> bool = default;
};

[[nodiscard]] auto open_failure(int error) -> Failure {
	switch (error) {
	case ENOENT:
	case ENOTDIR:
		return Failure::NotFound;
	case EACCES:
	case EPERM:
		return Failure::Permission;
	case EISDIR:
		return Failure::NotRegular;
	case EMFILE:
	case ENFILE:
	case ENOMEM:
		return Failure::Exhausted;
	default:
		return Failure::Unreadable;
	}
}

/** Owns one open descriptor for the span of a mapping call; the mapping outlives it. */
class Descriptor {
public:
	explicit Descriptor(int fd) : fd_{fd} {}

	Descriptor(Descriptor const&) = delete;
	auto operator=(Descriptor const&) -> Descriptor& = delete;
	Descriptor(Descriptor&&) = delete;
	auto operator=(Descriptor&&) -> Descriptor& = delete;

	~Descriptor() {
		std::ignore = ::close(fd_);
	}

	[[nodiscard]] auto get() const -> int {
		return fd_;
	}

private:
	int fd_;
};

/**
 * A whole regular file mapped read-only, private to this process. The bytes stay valid exactly as long as the
 * mapping lives; a moved-from mapping holds no bytes. Mapping instead of reading keeps a 29 GB checkpoint out of
 * the heap: pages are faulted in as they are touched and dropped by the kernel under pressure.
 *
 * The file must not be truncated by another process while mapped: touching a page past the new end raises SIGBUS.
 * The engine maps only its own inputs (checkpoint, baked image), which nothing rewrites while it runs.
 */
export class Mapping {
public:
	/** Maps the whole file at `path`. On failure nothing is left open or mapped. */
	[[nodiscard]] static auto open(std::filesystem::path const& path) -> core::Result<Mapping, FileError> {
		// O_NONBLOCK keeps opening a fifo from waiting for a writer; the regular-file check below then refuses it.
		auto const fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
		if (fd < 0) {
			return std::unexpected(FileError{.failure = open_failure(errno)});
		}
		auto const descriptor = Descriptor{fd};
		struct stat status{};
		if (::fstat(descriptor.get(), &status) != 0) {
			return std::unexpected(FileError{.failure = open_failure(errno)});
		}
		if (!S_ISREG(status.st_mode)) {
			return std::unexpected(FileError{.failure = Failure::NotRegular});
		}
		if (status.st_size <= 0) {
			return std::unexpected(FileError{.failure = Failure::Empty});
		}
		auto const size = core::try_narrow<std::size_t>(status.st_size);
		if (!size) {
			return std::unexpected(FileError{.failure = Failure::Exhausted});
		}
		auto* const base = ::mmap(nullptr, *size, PROT_READ, MAP_PRIVATE, descriptor.get(), 0);
		if (base == MAP_FAILED) {
			return std::unexpected(FileError{.failure = errno == ENOMEM ? Failure::Exhausted : Failure::MapFailed});
		}
		return Mapping{base, *size};
	}

	Mapping(Mapping const&) = delete;
	auto operator=(Mapping const&) -> Mapping& = delete;

	Mapping(Mapping&& other) noexcept : base_{std::exchange(other.base_, nullptr)}, size_{std::exchange(other.size_, 0)} {}

	auto operator=(Mapping&& other) noexcept -> Mapping& {
		if (this != &other) {
			release();
			base_ = std::exchange(other.base_, nullptr);
			size_ = std::exchange(other.size_, 0);
		}
		return *this;
	}

	~Mapping() {
		release();
	}

	/** The file's bytes, valid while this mapping lives and is not moved from. */
	[[nodiscard]] auto bytes() const -> std::span<std::byte const> {
		if (base_ == nullptr) {
			return {};
		}
		/* SAFETY: base_ is a live PROT_READ mapping of exactly size_ bytes, owned by this object. */
		return {static_cast<std::byte const*>(base_), size_};
	}

private:
	Mapping(void* base, std::size_t size) : base_{base}, size_{size} {}

	auto release() -> void {
		if (base_ != nullptr) {
			std::ignore = ::munmap(base_, size_);
		}
	}

	void* base_;
	std::size_t size_;
};

}
