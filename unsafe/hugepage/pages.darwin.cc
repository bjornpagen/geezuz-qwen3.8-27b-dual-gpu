// SPDX-License-Identifier: AGPL-3.0-only
module;

// The darwin page backend: ordinary anonymous mappings. Darwin on arm64 has no hugepages, and the development mac has
// no GPU whose pcie reads they would serve; the region exists there so that code over the driver double runs the
// same path as the production box.
#include <cerrno>
#include <sys/mman.h>

export module engine.hugepage:pages;

import std;
import engine.core;
import :error;

namespace engine::hugepage {

/** Maps `bytes` (a multiple of `page_bytes`) of zeroed memory in ordinary pages. */
[[nodiscard]] auto map_pages(std::size_t bytes) -> core::Result<std::span<std::byte>, HugepageError> {
	auto* const base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (base == MAP_FAILED) {
		return std::unexpected(HugepageError{.failure = errno == ENOMEM ? Failure::Exhausted : Failure::Refused});
	}
	/* SAFETY: the kernel mapped `bytes` writable bytes at `base`; the span lives until `unmap_pages`. */
	return std::span{static_cast<std::byte*>(base), bytes};
}

auto unmap_pages(std::span<std::byte> pages) -> void {
	// Unmapping a whole mapping this unit made cannot fail.
	std::ignore = ::munmap(pages.data(), pages.size());
}

}
