// SPDX-License-Identifier: AGPL-3.0-only
module;

// The linux page backend: anonymous mappings from the reserved hugetlb pool, faulted in when mapped.
#include <cerrno>
#include <sys/mman.h>

export module engine.hugepage:pages;

import std;
import engine.core;
import :error;

namespace engine::hugepage {

/**
 * Maps `bytes` (a multiple of `page_bytes`) of zeroed memory backed by hugepages of the system's default size, 2 MiB
 * on x86_64. Every page is faulted in now, at setup, so the serving path never takes a page fault in it.
 */
[[nodiscard]] auto map_pages(std::size_t bytes) -> core::Result<std::span<std::byte>, HugepageError> {
	auto* const base = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
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
