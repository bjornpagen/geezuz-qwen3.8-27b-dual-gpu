// SPDX-License-Identifier: AGPL-3.0-only
export module engine.hugepage:region;

import std;
import engine.core;
import :error;
import :pages;

namespace engine::hugepage {

/**
 * Zeroed, writable, page-aligned host memory in hugepages, unmapped when its owner dies: what the driver pins and maps
 * into the cards for the embedding table and the rings, so the cards' pcie reads walk few, large pages. Its size is the
 * requested size rounded up to whole pages; a moved-from region holds no bytes.
 */
export class Region {
public:
	[[nodiscard]] static auto allocate(std::size_t bytes) -> core::Result<Region, HugepageError> {
		core::invariant(bytes > 0);
		auto const rounded = (bytes + page_bytes - 1) / page_bytes * page_bytes;
		return map_pages(rounded).transform([](std::span<std::byte> pages) {
			return Region{pages};
		});
	}

	Region(Region const&) = delete;
	auto operator=(Region const&) -> Region& = delete;

	Region(Region&& other) noexcept : pages_{std::exchange(other.pages_, {})} {}

	auto operator=(Region&& other) noexcept -> Region& {
		std::swap(pages_, other.pages_);
		return *this;
	}

	~Region() {
		// A moved-from region holds no bytes.
		if (!pages_.empty()) {
			unmap_pages(pages_);
		}
	}

	[[nodiscard]] auto bytes() -> std::span<std::byte> {
		return pages_;
	}

	[[nodiscard]] auto bytes() const -> std::span<std::byte const> {
		return pages_;
	}

private:
	explicit Region(std::span<std::byte> pages) : pages_{pages} {}

	std::span<std::byte> pages_;
};

}
