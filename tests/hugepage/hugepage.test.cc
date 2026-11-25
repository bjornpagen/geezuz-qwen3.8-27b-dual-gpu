// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.hugepage;
import engine.testing;

namespace {

using engine::hugepage::page_bytes;
using engine::hugepage::Region;

/** A region is the request rounded up to whole pages, page-aligned, zeroed, and writable throughout. */
[[nodiscard]] auto regions_are_whole_zeroed_writable_pages() -> bool {
	auto holds = true;
	for (auto const bytes : {std::size_t{1}, page_bytes - 1, page_bytes, page_bytes + 1, 3 * page_bytes + 17}) {
		auto region = Region::allocate(bytes);
		if (!region) {
			return false;
		}
		auto const pages = region->bytes();
		auto const expected = (bytes + page_bytes - 1) / page_bytes * page_bytes;
		auto const address = std::bit_cast<std::uintptr_t>(pages.data());
		holds = holds && pages.size() == expected && address % 4096 == 0 && std::ranges::all_of(pages, [](std::byte value) {
			        return value == std::byte{0};
		        });
		auto index = std::uint8_t{0};
		for (auto& value : pages) {
			value = std::byte{index++};
		}
		index = 0;
		holds = holds && std::ranges::all_of(pages, [&](std::byte value) {
			        return value == std::byte{index++};
		        });
	}
	return holds;
}

[[nodiscard]] auto a_moved_region_has_one_owner() -> bool {
	auto first = Region::allocate(page_bytes);
	if (!first) {
		return false;
	}
	auto const data = first->bytes().data();
	auto second = std::move(*first);
	auto const moved_out = first->bytes().empty();
	auto third = Region::allocate(1);
	if (!third) {
		return false;
	}
	third = std::move(second);
	return moved_out && third->bytes().data() == data && third->bytes().size() == page_bytes;
}

[[nodiscard]] auto exhaustion_is_transient_and_refusal_permanent() -> bool {
	using engine::hugepage::Failure;
	using engine::hugepage::HugepageError;
	return HugepageError{.failure = Failure::Exhausted}.retry() == engine::core::Retry::Transient &&
	       HugepageError{.failure = Failure::Refused}.retry() == engine::core::Retry::Permanent;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "a region is the request rounded up to whole pages, aligned, zeroed and writable",
	        .holds = regions_are_whole_zeroed_writable_pages(),
	    },
	    engine::testing::Case{
	        .name = "moving a region moves its pages, leaving the source empty",
	        .holds = a_moved_region_has_one_owner(),
	    },
	    engine::testing::Case{
	        .name = "an exhausted hugepage pool is transient, any other refusal permanent",
	        .holds = exhaustion_is_transient_and_refusal_permanent(),
	    },
	};
	return engine::testing::report(cases);
}
