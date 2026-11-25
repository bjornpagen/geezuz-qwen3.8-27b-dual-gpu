// SPDX-License-Identifier: AGPL-3.0-only
//
// A kernel's resource budget as its own image records it. Where the budget is fixed by device code the host cannot
// see (a shared-memory layout, a warp-role split), the device compiler emits it beside the kernel, and the host reads
// it back from the image while compiling, so the launch and the kernel cannot disagree.
export module engine.ptx:budget;

import std;
import engine.core;
import engine.prims;
import :image;

namespace engine::ptx {

/**
 * The budget record `engine_budget_<entry>` the image holds, as the device compiler emitted it: four little-endian
 * 32-bit fields (threads, registers per thread, shared bytes, blocks per SM) as up to 16 byte values, trailing zeros
 * omitted. Records precede the first entry; a missing or malformed record does not compile.
 */
export [[nodiscard]] consteval auto recorded_budget(Image image, std::string_view entry) -> device::prims::Budget {
	auto const text = std::string_view{image.text.data(), image.text.size() - 1};
	auto const head = text.substr(0, text.find(".entry"));
	auto const prefix = std::string_view{"engine_budget_"};
	auto const suffix = std::string_view{"[16] = {"};
	auto found = std::string_view::npos;
	for (auto at = head.find(prefix); at != std::string_view::npos; at = head.find(prefix, at + 1)) {
		auto const rest = head.substr(at + prefix.size());
		if (rest.starts_with(entry) && rest.substr(entry.size()).starts_with(suffix)) {
			found = at + prefix.size() + entry.size() + suffix.size();
			break;
		}
	}
	core::invariant(found != std::string_view::npos);
	auto const values = head.substr(found, head.find('}', found) - found);
	auto bytes = std::array<std::uint32_t, 16>{};
	auto index = std::size_t{0};
	auto value = std::uint32_t{0};
	for (auto const character : values) {
		if (character == ',') {
			bytes.at(index++) = value;
			value = 0;
		} else if (character >= '0' && character <= '9') {
			value = value * 10 + static_cast<std::uint32_t>(character - '0');
		} else {
			core::invariant(character == ' ');
		}
	}
	bytes.at(index) = value;
	auto const field = [&](std::size_t first) {
		auto const word = bytes.at(first) | bytes.at(first + 1) << 8U | bytes.at(first + 2) << 16U | bytes.at(first + 3) << 24U;
		return static_cast<int>(word);
	};
	return {.threads = field(0), .registers_per_thread = field(4), .shared_bytes = field(8), .blocks_per_sm = field(12)};
}

}
