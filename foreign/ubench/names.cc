// SPDX-License-Identifier: AGPL-3.0-only
//
// How the host names a kernel and learns its budget. A sweep's kernels are one template each on the host, so an
// entry's name is built from the shape's parameters at compile time, in the spelling the device's extern "C" wrapper
// uses; loading checks the two agree. A kernel the host has no shape for (the primitives' self-checks) has its budget
// read from the budget record its own image carries, so the host never restates it.
export module engine.ubench_device:names;

import std;
import engine.core;
import engine.prims;
import engine.ptx;

namespace engine::ubench_device {

/** A kernel entry's name, NUL-terminated inside fixed storage so its view can reach the loader as a C string. */
export struct EntryName {
	std::array<char, 64> text{};
	std::size_t size = 0;

	[[nodiscard]] constexpr auto view() const -> std::string_view {
		return {text.data(), size};
	}
};

consteval void append(EntryName& name, std::string_view part) {
	for (auto const character : part) {
		core::invariant(name.size + 1 < name.text.size());
		name.text.at(name.size++) = character;
	}
}

consteval void append(EntryName& name, int number) {
	core::invariant(number >= 0);
	auto digits = std::array<char, 10>{};
	auto count = std::size_t{0};
	do {
		digits.at(count++) = static_cast<char>('0' + number % 10);
		number /= 10;
	} while (number != 0);
	while (count > 0) {
		append(name, std::string_view{&digits.at(--count), 1});
	}
}

/** The name the parts spell, in order: text as written, numbers in decimal. */
export template<class... Parts>
[[nodiscard]] consteval auto entry_name(Parts... parts) -> EntryName {
	auto name = EntryName{};
	(append(name, parts), ...);
	return name;
}

/**
 * The budget record `engine_budget_<entry>` an image holds, as the device compiler emitted it: four little-endian
 * 32-bit fields (threads, registers per thread, shared bytes, blocks per SM) as up to 16 byte values, trailing zeros
 * omitted. Records precede the first entry; a missing or malformed record does not compile.
 */
export [[nodiscard]] consteval auto recorded_budget(ptx::Image image, std::string_view entry) -> device::prims::Budget {
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
