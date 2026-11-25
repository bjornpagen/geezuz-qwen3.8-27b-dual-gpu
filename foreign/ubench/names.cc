// SPDX-License-Identifier: AGPL-3.0-only
//
// How the host names a kernel. A sweep's kernels are one template each on the host, so an entry's name is built from
// the shape's parameters at compile time, in the spelling the device's extern "C" wrapper uses; loading checks the two
// agree.
export module engine.ubench_device:names;

import std;
import engine.core;

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

}
