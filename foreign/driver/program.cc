// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver:program;

import std;
import engine.prims;
import engine.ptx;
import :abi;
import :memory;

namespace engine::driver {

/** A launch extent in blocks or threads. */
export struct Dim {
	std::uint32_t x;
	std::uint32_t y;
	std::uint32_t z;

	[[nodiscard]] constexpr auto count() const -> std::uint64_t {
		return std::uint64_t{x} * y * z;
	}
};

/**
 * How a kernel's blocks share the card. A co-resident kernel needs every block resident at once (it waits on other
 * blocks, as the persistent decode loop does): the driver launches it only if the whole grid fits, and refuses the
 * launch otherwise instead of deadlocking.
 */
export enum class Residency : std::uint8_t {
	Ordinary,
	CoResident,
};

/**
 * A kernel as the host launches it: the image holding it, its `extern "C"` entry name, its grid, its resource budget
 * (the same `prims::Budget` the kernel `static_assert`s, whose threads and shared bytes the launch uses), how its
 * blocks share the card, and the one parameter struct it takes by value. Every one is a compile-time fact, never a
 * launch argument.
 */
export template<class K>
concept Kernel = requires {
	{ K::image } -> std::convertible_to<ptx::Image>;
	{ K::entry } -> std::convertible_to<std::string_view>;
	{ K::grid } -> std::convertible_to<Dim>;
	{ K::budget } -> std::convertible_to<device::prims::Budget>;
	{ K::residency } -> std::convertible_to<Residency>;
	typename K::Params;
} && Wire<typename K::Params>;

/** The block a kernel launches with: its budget's threads, in one dimension. */
template<Kernel K>
inline constexpr Dim block_of = {.x = static_cast<std::uint32_t>(K::budget.threads), .y = 1, .z = 1};

/** The dynamic shared memory a kernel launches with: its budget's, all of it. */
template<Kernel K>
inline constexpr std::uint32_t shared_of = static_cast<std::uint32_t>(K::budget.shared_bytes);

/** The static laws of one kernel's declaration; a declaration that breaks one does not compile. */
template<Kernel K>
consteval auto lawful() -> bool {
	constexpr auto entry = std::string_view{K::entry};
	return !entry.empty() && entry.data()[entry.size()] == '\0' && Dim{K::grid}.count() > 0 && device::prims::fits(K::budget) &&
	       ptx::Image{K::image}.header == ptx::abi.header;
}

/**
 * A co-resident kernel's grid fits the cards at its budget's occupancy, so the driver's refusal of an oversized
 * cooperative grid is a budget error caught while compiling.
 */
template<Kernel K>
consteval auto co_resident() -> bool {
	constexpr auto resident = static_cast<std::uint64_t>(device::prims::sm86.sms) * static_cast<std::uint64_t>(K::budget.blocks_per_sm);
	return K::residency == Residency::Ordinary || Dim{K::grid}.count() <= resident;
}

export class Context;

/**
 * One loaded ptx image and the kernels the engine launches from it, looked up when it loads. Every kernel of one
 * program comes from the same image, so a kernel and its module share one lifetime by construction.
 */
export template<Kernel... Ks>
    requires(sizeof...(Ks) > 0)
class Program {
public:
	static_assert((lawful<Ks>() && ...), "a kernel's entry is a NUL-terminated name, its budget fits an sm_86 SM, and "
	                                     "its image shares the engine's header");
	static_assert((co_resident<Ks>() && ...), "a co-resident grid fits the cards at its budget's occupancy");
	/** The image every kernel of the program comes from. */
	static constexpr ptx::Image image = std::tuple_element_t<0, std::tuple<Ks...>>::image;

	static_assert(((ptx::Image{Ks::image}.text.data() == image.text.data()) && ...), "one program is one image");

	Program(Program const&) = delete;
	auto operator=(Program const&) -> Program& = delete;

	Program(Program&& other) noexcept
	    : unload_{std::exchange(other.unload_, {})},
	      ordinal_{other.ordinal_},
	      module_{std::exchange(other.module_, nullptr)},
	      functions_{other.functions_} {}

	auto operator=(Program&& other) noexcept -> Program& {
		std::swap(unload_, other.unload_);
		std::swap(ordinal_, other.ordinal_);
		std::swap(module_, other.module_);
		std::swap(functions_, other.functions_);
		return *this;
	}

	~Program() {
		// A moved-from program holds nothing; a failed unload at teardown has no remedy.
		if (unload_.function != nullptr) {
			std::ignore = unload_(module_);
		}
	}

	[[nodiscard]] auto ordinal() const -> CUdevice {
		return ordinal_;
	}

	/** Where `K` sits among the program's kernels; naming a kernel the program does not hold does not compile. */
	template<Kernel K>
	[[nodiscard]] static consteval auto position() -> std::size_t {
		constexpr auto matches = std::array{std::same_as<K, Ks>...};
		static_assert(std::ranges::count(matches, true) == 1, "the kernel is in the program exactly once");
		return core_position(matches);
	}

private:
	friend class Context;

	[[nodiscard]] static consteval auto core_position(std::array<bool, sizeof...(Ks)> matches) -> std::size_t {
		return static_cast<std::size_t>(std::ranges::find(matches, true) - matches.begin());
	}

	Program(Api const& api, CUdevice ordinal, CUmodule module) : unload_{api.module_unload}, ordinal_{ordinal}, module_{module} {}

	decltype(Api::module_unload) unload_;
	CUdevice ordinal_;
	CUmodule module_;
	std::array<CUfunction, sizeof...(Ks)> functions_{};
};

}
