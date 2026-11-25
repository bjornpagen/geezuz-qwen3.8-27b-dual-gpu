// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver:memory;

import std;
import engine.core;
import engine.hugepage;
import :abi;

namespace engine::driver {

/** A value that crosses to the device as bytes: the device reads it with the same layout the host wrote. */
export template<class T>
concept Wire = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

/**
 * An owner of host bytes whose address does not change when the owner moves (a file mapping, a hugepage region, a heap
 * buffer): the driver can pin them, and the pin survives the owner being moved into a transfer and back.
 */
export template<class Owner>
concept HostBytes = std::movable<Owner> && requires(Owner const& owner) {
	{ owner.bytes() } -> std::convertible_to<std::span<std::byte const>>;
};

/** A copy's pinned host source: the bytes a copy may read straight over pcie. */
export template<class Source>
concept PinnedBytes = std::movable<Source> && requires(Source const& source) {
	{ source.pinned() } -> std::convertible_to<std::span<std::byte const>>;
};

export class Context;

/** Device memory holding `Count` values of `T` on one card, freed when its owner dies. Only a context allocates it. */
export template<Wire T, std::size_t Count>
class DeviceArray {
public:
	static_assert(Count > 0, "device memory is planned at build time; an empty array is a planning error");
	static constexpr std::size_t bytes = sizeof(T) * Count;

	DeviceArray(DeviceArray const&) = delete;
	auto operator=(DeviceArray const&) -> DeviceArray& = delete;

	DeviceArray(DeviceArray&& other) noexcept
	    : free_{std::exchange(other.free_, {})}, ordinal_{other.ordinal_}, address_{std::exchange(other.address_, 0)} {}

	auto operator=(DeviceArray&& other) noexcept -> DeviceArray& {
		std::swap(free_, other.free_);
		std::swap(ordinal_, other.ordinal_);
		std::swap(address_, other.address_);
		return *this;
	}

	~DeviceArray() {
		// A moved-from owner holds nothing. A free that fails at teardown means the context is already poisoned, which
		// the call that poisoned it reported; there is nothing left to do with the memory.
		if (free_.function != nullptr) {
			std::ignore = free_(address_);
		}
	}

	/** The device address of the first value, as kernel parameters carry it. */
	[[nodiscard]] auto address() const -> CUdeviceptr {
		return address_;
	}

	[[nodiscard]] auto ordinal() const -> CUdevice {
		return ordinal_;
	}

private:
	friend class Context;

	DeviceArray(decltype(Api::mem_free) free, CUdevice ordinal, CUdeviceptr address) : free_{free}, ordinal_{ordinal}, address_{address} {}

	decltype(Api::mem_free) free_;
	CUdevice ordinal_;
	CUdeviceptr address_;
};

/**
 * `Count` values of `T` in hugepage-backed host memory the driver has pinned and mapped into both cards' address
 * spaces: the embedding table, the rings, staging for copies. The values start value-initialized (zero for every `Wire`
 * aggregate). Host and cards both reach it: what each side may touch when is the protocol of whatever lives in it,
 * never this owner's. Unpinned, then unmapped, when its owner dies.
 */
export template<Wire T, std::size_t Count>
class MappedArray {
public:
	static_assert(Count > 0, "pinned memory is planned at build time; an empty array is a planning error");
	static_assert(alignof(T) <= hugepage::page_bytes, "pinned memory is page-aligned");
	static constexpr std::size_t bytes = sizeof(T) * Count;

	MappedArray(MappedArray const&) = delete;
	auto operator=(MappedArray const&) -> MappedArray& = delete;

	MappedArray(MappedArray&& other) noexcept
	    : region_{std::move(other.region_)},
	      unregister_{std::exchange(other.unregister_, {})},
	      host_{std::exchange(other.host_, nullptr)},
	      device_{std::exchange(other.device_, 0)} {}

	auto operator=(MappedArray&& other) noexcept -> MappedArray& {
		std::swap(region_, other.region_);
		std::swap(unregister_, other.unregister_);
		std::swap(host_, other.host_);
		std::swap(device_, other.device_);
		return *this;
	}

	~MappedArray() {
		// Unpinned before the region unmaps (it is declared first, so it dies last). A moved-from or never-pinned owner
		// holds no pin, and a failed unpin at teardown has no remedy.
		if (unregister_.function != nullptr) {
			std::ignore = unregister_(region_.bytes().data());
		}
	}

	[[nodiscard]] auto host() -> std::span<T, Count> {
		return std::span<T, Count>{host_, Count};
	}

	[[nodiscard]] auto host() const -> std::span<T const, Count> {
		return std::span<T const, Count>{host_, Count};
	}

	/** The pinned bytes a copy reads, when this is a copy's source. */
	[[nodiscard]] auto pinned() const -> std::span<std::byte const> {
		return std::as_bytes(host());
	}

	/** The address the cards reach the same memory at, as kernel parameters carry it. */
	[[nodiscard]] auto device_address() const -> CUdeviceptr {
		return device_;
	}

private:
	friend class Context;

	explicit MappedArray(hugepage::Region region)
	    : region_{std::move(region)}, unregister_{}, host_{reinterpret_cast<T*>(region_.bytes().data())}, device_{0} {
		/* SAFETY: the region is at least `bytes` page-aligned bytes; the values are created here, before any use. */
		std::uninitialized_value_construct_n(host_, Count);
	}

	hugepage::Region region_;
	decltype(Api::mem_host_unregister) unregister_;
	T* host_;
	CUdeviceptr device_;
};

/**
 * Host bytes the driver has pinned, read-only, so a copy can read them straight over pcie: a baked image's file
 * mapping on its way to a card, say. The owner is held for the pin's life and handed back by `unpin`.
 */
export template<HostBytes Owner>
class PinnedSource {
public:
	PinnedSource(PinnedSource const&) = delete;
	auto operator=(PinnedSource const&) -> PinnedSource& = delete;

	PinnedSource(PinnedSource&& other) noexcept
	    : owner_{std::move(other.owner_)}, unregister_{std::exchange(other.unregister_, {})}, base_{std::exchange(other.base_, nullptr)} {}

	auto operator=(PinnedSource&& other) noexcept -> PinnedSource& {
		std::swap(owner_, other.owner_);
		std::swap(unregister_, other.unregister_);
		std::swap(base_, other.base_);
		return *this;
	}

	~PinnedSource() {
		release();
	}

	/** The pinned bytes a copy reads. */
	[[nodiscard]] auto pinned() const -> std::span<std::byte const> {
		return owner_.bytes();
	}

	/** Unpins the bytes and hands the owner back. */
	[[nodiscard]] auto unpin() && -> Owner {
		release();
		return std::move(owner_);
	}

private:
	friend class Context;

	explicit PinnedSource(Owner owner) : owner_{std::move(owner)}, unregister_{}, base_{nullptr} {}

	auto release() -> void {
		// A moved-from, unpinned or never-pinned source holds no pin; a failed unpin at teardown has no remedy.
		if (unregister_.function != nullptr) {
			core::invariant(static_cast<void const*>(owner_.bytes().data()) == base_);
			std::ignore = std::exchange(unregister_, {})(base_);
		}
	}

	Owner owner_;
	decltype(Api::mem_host_unregister) unregister_;
	void* base_;
};

}
