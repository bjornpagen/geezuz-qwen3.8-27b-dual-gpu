// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver:memory;

import std;
import :abi;

namespace engine::driver {

/** A value that crosses to the device as bytes: the device reads it with the same layout the host wrote. */
export template<class T>
concept Wire = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

export class Context;

/** Device memory holding `Count` values of `T`, freed when its owner dies. Only a context allocates it. */
export template<Wire T, std::size_t Count>
class DeviceArray {
public:
	static_assert(Count > 0, "device memory is planned at build time; an empty array is a planning error");
	static constexpr std::size_t bytes = sizeof(T) * Count;

	DeviceArray(DeviceArray const&) = delete;
	auto operator=(DeviceArray const&) -> DeviceArray& = delete;

	DeviceArray(DeviceArray&& other) noexcept : free_{std::exchange(other.free_, {})}, address_{std::exchange(other.address_, 0)} {}

	auto operator=(DeviceArray&& other) noexcept -> DeviceArray& {
		std::swap(free_, other.free_);
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

private:
	friend class Context;

	DeviceArray(decltype(Api::mem_free) free, CUdeviceptr address) : free_{free}, address_{address} {}

	decltype(Api::mem_free) free_;
	CUdeviceptr address_;
};

/**
 * Pinned host memory holding `Count` values of `T`, mapped into one card's address space, freed when its owner dies.
 * The values start value-initialized (zero for every `Wire` aggregate). Host and device both reach it: what each side
 * may touch when is the protocol of whatever lives in it, never this owner's.
 */
export template<Wire T, std::size_t Count>
class MappedArray {
public:
	static_assert(Count > 0, "pinned memory is planned at build time; an empty array is a planning error");
	static constexpr std::size_t bytes = sizeof(T) * Count;

	MappedArray(MappedArray const&) = delete;
	auto operator=(MappedArray const&) -> MappedArray& = delete;

	MappedArray(MappedArray&& other) noexcept
	    : free_{std::exchange(other.free_, {})}, host_{std::exchange(other.host_, nullptr)}, device_{std::exchange(other.device_, 0)} {}

	auto operator=(MappedArray&& other) noexcept -> MappedArray& {
		std::swap(free_, other.free_);
		std::swap(host_, other.host_);
		std::swap(device_, other.device_);
		return *this;
	}

	~MappedArray() {
		// As for device memory: a moved-from owner holds nothing, and a failed free at teardown has no remedy.
		if (free_.function != nullptr) {
			std::ignore = free_(host_);
		}
	}

	[[nodiscard]] auto host() -> std::span<T, Count> {
		return std::span<T, Count>{host_, Count};
	}

	[[nodiscard]] auto host() const -> std::span<T const, Count> {
		return std::span<T const, Count>{host_, Count};
	}

	/** The address the card reaches the same memory at, as kernel parameters carry it. */
	[[nodiscard]] auto device_address() const -> CUdeviceptr {
		return device_;
	}

private:
	friend class Context;

	MappedArray(decltype(Api::mem_free_host) free, void* host) : free_{free}, host_{static_cast<T*>(host)}, device_{0} {
		/* SAFETY: the driver returned `bytes` page-aligned bytes; the values are created here, before any use. */
		std::uninitialized_value_construct_n(host_, Count);
	}

	decltype(Api::mem_free_host) free_;
	T* host_;
	CUdeviceptr device_;
};

}
