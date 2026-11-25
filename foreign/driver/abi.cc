// SPDX-License-Identifier: AGPL-3.0-only
module;

// offsetof is a macro: the module's layout checks need it.
#include <cstddef>

export module engine.driver:abi;

import std;

// The subset of the CUDA driver API and of NVML the engine calls, declared here from the two vendors' API references:
// no CUDA SDK exists at build time and no vendor header is included. Each entry names the exact exported symbol whose
// ABI its signature matches, versioned where the vendor versions it (`cuMemAlloc_v2`, not the `cuMemAlloc` macro of
// cuda.h), and the loader resolves exactly those names. Enumerators carry the vendor's numeric values.

namespace engine::driver {

/**
 * An exported C symbol's name as a template argument, so that an entry's name and its signature are one fact. The
 * array keeps the literal's NUL, which the loader hands to `dlsym`.
 */
export template<std::size_t Size>
struct Symbol {
	std::array<char, Size> text{};

	consteval Symbol(char const (&name)[Size]) {
		std::ranges::copy(name, text.begin());
	}

	[[nodiscard]] constexpr auto view() const -> std::string_view {
		return {text.data(), Size - 1};
	}
};

/** One entry point of a vendor library: its symbol and the function it resolved to (null until resolved). */
export template<Symbol Name, class Signature>
struct Entry;

export template<Symbol Name, class Return, class... Parameters>
struct Entry<Name, Return(Parameters...)> {
	static constexpr std::string_view symbol = Name.view();

	Return (*function)(Parameters...) = nullptr;

	[[nodiscard]] auto operator()(Parameters... arguments) const -> Return {
		return function(arguments...);
	}
};

export struct CUctx_st;
export struct CUmod_st;
export struct CUfunc_st;
export struct CUstream_st;
export struct CUevent_st;
export struct nvmlDevice_st;

export using CUdevice = int;
export using CUdeviceptr = std::uint64_t;
export using CUcontext = CUctx_st*;
export using CUmodule = CUmod_st*;
export using CUfunction = CUfunc_st*;
export using CUstream = CUstream_st*;
export using CUevent = CUevent_st*;
export using nvmlDevice_t = nvmlDevice_st*;

/* PIN(driver-abi-enum-size): the vendors' C enums are int-sized; these must match that width, not their value set */
// NOLINTBEGIN(performance-enum-size)

/**
 * `CUresult`: cuda.h's `CUDA_ERROR_*` values in camel case. The driver may return a value not named here; the fixed
 * underlying type holds it, and it renders as its number.
 */
export enum class Status : int {
	Success = 0,
	InvalidValue = 1,
	OutOfMemory = 2,
	NotInitialized = 3,
	InsufficientDriver = 35,
	NoDevice = 100,
	InvalidDevice = 101,
	InvalidImage = 200,
	InvalidContext = 201,
	PeerAccessUnsupported = 217,
	InvalidPtx = 218,
	UnsupportedPtxVersion = 222,
	InvalidHandle = 400,
	NotFound = 500,
	NotReady = 600,
	LaunchOutOfResources = 701,
	PeerAccessAlreadyEnabled = 704,
	HostMemoryAlreadyRegistered = 712,
	HostMemoryNotRegistered = 713,
	ContextIsDestroyed = 709,
	LaunchFailed = 719,
	CooperativeLaunchTooLarge = 720,
	Unknown = 999,
};

/** `CUdevice_attribute`, the attributes the engine reads. */
export enum class Attribute : int {
	MultiprocessorCount = 16,
	PciBusId = 33,
	PciDeviceId = 34,
	PciDomainId = 50,
	ComputeCapabilityMajor = 75,
	ComputeCapabilityMinor = 76,
};

/** `CUjit_option`, the options the engine passes to the jit. */
export enum class JitOption : int {
	ErrorLogBuffer = 5,
	ErrorLogBufferSizeBytes = 6,
};

/** `CUfunction_attribute`, the attribute the engine sets. */
export enum class FunctionAttribute : int {
	MaxDynamicSharedSizeBytes = 8,
};

/** Flag values of `cuStreamCreate`, `cuEventCreate` and `cuMemHostRegister_v2`. */
export inline constexpr unsigned int stream_non_blocking = 0x1;
export inline constexpr unsigned int event_disable_timing = 0x2;
export inline constexpr unsigned int host_register_portable = 0x1;
export inline constexpr unsigned int host_register_device_map = 0x2;
export inline constexpr unsigned int host_register_read_only = 0x8;

/** `nvmlReturn_t`: nvml.h's `NVML_ERROR_*` values in camel case; like `Status`, open to values not named here. */
export enum class ManagementStatus : int {
	Success = 0,
	Uninitialized = 1,
	InvalidArgument = 2,
	NotSupported = 3,
	NoPermission = 4,
	NotFound = 6,
	DriverNotLoaded = 9,
	GpuIsLost = 15,
	Unknown = 999,
};

/** `nvmlEnableState_t`. */
export enum class EnableState : int {
	Disabled = 0,
	Enabled = 1,
};

/** `nvmlClockType_t`. */
export enum class ClockType : int {
	Graphics = 0,
	Sm = 1,
	Memory = 2,
	Video = 3,
};

// NOLINTEND(performance-enum-size)

/** `nvmlPciInfo_t`, the layout `nvmlDeviceGetNvLinkRemotePciInfo_v2` fills. */
export struct PciInfo {
	std::array<char, 16> bus_id_legacy;
	std::uint32_t domain;
	std::uint32_t bus;
	std::uint32_t device;
	std::uint32_t pci_device_id;
	std::uint32_t pci_subsystem_id;
	std::array<char, 32> bus_id;
};

static_assert(sizeof(PciInfo) == 68 && offsetof(PciInfo, bus_id) == 36, "nvmlPciInfo_t's layout since nvml's v2 pci info");

/** A card's place on the pci bus: the identity both libraries agree on. GPUs are always function 0. */
export struct PciAddress {
	std::uint32_t domain;
	std::uint32_t bus;
	std::uint32_t device;

	[[nodiscard]] constexpr auto operator==(PciAddress const&) const -> bool = default;
};

/** The CUDA driver entry points, filled by the loader in the product and by the test double in tests. */
export struct Api {
	Entry<"cuInit", Status(unsigned int)> init;
	Entry<"cuDriverGetVersion", Status(int*)> driver_get_version;
	Entry<"cuDeviceGetCount", Status(int*)> device_get_count;
	Entry<"cuDeviceGet", Status(CUdevice*, int)> device_get;
	Entry<"cuDeviceGetAttribute", Status(int*, Attribute, CUdevice)> device_get_attribute;
	Entry<"cuDeviceTotalMem_v2", Status(std::size_t*, CUdevice)> device_total_mem;
	Entry<"cuDeviceCanAccessPeer", Status(int*, CUdevice, CUdevice)> device_can_access_peer;
	Entry<"cuDevicePrimaryCtxRetain", Status(CUcontext*, CUdevice)> primary_ctx_retain;
	Entry<"cuDevicePrimaryCtxRelease_v2", Status(CUdevice)> primary_ctx_release;
	Entry<"cuCtxSetCurrent", Status(CUcontext)> ctx_set_current;
	Entry<"cuCtxEnablePeerAccess", Status(CUcontext, unsigned int)> ctx_enable_peer_access;
	Entry<"cuMemAlloc_v2", Status(CUdeviceptr*, std::size_t)> mem_alloc;
	Entry<"cuMemFree_v2", Status(CUdeviceptr)> mem_free;
	Entry<"cuMemHostRegister_v2", Status(void*, std::size_t, unsigned int)> mem_host_register;
	Entry<"cuMemHostUnregister", Status(void*)> mem_host_unregister;
	Entry<"cuMemHostGetDevicePointer_v2", Status(CUdeviceptr*, void*, unsigned int)> mem_host_get_device_pointer;
	Entry<"cuMemcpyHtoDAsync_v2", Status(CUdeviceptr, void const*, std::size_t, CUstream)> memcpy_htod_async;
	Entry<"cuMemcpyDtoHAsync_v2", Status(void*, CUdeviceptr, std::size_t, CUstream)> memcpy_dtoh_async;
	Entry<"cuMemcpyPeerAsync", Status(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, std::size_t, CUstream)> memcpy_peer_async;
	Entry<"cuStreamCreate", Status(CUstream*, unsigned int)> stream_create;
	Entry<"cuStreamDestroy_v2", Status(CUstream)> stream_destroy;
	Entry<"cuStreamWaitEvent", Status(CUstream, CUevent, unsigned int)> stream_wait_event;
	Entry<"cuEventCreate", Status(CUevent*, unsigned int)> event_create;
	Entry<"cuEventDestroy_v2", Status(CUevent)> event_destroy;
	Entry<"cuEventRecord", Status(CUevent, CUstream)> event_record;
	Entry<"cuEventQuery", Status(CUevent)> event_query;
	Entry<"cuEventElapsedTime", Status(float*, CUevent, CUevent)> event_elapsed_time;
	Entry<"cuModuleLoadDataEx", Status(CUmodule*, void const*, unsigned int, JitOption*, void**)> module_load_data_ex;
	Entry<"cuModuleUnload", Status(CUmodule)> module_unload;
	Entry<"cuModuleGetFunction", Status(CUfunction*, CUmodule, char const*)> module_get_function;
	Entry<"cuFuncGetParamInfo", Status(CUfunction, std::size_t, std::size_t*, std::size_t*)> func_get_param_info;
	Entry<"cuFuncSetAttribute", Status(CUfunction, FunctionAttribute, int)> func_set_attribute;
	Entry<"cuLaunchKernel", Status(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
	                               unsigned int, CUstream, void**, void**)>
	    launch_kernel;
	Entry<"cuLaunchCooperativeKernel", Status(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
	                                          unsigned int, unsigned int, CUstream, void**)>
	    launch_cooperative_kernel;
};

/** The NVML entry points: the only source that can tell an nvlink peer path from a pcie one. */
export struct Management {
	Entry<"nvmlInit_v2", ManagementStatus()> init;
	Entry<"nvmlShutdown", ManagementStatus()> shutdown;
	Entry<"nvmlDeviceGetHandleByPciBusId_v2", ManagementStatus(char const*, nvmlDevice_t*)> device_by_pci_bus_id;
	Entry<"nvmlDeviceGetNvLinkState", ManagementStatus(nvmlDevice_t, unsigned int, EnableState*)> nvlink_state;
	Entry<"nvmlDeviceGetNvLinkRemotePciInfo_v2", ManagementStatus(nvmlDevice_t, unsigned int, PciInfo*)> nvlink_remote_pci_info;
	Entry<"nvmlDeviceGetClockInfo", ManagementStatus(nvmlDevice_t, ClockType, unsigned int*)> clock_info;
	Entry<"nvmlDeviceGetPerformanceState", ManagementStatus(nvmlDevice_t, int*)> performance_state;
};

/** Both vendor tables: what the engine's device layer is built over. */
export struct Driver {
	Api cuda;
	Management management;
};

}
