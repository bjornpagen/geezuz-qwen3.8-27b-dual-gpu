// SPDX-License-Identifier: AGPL-3.0-only
export module engine.driver:context;

import std;
import engine.core;
import engine.ptx;
import :abi;
import :error;
import :memory;
import :stream;
import :program;

namespace engine::driver {

/** How many bytes of the jit's error log a refused image keeps. */
inline constexpr std::size_t jit_log_bytes = std::size_t{16} * 1024;

/**
 * One card's primary context, retained for the owner's lifetime, and the only source of everything that lives in it:
 * device and mapped memory, streams, events, programs. Every such owner must die before the context it came from,
 * because releasing the last retain destroys the context and everything still in it; the owner that holds a context
 * declares it before what it creates from it.
 *
 * The driver acts on the calling thread's current context, so every call that creates or launches makes this one
 * current first: which card a call reaches is decided here, never by what another call left behind.
 */
export class Context {
public:
	[[nodiscard]] static auto retain(Api const& api, CUdevice ordinal) -> core::Result<Context, DeviceError> {
		auto handle = CUcontext{};
		return call(api.primary_ctx_retain, &handle, ordinal).transform([&] {
			return Context{api, ordinal, handle};
		});
	}

	Context(Context const&) = delete;
	auto operator=(Context const&) -> Context& = delete;

	Context(Context&& other) noexcept
	    : api_{std::exchange(other.api_, Api{})}, ordinal_{other.ordinal_}, handle_{std::exchange(other.handle_, nullptr)} {}

	auto operator=(Context&& other) noexcept -> Context& {
		std::swap(api_, other.api_);
		std::swap(ordinal_, other.ordinal_);
		std::swap(handle_, other.handle_);
		return *this;
	}

	~Context() {
		// A moved-from context holds nothing; a failed release at teardown has no remedy.
		if (api_.primary_ctx_release.function != nullptr) {
			std::ignore = api_.primary_ctx_release(ordinal_);
		}
	}

	[[nodiscard]] auto ordinal() const -> CUdevice {
		return ordinal_;
	}

	/** Maps `peer`'s memory into this card's address space. */
	[[nodiscard]] auto enable_peer_access(Context const& peer) const -> core::Result<void, DeviceError> {
		return make_current().and_then([&] {
			return call(api_.ctx_enable_peer_access, peer.handle_, 0U);
		});
	}

	template<Wire T, std::size_t Count>
	[[nodiscard]] auto allocate() const -> core::Result<DeviceArray<T, Count>, DeviceError> {
		auto address = CUdeviceptr{};
		return make_current()
		    .and_then([&] {
			    return call(api_.mem_alloc, &address, DeviceArray<T, Count>::bytes);
		    })
		    .transform([&] {
			    return DeviceArray<T, Count>{api_.mem_free, address};
		    });
	}

	template<Wire T, std::size_t Count>
	[[nodiscard]] auto allocate_mapped() const -> core::Result<MappedArray<T, Count>, DeviceError> {
		static_assert(alignof(T) <= 4096, "pinned memory is page-aligned");
		auto* host = static_cast<void*>(nullptr);
		auto made = make_current()
		                .and_then([&] {
			                return call(api_.mem_host_alloc, &host, MappedArray<T, Count>::bytes, host_alloc_device_map);
		                })
		                .transform([&] {
			                return MappedArray<T, Count>{api_.mem_free_host, host};
		                });
		if (!made) {
			return made;
		}
		return call(api_.mem_host_get_device_pointer, &made->device_, host, 0U).transform([&] {
			return std::move(*made);
		});
	}

	[[nodiscard]] auto create_stream() const -> core::Result<Stream, DeviceError> {
		auto handle = CUstream{};
		return make_current()
		    .and_then([&] {
			    return call(api_.stream_create, &handle, stream_non_blocking);
		    })
		    .transform([&] {
			    return Stream{api_, ordinal_, handle};
		    });
	}

	[[nodiscard]] auto create_event() const -> core::Result<Event, DeviceError> {
		auto handle = CUevent{};
		return make_current()
		    .and_then([&] {
			    return call(api_.event_create, &handle, event_disable_timing);
		    })
		    .transform([&] {
			    return Event{api_, ordinal_, handle};
		    });
	}

	/**
	 * Jit-loads the kernels' image and looks up every kernel in it, checking that each entry takes exactly the one
	 * parameter struct its declaration launches it with: a host/device signature drift fails here, not as a wrong read
	 * on the card.
	 */
	template<Kernel... Ks>
	[[nodiscard]] auto load() const -> core::Result<Program<Ks...>, DeviceError> {
		if (auto current = make_current(); !current) {
			return std::unexpected(current.error());
		}
		auto log = std::string(jit_log_bytes, '\0');
		auto options = std::array{JitOption::ErrorLogBuffer, JitOption::ErrorLogBufferSizeBytes};
		// The jit reads the log size option's value as an integer carried in the pointer-sized slot.
		auto values = std::array<void*, 2>{log.data(), std::bit_cast<void*>(std::uintptr_t{log.size()})};
		auto module = CUmodule{};
		auto const text = Program<Ks...>::image.text;
		auto const status =
		    api_.module_load_data_ex(&module, text.data(), core::narrow<unsigned int>(options.size()), options.data(), values.data());
		if (status != Status::Success) {
			log.resize(std::min(log.find('\0'), log.size()));
			return std::unexpected(DeviceError{.reason = ModuleRejected{.status = status, .log = std::move(log)}});
		}
		auto program = Program<Ks...>{api_, ordinal_, module};
		auto found = core::Result<void, DeviceError>{};
		auto const look_up = [&]<Kernel K>() {
			found = found.and_then([&] {
				return find<K>(program);
			});
		};
		(look_up.template operator()<Ks>(), ...);
		return found.transform([&] {
			return std::move(program);
		});
	}

	/**
	 * The typed launch primitive every kernel's own launch function is built from: `K`'s geometry and shared memory
	 * come from its declaration, and `params` is copied into the launch, so it may die when this returns.
	 */
	template<Kernel K, Kernel... Ks>
	[[nodiscard]] auto launch(Program<Ks...> const& program, Stream const& stream, typename K::Params const& params) const
	    -> core::Result<void, DeviceError> {
		core::invariant(program.ordinal_ == ordinal_ && stream.ordinal_ == ordinal_);
		auto argument = params;
		auto arguments = std::array<void*, 1>{&argument};
		auto const function = program.functions_[Program<Ks...>::template position<K>()];
		return make_current().and_then([&] {
			if constexpr (K::residency == Residency::CoResident) {
				return call(api_.launch_cooperative_kernel, function, K::grid.x, K::grid.y, K::grid.z, K::block.x, K::block.y, K::block.z,
				            std::uint32_t{K::shared_bytes}, stream.handle_, arguments.data());
			} else {
				return call(api_.launch_kernel, function, K::grid.x, K::grid.y, K::grid.z, K::block.x, K::block.y, K::block.z,
				            std::uint32_t{K::shared_bytes}, stream.handle_, arguments.data(), static_cast<void**>(nullptr));
			}
		});
	}

private:
	Context(Api const& api, CUdevice ordinal, CUcontext handle) : api_{api}, ordinal_{ordinal}, handle_{handle} {}

	[[nodiscard]] auto make_current() const -> core::Result<void, DeviceError> {
		return call(api_.ctx_set_current, handle_);
	}

	/** A parameter's byte size, or nullopt where the entry has no parameter at that index. */
	[[nodiscard]] auto parameter_bytes(CUfunction function, std::size_t index) const
	    -> core::Result<std::optional<std::size_t>, DeviceError> {
		auto offset = std::size_t{0};
		auto size = std::size_t{0};
		// The driver answers an index past the last parameter with "invalid value": the answer, not a failure.
		if (auto const status = api_.func_get_param_info(function, index, &offset, &size); status == Status::InvalidValue) {
			return std::optional<std::size_t>{};
		} else if (status != Status::Success) {
			return std::unexpected(DeviceError{.reason = Failure{.symbol = api_.func_get_param_info.symbol, .status = status}});
		}
		return std::optional{size};
	}

	template<Kernel K, Kernel... Ks>
	[[nodiscard]] auto find(Program<Ks...>& program) const -> core::Result<void, DeviceError> {
		auto function = CUfunction{};
		// A missing entry is the image not holding the kernel the host declared: its own refusal, not a driver fault.
		if (auto const status = api_.module_get_function(&function, program.module_, std::string_view{K::entry}.data());
		    status == Status::NotFound) {
			return std::unexpected(DeviceError{.reason = KernelMissing{.entry = K::entry}});
		} else if (status != Status::Success) {
			return std::unexpected(DeviceError{.reason = Failure{.symbol = api_.module_get_function.symbol, .status = status}});
		}
		auto const first = parameter_bytes(function, 0);
		auto const second = first.and_then([&](auto) {
			return parameter_bytes(function, 1);
		});
		if (!second) {
			return std::unexpected(second.error());
		}
		if (*first != sizeof(typename K::Params) || second->has_value()) {
			return std::unexpected(DeviceError{.reason = KernelSignature{
			                                       .entry = K::entry,
			                                       .expected = sizeof(typename K::Params),
			                                       .first = *first,
			                                       .second = *second,
			                                   }});
		}
		return call(api_.func_set_attribute, function, FunctionAttribute::MaxDynamicSharedSizeBytes, core::narrow<int>(K::shared_bytes))
		    .transform([&] {
			    program.functions_[Program<Ks...>::template position<K>()] = function;
		    });
	}

	Api api_;
	CUdevice ordinal_;
	CUcontext handle_;
};

}
