// SPDX-License-Identifier: AGPL-3.0-only
module;

// The portable berkeley socket calls; the ones whose spelling differs per platform live in the wait backend.
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

export module engine.net:socket;

import std;
import engine.core;
import :error;
import :fd;
import :wait;

namespace engine::net {

/** An IPv4 address in network order, most significant byte first. */
export using Address = std::array<std::uint8_t, 4>;

export inline constexpr Address loopback{127, 0, 0, 1};

export struct Endpoint {
	Address address;
	/** Zero asks the kernel for an ephemeral port; `Listener::port` reports the one it chose. */
	std::uint16_t port;
};

/** Some bytes moved; `count` is never zero. */
export struct Transferred {
	std::size_t count;

	[[nodiscard]] constexpr auto operator==(Transferred const&) const -> bool = default;
};

/** The kernel has no bytes (or no room) now; the owner watches the socket and retries on its fact. */
export struct WouldBlock {
	[[nodiscard]] constexpr auto operator==(WouldBlock const&) const -> bool = default;
};

/** The peer will send nothing more. */
export struct EndOfStream {
	[[nodiscard]] constexpr auto operator==(EndOfStream const&) const -> bool = default;
};

/** The connection failed (reset, timed out); `code` is the system's error number, for rendering only. */
export struct Broken {
	std::int32_t code;

	[[nodiscard]] constexpr auto operator==(Broken const&) const -> bool = default;
};

export using Received = std::variant<Transferred, WouldBlock, EndOfStream, Broken>;
export using Sent = std::variant<Transferred, WouldBlock, Broken>;

constexpr int listen_backlog = 256;

[[nodiscard]] auto socket_address(Endpoint endpoint) -> ::sockaddr_in {
	auto address = ::sockaddr_in{};
	address.sin_family = AF_INET;
	address.sin_port = htons(endpoint.port);
	address.sin_addr.s_addr = std::bit_cast<std::uint32_t>(endpoint.address);
	return address;
}

export class Poller;

/** A connected, nonblocking stream socket, owned once. */
export class Socket {
public:
	Socket(Socket&&) noexcept = default;
	auto operator=(Socket&&) noexcept -> Socket& = default;
	Socket(Socket const&) = delete;
	auto operator=(Socket const&) -> Socket& = delete;
	~Socket() = default;

	/** A client connection to `endpoint`, connected before it returns; for clients and tests of the server. */
	[[nodiscard]] static auto connect(Endpoint endpoint) -> core::Result<Socket, NetError> {
		auto fd = Fd{open_stream_socket()};
		if (!fd.valid()) {
			return std::unexpected(failure(Stage::Socket, errno));
		}
		auto const address = socket_address(endpoint);
		/* SAFETY: sockaddr_in -> sockaddr is the connect(2) ABI's aliasing rule */
		if (::connect(fd.get(), reinterpret_cast<::sockaddr const*>(&address), sizeof address) < 0) {
			return std::unexpected(failure(Stage::Connect, errno));
		}
		if (auto const code = prepare_connection(fd.get()); code != 0) {
			return std::unexpected(failure(Stage::Option, code));
		}
		return Socket{std::move(fd)};
	}

	[[nodiscard]] auto receive(std::span<char> into) -> Received {
		// A zero-byte read returns zero, which is also how the kernel spells end of stream.
		core::invariant(!into.empty());
		for (;;) {
			auto const count = ::recv(fd_.get(), into.data(), into.size(), 0);
			if (count > 0) {
				return Transferred{.count = core::narrow<std::size_t>(count)};
			}
			if (count == 0) {
				return EndOfStream{};
			}
			if (errno == EINTR) {
				continue;
			}
			if (errno == EAGAIN || errno == EWOULDBLOCK) {
				return WouldBlock{};
			}
			return Broken{.code = errno};
		}
	}

	[[nodiscard]] auto send(std::span<char const> bytes) -> Sent {
		core::invariant(!bytes.empty());
		for (;;) {
			auto const count = send_some(fd_.get(), bytes);
			if (count > 0) {
				return Transferred{.count = core::narrow<std::size_t>(count)};
			}
			if (count < 0 && errno == EINTR) {
				continue;
			}
			if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
				return WouldBlock{};
			}
			return Broken{.code = count < 0 ? errno : 0};
		}
	}

	/** Ends this side's byte stream; the peer reads end of stream after the bytes already sent. */
	auto shut_write() -> void {
		std::ignore = ::shutdown(fd_.get(), SHUT_WR);
	}

	/** Closes with a reset instead of an orderly end: what a crashed or killed peer looks like. */
	auto abort() && -> void {
		auto const linger = ::linger{.l_onoff = 1, .l_linger = 0};
		std::ignore = ::setsockopt(fd_.get(), SOL_SOCKET, SO_LINGER, &linger, sizeof linger);
		auto const closed = std::move(fd_);
	}

private:
	explicit Socket(Fd fd) : fd_{std::move(fd)} {}

	Fd fd_;

	friend class Listener;
	friend class Poller;
};

/** A nonblocking listening socket, owned once. */
export class Listener {
public:
	[[nodiscard]] static auto open(Endpoint endpoint) -> core::Result<Listener, NetError> {
		auto fd = Fd{open_stream_socket()};
		if (!fd.valid()) {
			return std::unexpected(failure(Stage::Socket, errno));
		}
		auto const one = 1;
		if (::setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof one) < 0) {
			return std::unexpected(failure(Stage::Option, errno));
		}
		auto address = socket_address(endpoint);
		/* SAFETY: sockaddr_in -> sockaddr is the bind(2) ABI's aliasing rule */
		if (::bind(fd.get(), reinterpret_cast<::sockaddr const*>(&address), sizeof address) < 0) {
			return std::unexpected(failure(Stage::Bind, errno));
		}
		if (auto const code = set_nonblocking(fd.get()); code != 0) {
			return std::unexpected(failure(Stage::Option, code));
		}
		if (::listen(fd.get(), listen_backlog) < 0) {
			return std::unexpected(failure(Stage::Listen, errno));
		}
		auto length = ::socklen_t{sizeof address};
		/* SAFETY: sockaddr_in -> sockaddr is the getsockname(2) ABI's aliasing rule */
		if (::getsockname(fd.get(), reinterpret_cast<::sockaddr*>(&address), &length) < 0) {
			return std::unexpected(failure(Stage::Resolve, errno));
		}
		return Listener{std::move(fd), ntohs(address.sin_port)};
	}

	Listener(Listener&&) noexcept = default;
	auto operator=(Listener&&) noexcept -> Listener& = default;
	Listener(Listener const&) = delete;
	auto operator=(Listener const&) -> Listener& = delete;
	~Listener() = default;

	[[nodiscard]] auto port() const -> std::uint16_t {
		return port_;
	}

	/**
	 * The next pending connection, nullopt when none is pending. A transient error (descriptors exhausted) asks the
	 * owner to stop accepting for a while; a permanent one means the listener is unusable.
	 */
	[[nodiscard]] auto accept() -> core::Result<std::optional<Socket>, NetError> {
		for (;;) {
			auto fd = Fd{accept_from(fd_.get())};
			if (!fd.valid()) {
				// The peer gave up between its connect and this accept, or a signal interrupted the call.
				if (errno == EINTR || errno == ECONNABORTED) {
					continue;
				}
				if (errno == EAGAIN || errno == EWOULDBLOCK) {
					return std::nullopt;
				}
				return std::unexpected(failure(Stage::Accept, errno));
			}
			// A connection the kernel will not configure is dropped; the listener stays healthy.
			if (prepare_connection(fd.get()) != 0) {
				continue;
			}
			return Socket{std::move(fd)};
		}
	}

private:
	Listener(Fd fd, std::uint16_t port) : fd_{std::move(fd)}, port_{port} {}

	Fd fd_;
	std::uint16_t port_;

	friend class Poller;
};

}
