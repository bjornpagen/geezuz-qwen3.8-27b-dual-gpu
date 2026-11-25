// SPDX-License-Identifier: AGPL-3.0-only
module;

// The darwin wait backend: kqueue with one-shot socket filters, a user event for cross-thread wakes and signal
// filters for SIGINT/SIGTERM; plus the socket calls whose spelling is darwin's (SO_NOSIGPIPE, no accept4).
#include <cerrno>
#include <csignal>
#include <ctime>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <unistd.h>

export module engine.net:wait;

import std;
import engine.core;
import :error;
import :fact;
import :fd;

namespace engine::net {

using SignalAction = void (*)(int);

struct SavedSignals {
	SignalAction interrupt;
	SignalAction terminate;
};

// The wake event's identifier in the user-event filter's namespace, which no descriptor shares.
constexpr std::uint64_t wake_ident = 1;

[[nodiscard]] auto change(std::uint64_t ident, std::int16_t filter, std::uint16_t flags, std::uint32_t fflags, std::uint64_t token)
    -> ::kevent64_s {
	return ::kevent64_s{.ident = ident, .filter = filter, .flags = flags, .fflags = fflags, .data = 0, .udata = token, .ext = {0, 0}};
}

/** Applies `changes` to the queue; returns zero or the errno of the refusal. */
[[nodiscard]] auto submit(int queue, std::span<::kevent64_s const> changes) -> int {
	if (::kevent64(queue, changes.data(), core::narrow<int>(changes.size()), nullptr, 0, 0, nullptr) < 0) {
		return errno;
	}
	return 0;
}

[[nodiscard]] auto timeout_until(Deadline deadline) -> ::timespec {
	auto const remaining = std::max(deadline - Clock::now(), Clock::duration::zero());
	auto const seconds = std::chrono::floor<std::chrono::seconds>(remaining);
	auto const nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(remaining - seconds);
	return ::timespec{.tv_sec = core::narrow<::time_t>(seconds.count()), .tv_nsec = core::narrow<long>(nanoseconds.count())};
}

/** Wakes the queue whose wake target is `target`; callable from any thread while the queue lives. */
auto wake(int target) noexcept -> void {
	auto const trigger = std::array{change(wake_ident, EVFILT_USER, 0, NOTE_TRIGGER, 0)};
	std::ignore = submit(target, trigger);
}

class Queue {
public:
	[[nodiscard]] static auto open() -> core::Result<Queue, NetError> {
		auto queue = Fd{::kqueue()};
		if (!queue.valid()) {
			return std::unexpected(failure(Stage::Queue, errno));
		}
		if (::fcntl(queue.get(), F_SETFD, FD_CLOEXEC) < 0) {
			return std::unexpected(failure(Stage::Queue, errno));
		}
		// EV_CLEAR resets the event when it is reported, so triggers between two waits coalesce into one fact.
		auto const user = std::array{change(wake_ident, EVFILT_USER, EV_ADD | EV_CLEAR, 0, 0)};
		if (auto const code = submit(queue.get(), user); code != 0) {
			return std::unexpected(failure(Stage::Queue, code));
		}
		return Queue{std::move(queue)};
	}

	Queue(Queue&& other) noexcept : queue_{std::move(other.queue_)}, saved_{std::exchange(other.saved_, std::nullopt)} {}

	auto operator=(Queue&&) -> Queue& = delete;
	Queue(Queue const&) = delete;
	auto operator=(Queue const&) -> Queue& = delete;

	~Queue() {
		if (saved_) {
			std::ignore = std::signal(SIGINT, saved_->interrupt);
			std::ignore = std::signal(SIGTERM, saved_->terminate);
		}
	}

	[[nodiscard]] auto watch(int fd, std::uint64_t token, Interest interest) -> core::Result<void, NetError> {
		auto const ident = core::narrow<std::uint64_t>(fd);
		auto const read = change(ident, EVFILT_READ, EV_ADD | EV_ONESHOT, 0, token);
		auto const write = change(ident, EVFILT_WRITE, EV_ADD | EV_ONESHOT, 0, token);
		auto code = 0;
		switch (interest) {
		case Interest::Read:
			code = submit(queue_.get(), std::array{read});
			break;
		case Interest::Write:
			code = submit(queue_.get(), std::array{write});
			break;
		case Interest::ReadWrite:
			code = submit(queue_.get(), std::array{read, write});
			break;
		}
		if (code != 0) {
			return std::unexpected(failure(Stage::Watch, code));
		}
		return {};
	}

	/** SIGINT and SIGTERM become stop facts until the queue is destroyed, which restores their dispositions. */
	[[nodiscard]] auto subscribe_stop() -> core::Result<void, NetError> {
		if (saved_) {
			return {};
		}
		/* SAFETY: darwin's EVFILT_SIGNAL records every delivery attempt even under SIG_IGN, which only stops the default action */
		auto const interrupt = std::signal(SIGINT, SIG_IGN);
		if (interrupt == SIG_ERR) {
			return std::unexpected(failure(Stage::Signal, errno));
		}
		auto const terminate = std::signal(SIGTERM, SIG_IGN);
		if (terminate == SIG_ERR) {
			auto const code = errno;
			std::ignore = std::signal(SIGINT, interrupt);
			return std::unexpected(failure(Stage::Signal, code));
		}
		saved_ = SavedSignals{.interrupt = interrupt, .terminate = terminate};
		auto const filters = std::array{
		    change(core::narrow<std::uint64_t>(SIGINT), EVFILT_SIGNAL, EV_ADD | EV_CLEAR, 0, 0),
		    change(core::narrow<std::uint64_t>(SIGTERM), EVFILT_SIGNAL, EV_ADD | EV_CLEAR, 0, 0),
		};
		if (auto const code = submit(queue_.get(), filters); code != 0) {
			return std::unexpected(failure(Stage::Signal, code));
		}
		return {};
	}

	[[nodiscard]] auto wait_until(std::optional<Deadline> deadline) -> core::Result<Facts, NetError> {
		auto events = std::array<::kevent64_s, max_facts>{};
		auto count = 0;
		do {
			auto const timeout = deadline.transform(timeout_until);
			count = ::kevent64(queue_.get(), nullptr, 0, events.data(), core::narrow<int>(events.size()), 0, timeout ? &*timeout : nullptr);
		} while (count < 0 && errno == EINTR);
		if (count < 0) {
			return std::unexpected(failure(Stage::Wait, errno));
		}
		auto facts = Facts{.items = {}, .count = 0};
		for (auto const& event : std::span{events}.first(core::narrow<std::size_t>(count))) {
			facts.items[facts.count] = translate(event);
			++facts.count;
		}
		return facts;
	}

	[[nodiscard]] auto wake_target() const -> int {
		return queue_.get();
	}

private:
	explicit Queue(Fd queue) : queue_{std::move(queue)}, saved_{std::nullopt} {}

	[[nodiscard]] static auto translate(::kevent64_s const& event) -> Fact {
		switch (event.filter) {
		case EVFILT_USER:
			return Woken{};
		case EVFILT_SIGNAL:
			return StopRequested{};
		case EVFILT_WRITE:
			return Ready{.token = event.udata, .direction = Direction::Writable};
		// Only the filters this queue registered can be reported; EVFILT_READ, with or without EV_EOF, is the rest.
		default:
			return Ready{.token = event.udata, .direction = Direction::Readable};
		}
	}

	Fd queue_;
	std::optional<SavedSignals> saved_;
};

/** A new close-on-exec stream socket, or -1 with errno set. */
[[nodiscard]] auto open_stream_socket() -> int {
	auto const fd = ::socket(AF_INET, SOCK_STREAM, 0);
	if (fd >= 0 && ::fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
		auto const code = errno;
		std::ignore = ::close(fd);
		errno = code;
		return -1;
	}
	return fd;
}

/** The next pending connection, or -1 with errno set; `prepare_connection` configures it. */
[[nodiscard]] auto accept_from(int listener) -> int {
	return ::accept(listener, nullptr, nullptr);
}

/** Makes a connected socket nonblocking, close-on-exec, free of SIGPIPE and of Nagle's delay; zero or an errno. */
[[nodiscard]] auto prepare_connection(int fd) -> int {
	if (auto const code = set_nonblocking(fd); code != 0) {
		return code;
	}
	if (::fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
		return errno;
	}
	auto const one = 1;
	if (::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one) < 0) {
		return errno;
	}
	// Token streams are many small writes: Nagle's algorithm would hold each one for the previous one's ack.
	if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one) < 0) {
		return errno;
	}
	return 0;
}

/** Bytes written, or -1 with errno set; never raises SIGPIPE (the socket carries SO_NOSIGPIPE). */
[[nodiscard]] auto send_some(int fd, std::span<char const> bytes) -> std::ptrdiff_t {
	return ::send(fd, bytes.data(), bytes.size(), 0);
}

}
