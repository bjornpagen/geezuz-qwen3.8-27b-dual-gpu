// SPDX-License-Identifier: AGPL-3.0-only
module;

// The linux wait backend: epoll with one-shot socket registrations, an eventfd for cross-thread wakes and a signalfd
// for SIGINT/SIGTERM; plus the socket calls whose spelling is linux's (accept4, MSG_NOSIGNAL, SOCK_CLOEXEC).
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <unistd.h>

export module engine.net:wait;

import std;
import engine.core;
import :error;
import :fact;
import :fd;

namespace engine::net {

// The queue's own descriptors carry tokens at and above token_limit, which no owner token reaches.
constexpr std::uint64_t wake_token = token_limit;
constexpr std::uint64_t stop_token = token_limit + 1;

/** Registers or re-arms `fd` in the epoll set; zero or the errno of the refusal. */
[[nodiscard]] auto arm(int queue, int fd, std::uint32_t events, std::uint64_t token) -> int {
	auto event = ::epoll_event{};
	event.events = events;
	event.data.u64 = token;
	if (::epoll_ctl(queue, EPOLL_CTL_MOD, fd, &event) == 0) {
		return 0;
	}
	if (errno != ENOENT) {
		return errno;
	}
	if (::epoll_ctl(queue, EPOLL_CTL_ADD, fd, &event) < 0) {
		return errno;
	}
	return 0;
}

[[nodiscard]] auto timeout_milliseconds(std::optional<Deadline> deadline) -> int {
	if (!deadline) {
		return -1;
	}
	auto const remaining = std::max(*deadline - Clock::now(), Clock::duration::zero());
	auto const milliseconds = std::chrono::ceil<std::chrono::milliseconds>(remaining).count();
	return core::narrow<int>(std::min<std::int64_t>(milliseconds, std::numeric_limits<int>::max()));
}

/** Wakes the queue whose wake target is `target`; callable from any thread while the queue lives. */
auto wake(int target) noexcept -> void {
	auto const one = std::uint64_t{1};
	std::ignore = ::write(target, &one, sizeof one);
}

struct StopSubscription {
	Fd signals;
	::sigset_t saved_mask;
};

class Queue {
public:
	[[nodiscard]] static auto open() -> core::Result<Queue, NetError> {
		auto queue = Fd{::epoll_create1(EPOLL_CLOEXEC)};
		if (!queue.valid()) {
			return std::unexpected(failure(Stage::Queue, errno));
		}
		auto waker = Fd{::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)};
		if (!waker.valid()) {
			return std::unexpected(failure(Stage::Queue, errno));
		}
		// Level-triggered: the counter stays readable until a wait drains it, so no wake between two waits is lost.
		if (auto const code = arm(queue.get(), waker.get(), EPOLLIN, wake_token); code != 0) {
			return std::unexpected(failure(Stage::Queue, code));
		}
		return Queue{std::move(queue), std::move(waker)};
	}

	Queue(Queue&& other) noexcept
	    : queue_{std::move(other.queue_)}, waker_{std::move(other.waker_)}, stop_{std::exchange(other.stop_, std::nullopt)} {}

	auto operator=(Queue&&) -> Queue& = delete;
	Queue(Queue const&) = delete;
	auto operator=(Queue const&) -> Queue& = delete;

	~Queue() {
		if (stop_) {
			std::ignore = ::pthread_sigmask(SIG_SETMASK, &stop_->saved_mask, nullptr);
		}
	}

	[[nodiscard]] auto watch(int fd, std::uint64_t token, Interest interest) -> core::Result<void, NetError> {
		auto events = std::uint32_t{EPOLLONESHOT};
		switch (interest) {
		case Interest::Read:
			events |= EPOLLIN | EPOLLRDHUP;
			break;
		case Interest::Write:
			events |= EPOLLOUT;
			break;
		case Interest::ReadWrite:
			events |= EPOLLIN | EPOLLRDHUP | EPOLLOUT;
			break;
		}
		if (auto const code = arm(queue_.get(), fd, events, token); code != 0) {
			return std::unexpected(failure(Stage::Watch, code));
		}
		return {};
	}

	/**
	 * SIGINT and SIGTERM become stop facts until the queue is destroyed, which restores the signal mask. The mask is
	 * per thread and inherited by threads created later, so the owner subscribes before any other thread starts.
	 */
	[[nodiscard]] auto subscribe_stop() -> core::Result<void, NetError> {
		if (stop_) {
			return {};
		}
		auto mask = ::sigset_t{};
		auto saved = ::sigset_t{};
		if (::sigemptyset(&mask) < 0 || ::sigaddset(&mask, SIGINT) < 0 || ::sigaddset(&mask, SIGTERM) < 0) {
			return std::unexpected(failure(Stage::Signal, errno));
		}
		if (auto const code = ::pthread_sigmask(SIG_BLOCK, &mask, &saved); code != 0) {
			return std::unexpected(failure(Stage::Signal, code));
		}
		auto signals = Fd{::signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC)};
		if (!signals.valid()) {
			auto const code = errno;
			std::ignore = ::pthread_sigmask(SIG_SETMASK, &saved, nullptr);
			return std::unexpected(failure(Stage::Signal, code));
		}
		if (auto const code = arm(queue_.get(), signals.get(), EPOLLIN, stop_token); code != 0) {
			std::ignore = ::pthread_sigmask(SIG_SETMASK, &saved, nullptr);
			return std::unexpected(failure(Stage::Signal, code));
		}
		stop_ = StopSubscription{.signals = std::move(signals), .saved_mask = saved};
		return {};
	}

	[[nodiscard]] auto wait_until(std::optional<Deadline> deadline) -> core::Result<Facts, NetError> {
		// One socket event can yield two facts (readable and writable), so the kernel batch is half the fact capacity.
		auto events = std::array<::epoll_event, max_facts / 2>{};
		auto count = 0;
		do {
			count = ::epoll_wait(queue_.get(), events.data(), core::narrow<int>(events.size()), timeout_milliseconds(deadline));
		} while (count < 0 && errno == EINTR);
		if (count < 0) {
			return std::unexpected(failure(Stage::Wait, errno));
		}
		auto facts = Facts{.items = {}, .count = 0};
		auto const push = [&facts](Fact fact) {
			facts.items[facts.count] = fact;
			++facts.count;
		};
		for (auto const& event : std::span{events}.first(core::narrow<std::size_t>(count))) {
			auto const token = event.data.u64;
			if (token == wake_token) {
				auto counter = std::uint64_t{0};
				std::ignore = ::read(waker_.get(), &counter, sizeof counter);
				push(Woken{});
				continue;
			}
			if (token == stop_token) {
				drain_signals(facts);
				continue;
			}
			if ((event.events & (EPOLLIN | EPOLLRDHUP | EPOLLHUP | EPOLLERR)) != 0) {
				push(Ready{.token = token, .direction = Direction::Readable});
			}
			if ((event.events & (EPOLLOUT | EPOLLHUP | EPOLLERR)) != 0) {
				push(Ready{.token = token, .direction = Direction::Writable});
			}
		}
		return facts;
	}

	[[nodiscard]] auto wake_target() const -> int {
		return waker_.get();
	}

private:
	Queue(Fd queue, Fd waker) : queue_{std::move(queue)}, waker_{std::move(waker)}, stop_{std::nullopt} {}

	/** One stop fact per queued signal, as far as the batch has room, and at least one. */
	auto drain_signals(Facts& facts) -> void {
		auto info = ::signalfd_siginfo{};
		auto delivered = std::size_t{0};
		while (stop_ && ::read(stop_->signals.get(), &info, sizeof info) == static_cast<::ssize_t>(sizeof info)) {
			++delivered;
		}
		auto const room = facts.items.size() - facts.count;
		for (auto index = std::size_t{0}; index < std::min(std::max<std::size_t>(delivered, 1), room); ++index) {
			facts.items[facts.count] = StopRequested{};
			++facts.count;
		}
	}

	Fd queue_;
	Fd waker_;
	std::optional<StopSubscription> stop_;
};

/** A new close-on-exec stream socket, or -1 with errno set. */
[[nodiscard]] auto open_stream_socket() -> int {
	return ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
}

/** The next pending connection, already nonblocking and close-on-exec, or -1 with errno set. */
[[nodiscard]] auto accept_from(int listener) -> int {
	return ::accept4(listener, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
}

/** Makes a connected socket nonblocking and free of Nagle's delay; zero or an errno. */
[[nodiscard]] auto prepare_connection(int fd) -> int {
	if (auto const code = set_nonblocking(fd); code != 0) {
		return code;
	}
	// Token streams are many small writes: Nagle's algorithm would hold each one for the previous one's ack.
	auto const one = 1;
	if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one) < 0) {
		return errno;
	}
	return 0;
}

/** Bytes written, or -1 with errno set; MSG_NOSIGNAL keeps a reset peer from raising SIGPIPE. */
[[nodiscard]] auto send_some(int fd, std::span<char const> bytes) -> std::ptrdiff_t {
	return ::send(fd, bytes.data(), bytes.size(), MSG_NOSIGNAL);
}

}
