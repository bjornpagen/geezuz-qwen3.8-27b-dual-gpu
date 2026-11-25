// SPDX-License-Identifier: AGPL-3.0-only
module;

// posix_spawn rather than fork: the callers are threaded, and only an exec-at-once spawn is safe there. The waits are
// poll on the output pipe and waitpid with WNOHANG, both bounded by the caller's deadline.
#include <cerrno>
#include <csignal>
#include <poll.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

export module engine.process:child;

import std;
import engine.core;
import :error;
import :spawn;

namespace engine::process {

/** The signal `terminate` sends, as a `Signalled` exit reports it. */
export inline constexpr int termination_signal = SIGTERM;

/** The longest line the child may write: a longer one fails the read rather than growing the buffer without bound. */
export inline constexpr std::size_t max_line_bytes = 4096;

/** The child returned from main or called exit, with this status. */
export struct Exited {
	int status;

	[[nodiscard]] constexpr auto operator==(Exited const&) const -> bool = default;
};

/** A signal ended the child. */
export struct Signalled {
	int signal;

	[[nodiscard]] constexpr auto operator==(Signalled const&) const -> bool = default;
};

export using Exit = std::variant<Exited, Signalled>;

/** One line of the child's output, without its newline (the last line may have had none). */
export struct Line {
	std::string text;

	[[nodiscard]] auto operator==(Line const&) const -> bool = default;
};

/** The child closed its output and every line has been read. */
export struct Closed {
	[[nodiscard]] constexpr auto operator==(Closed const&) const -> bool = default;
};

/** The deadline passed before a whole line arrived; what did arrive is kept for the next read. */
export struct Quiet {
	[[nodiscard]] constexpr auto operator==(Quiet const&) const -> bool = default;
};

export using Read = std::variant<Line, Closed, Quiet>;

/** The one owner of a descriptor; the number never leaves this module. */
class Fd {
public:
	explicit Fd(int value) noexcept : value_{value} {}

	Fd(Fd&& other) noexcept : value_{std::exchange(other.value_, -1)} {}

	auto operator=(Fd&& other) noexcept -> Fd& {
		if (this != &other) {
			reset();
			value_ = std::exchange(other.value_, -1);
		}
		return *this;
	}

	Fd(Fd const&) = delete;
	auto operator=(Fd const&) -> Fd& = delete;

	~Fd() {
		reset();
	}

	[[nodiscard]] auto get() const noexcept -> int {
		return value_;
	}

	/** False once closed or moved from. */
	[[nodiscard]] auto open() const noexcept -> bool {
		return value_ >= 0;
	}

	auto reset() noexcept -> void {
		if (value_ >= 0) {
			std::ignore = ::close(std::exchange(value_, -1));
		}
	}

private:
	int value_;
};

/** The file actions of one spawn, destroyed with the call. */
class Actions {
public:
	Actions() = default;
	Actions(Actions const&) = delete;
	auto operator=(Actions const&) -> Actions& = delete;
	Actions(Actions&&) = delete;
	auto operator=(Actions&&) -> Actions& = delete;

	~Actions() {
		if (live_) {
			std::ignore = ::posix_spawn_file_actions_destroy(&actions_);
		}
	}

	[[nodiscard]] auto init() -> int {
		auto const error = ::posix_spawn_file_actions_init(&actions_);
		live_ = error == 0;
		return error;
	}

	[[nodiscard]] auto get() -> posix_spawn_file_actions_t& {
		return actions_;
	}

private:
	posix_spawn_file_actions_t actions_{};
	bool live_ = false;
};

/** The attributes of one spawn, destroyed with the call. */
class Attributes {
public:
	Attributes() = default;
	Attributes(Attributes const&) = delete;
	auto operator=(Attributes const&) -> Attributes& = delete;
	Attributes(Attributes&&) = delete;
	auto operator=(Attributes&&) -> Attributes& = delete;

	~Attributes() {
		if (live_) {
			std::ignore = ::posix_spawnattr_destroy(&attributes_);
		}
	}

	[[nodiscard]] auto init() -> int {
		auto const error = ::posix_spawnattr_init(&attributes_);
		live_ = error == 0;
		return error;
	}

	[[nodiscard]] auto get() -> posix_spawnattr_t& {
		return attributes_;
	}

private:
	posix_spawnattr_t attributes_{};
	bool live_ = false;
};

/**
 * The child starts with no blocked signals and the default action for the ones that stop a program, whatever the
 * spawning thread blocked or ignored (a server's reactor blocks SIGINT and SIGTERM to read them as events).
 */
[[nodiscard]] auto plain_signals(posix_spawnattr_t& attributes, std::uint32_t flags) -> int {
	sigset_t none{};
	sigset_t defaults{};
	sigemptyset(&none);
	sigemptyset(&defaults);
	for (auto const signal : {SIGINT, SIGTERM, SIGHUP, SIGPIPE}) {
		sigaddset(&defaults, signal);
	}
	if (auto const error = ::posix_spawnattr_setsigmask(&attributes, &none); error != 0) {
		return error;
	}
	if (auto const error = ::posix_spawnattr_setsigdefault(&attributes, &defaults); error != 0) {
		return error;
	}
	auto const all = flags | std::uint32_t{POSIX_SPAWN_SETSIGMASK} | std::uint32_t{POSIX_SPAWN_SETSIGDEF};
	return ::posix_spawnattr_setflags(&attributes, core::narrow<short>(all));
}

[[nodiscard]] auto exit_of(int status) -> Exit {
	if (WIFEXITED(status)) {
		return Exited{.status = WEXITSTATUS(status)};
	}
	// waitpid without WUNTRACED reports only children that ended, by exit or by signal.
	core::invariant(WIFSIGNALED(status));
	return Signalled{.signal = WTERMSIG(status)};
}

/** How long a wait may sleep between checks of a child that has not ended. */
inline constexpr auto wait_interval = std::chrono::milliseconds{2};

/**
 * A child process and the read end of its standard output. Its standard input and error are this process's; no other
 * descriptor reaches it. Move-only; dropping a child that has not been reaped kills it with SIGKILL and reaps it.
 */
export class Child {
public:
	/** Spawns `program` (a path, not searched for) with `arguments` after its own name. */
	[[nodiscard]] static auto spawn(std::filesystem::path const& program, std::span<std::string const> arguments)
	    -> core::Result<Child, ProcessError> {
		auto storage = std::vector<std::string>{program.string()};
		storage.append_range(arguments);
		auto argv = std::vector<char*>{};
		argv.reserve(storage.size() + 1);
		for (auto& argument : storage) {
			argv.push_back(argument.data());
		}
		argv.push_back(nullptr);

		auto ends = open_pipe();
		if (!ends) {
			return std::unexpected(ends.error());
		}
		auto read_end = Fd{(*ends)[0]};
		auto write_end = Fd{(*ends)[1]};

		auto actions = Actions{};
		auto attributes = Attributes{};
		auto flags = std::uint32_t{0};
		auto prepared = actions.init();
		prepared = prepared != 0 ? prepared : attributes.init();
		prepared = prepared != 0 ? prepared : ::posix_spawn_file_actions_adddup2(&actions.get(), write_end.get(), STDOUT_FILENO);
		prepared = prepared != 0 ? prepared : keep_only_standard(actions.get(), flags);
		prepared = prepared != 0 ? prepared : plain_signals(attributes.get(), flags);
		if (prepared != 0) {
			return failed(Stage::Spawn, prepared);
		}
		auto pid = pid_t{0};
		/* SAFETY: argv and the environment are null-terminated arrays that outlive the call; posix_spawn copies them */
		auto const spawned = ::posix_spawn(&pid, storage.front().c_str(), &actions.get(), &attributes.get(), argv.data(), environment());
		if (spawned != 0) {
			return failed(Stage::Spawn, spawned);
		}
		// Only the child writes: the parent's copy of the write end would keep the pipe from ever reading closed.
		write_end.reset();
		return Child{pid, std::move(read_end)};
	}

	Child(Child&& other) noexcept
	    : pid_{std::exchange(other.pid_, 0)},
	      out_{std::move(other.out_)},
	      buffer_{std::move(other.buffer_)},
	      exit_{std::exchange(other.exit_, std::nullopt)} {}

	auto operator=(Child&&) -> Child& = delete;
	Child(Child const&) = delete;
	auto operator=(Child const&) -> Child& = delete;

	~Child() {
		if (pid_ <= 0 || exit_) {
			return;
		}
		std::ignore = ::kill(pid_, SIGKILL);
		auto status = 0;
		while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
		}
	}

	/** The next line of the child's output, its end, or `Quiet` once `deadline` passes without a whole line. */
	[[nodiscard]] auto read_line_until(std::chrono::steady_clock::time_point deadline) -> core::Result<Read, ProcessError> {
		for (;;) {
			auto const end = buffer_.find('\n');
			if (std::min(end, buffer_.size()) > max_line_bytes) {
				return std::unexpected(ProcessError{.stage = Stage::Read, .cause = Cause::LineTooLong, .code = 0});
			}
			if (end != std::string::npos) {
				auto line = Line{.text = buffer_.substr(0, end)};
				buffer_.erase(0, end + 1);
				return line;
			}
			if (!out_.open()) {
				if (!buffer_.empty()) {
					return Line{.text = std::exchange(buffer_, {})};
				}
				return Closed{};
			}
			auto const now = std::chrono::steady_clock::now();
			if (now >= deadline) {
				return Quiet{};
			}
			auto const left = std::chrono::ceil<std::chrono::milliseconds>(deadline - now).count();
			auto ready = ::pollfd{.fd = out_.get(), .events = POLLIN, .revents = 0};
			auto const polled = ::poll(&ready, 1, core::narrow<int>(std::min<std::int64_t>(left, std::numeric_limits<int>::max())));
			if (polled < 0 && errno != EINTR) {
				return failed(Stage::Read, errno);
			}
			if (polled <= 0) {
				continue;
			}
			auto chunk = std::array<char, max_line_bytes>{};
			auto const got = ::read(out_.get(), chunk.data(), chunk.size());
			if (got < 0) {
				if (errno == EINTR || errno == EAGAIN) {
					continue;
				}
				return failed(Stage::Read, errno);
			}
			if (got == 0) {
				out_.reset();
				continue;
			}
			buffer_.append(chunk.data(), core::narrow<std::size_t>(got));
		}
	}

	/** Asks the child to stop with SIGTERM; a child already reaped is left alone. */
	[[nodiscard]] auto terminate() -> core::Result<void, ProcessError> {
		if (exit_) {
			return {};
		}
		// Until it is reaped the child exists, if only as a zombie, so the signal cannot reach another process.
		if (::kill(pid_, SIGTERM) != 0) {
			return failed(Stage::Signal, errno);
		}
		return {};
	}

	/** How the child ended, or nothing if it is still running at `deadline`. Once reaped, the same exit every call. */
	[[nodiscard]] auto wait_until(std::chrono::steady_clock::time_point deadline) -> core::Result<std::optional<Exit>, ProcessError> {
		while (!exit_) {
			auto status = 0;
			auto const reaped = ::waitpid(pid_, &status, WNOHANG);
			if (reaped == pid_) {
				exit_ = exit_of(status);
				break;
			}
			if (reaped < 0) {
				if (errno == EINTR) {
					continue;
				}
				return failed(Stage::Wait, errno);
			}
			auto const now = std::chrono::steady_clock::now();
			if (now >= deadline) {
				return std::nullopt;
			}
			std::this_thread::sleep_until(std::min(now + wait_interval, deadline));
		}
		return exit_;
	}

private:
	Child(pid_t pid, Fd out) : pid_{pid}, out_{std::move(out)} {}

	pid_t pid_;
	Fd out_;
	/** Output read and not yet returned as a line: at most one line's bound past the last newline. */
	std::string buffer_;
	std::optional<Exit> exit_;
};

}
