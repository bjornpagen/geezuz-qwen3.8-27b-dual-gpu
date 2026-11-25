// SPDX-License-Identifier: AGPL-3.0-only
module;

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

export module engine.net:fd;

import std;

namespace engine::net {

/** The one owner of a kernel descriptor; the number never leaves this module. */
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

	/** False only for a failed open or a moved-from owner. */
	[[nodiscard]] auto valid() const noexcept -> bool {
		return value_ >= 0;
	}

private:
	auto reset() noexcept -> void {
		if (value_ >= 0) {
			std::ignore = ::close(std::exchange(value_, -1));
		}
	}

	int value_;
};

/** Adds O_NONBLOCK to `fd`'s status flags; zero or the errno of the refusal. */
[[nodiscard]] auto set_nonblocking(int fd) -> int {
	auto const flags = ::fcntl(fd, F_GETFL, 0);
	if (flags < 0) {
		return errno;
	}
	// The status flags are a bit set the fcntl(2) ABI spells as an int.
	auto const added = static_cast<unsigned>(flags) | static_cast<unsigned>(O_NONBLOCK);
	if (::fcntl(fd, F_SETFL, static_cast<int>(added)) < 0) {
		return errno;
	}
	return 0;
}

}
