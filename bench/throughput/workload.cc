// SPDX-License-Identifier: AGPL-3.0-only
export module engine.throughput:workload;

import std;
import engine.core;
import engine.net;
import engine.simdjson;
import :limits;

/**
 * What one invocation measures: how many streams run at once, and the prompts they send. A run sends every prompt
 * once, in waves of exactly that many streams, so two runs decode the same text and differ only in their timing.
 */
namespace engine::throughput {

/** How many streams a wave runs at once: the closed set the outcome table is measured at. */
export enum class Streams : std::uint8_t {
	One,
	Two,
	Four,
	Eight,
};

export inline constexpr auto stream_modes = std::array{Streams::One, Streams::Two, Streams::Four, Streams::Eight};

export [[nodiscard]] constexpr auto count(Streams streams) -> std::size_t {
	return std::size_t{1} << std::to_underlying(streams);
}

static_assert(count(stream_modes.back()) == max_streams, "the widest mode fills every slot of the engine");

/** The mode as the command line spells it: its count in decimal. */
export [[nodiscard]] constexpr auto spelling(Streams streams) -> std::string_view {
	switch (streams) {
	case Streams::One:
		return "1";
	case Streams::Two:
		return "2";
	case Streams::Four:
		return "4";
	case Streams::Eight:
		return "8";
	}
	std::unreachable();
}

/** The mode a command-line word names; nullopt for any other word. */
export [[nodiscard]] constexpr auto streams_of(std::string_view word) -> std::optional<Streams> {
	auto const found = std::ranges::find(stream_modes, word, [](Streams streams) {
		return spelling(streams);
	});
	if (found == stream_modes.end()) {
		return std::nullopt;
	}
	return *found;
}

/** The decimal number that is all of `text`, if it fits `T`. */
template<std::unsigned_integral T>
[[nodiscard]] auto number(std::string_view text) -> std::optional<T> {
	auto value = T{0};
	auto const read = std::from_chars(text.data(), std::to_address(text.end()), value);
	if (text.empty() || read.ec != std::errc{} || read.ptr != std::to_address(text.end())) {
		return std::nullopt;
	}
	return value;
}

/** The server a command-line word names: `a.b.c.d:port`, each part decimal, the port one a server can listen on. */
export [[nodiscard]] auto endpoint_of(std::string_view word) -> std::optional<net::Endpoint> {
	auto const colon = word.rfind(':');
	if (colon == std::string_view::npos) {
		return std::nullopt;
	}
	auto const port = number<std::uint16_t>(word.substr(colon + 1));
	auto address = net::Address{};
	auto octets = std::size_t{0};
	for (auto const part : std::views::split(word.substr(0, colon), '.')) {
		auto const octet = number<std::uint8_t>(std::string_view{part.begin(), part.end()});
		if (!octet || octets == address.size()) {
			return std::nullopt;
		}
		address.at(octets) = *octet;
		octets += 1;
	}
	if (!port || *port == 0 || octets != address.size()) {
		return std::nullopt;
	}
	return net::Endpoint{.address = address, .port = *port};
}

/** Why a text cannot be sent as a prompt. */
export enum class PromptFault : std::uint8_t {
	Empty,
	/** Longer than max_prompt_bytes. */
	TooLong,
	/** Not UTF-8: a JSON string cannot carry it. */
	NotUtf8,
};

export struct [[nodiscard]] PromptError {
	PromptFault fault;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(PromptError const&) const -> bool = default;
};

/** One user message a stream sends, with the name the report shows it under. Only `make` makes one, so it is sendable. */
export class Prompt {
public:
	[[nodiscard]] static auto make(std::string name, std::string text) -> core::Result<Prompt, PromptError> {
		if (text.empty()) {
			return std::unexpected(PromptError{.fault = PromptFault::Empty});
		}
		if (text.size() > max_prompt_bytes) {
			return std::unexpected(PromptError{.fault = PromptFault::TooLong});
		}
		if (!simdjson::valid_utf8(text)) {
			return std::unexpected(PromptError{.fault = PromptFault::NotUtf8});
		}
		return Prompt{std::move(name), std::move(text)};
	}

	[[nodiscard]] auto name() const -> std::string_view {
		return name_;
	}

	[[nodiscard]] auto text() const -> std::string_view {
		return text_;
	}

	[[nodiscard]] auto operator==(Prompt const&) const -> bool = default;

private:
	Prompt(std::string name, std::string text) : name_{std::move(name)}, text_{std::move(text)} {}

	std::string name_;
	std::string text_;
};

/** Why a set of prompts is no workload for a mode. */
export enum class WorkloadFault : std::uint8_t {
	NoPrompts,
	/** More than max_prompts. */
	TooManyPrompts,
	/** The prompts do not fill whole waves: the last wave would run fewer streams than the mode names. */
	PartialWave,
};

export struct [[nodiscard]] WorkloadError {
	WorkloadFault fault;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		return core::Retry::Permanent;
	}

	[[nodiscard]] constexpr auto operator==(WorkloadError const&) const -> bool = default;
};

/** A mode and prompts that fill whole waves of it: every wave runs exactly the mode's streams, one prompt each. */
export class Workload {
public:
	[[nodiscard]] static auto make(Streams streams, std::vector<Prompt> prompts) -> core::Result<Workload, WorkloadError> {
		if (prompts.empty()) {
			return std::unexpected(WorkloadError{.fault = WorkloadFault::NoPrompts});
		}
		if (prompts.size() > max_prompts) {
			return std::unexpected(WorkloadError{.fault = WorkloadFault::TooManyPrompts});
		}
		if (prompts.size() % count(streams) != 0) {
			return std::unexpected(WorkloadError{.fault = WorkloadFault::PartialWave});
		}
		return Workload{streams, std::move(prompts)};
	}

	[[nodiscard]] auto streams() const -> Streams {
		return streams_;
	}

	[[nodiscard]] auto prompts() const -> std::span<Prompt const> {
		return prompts_;
	}

	[[nodiscard]] auto waves() const -> std::size_t {
		return prompts_.size() / count(streams_);
	}

	/** The prompts wave `index` sends, one per stream; the first is the workload's prompt `index * count(streams())`. */
	[[nodiscard]] auto wave(std::size_t index) const -> std::span<Prompt const> {
		core::invariant(index < waves());
		return prompts().subspan(index * count(streams_), count(streams_));
	}

private:
	Workload(Streams streams, std::vector<Prompt> prompts) : streams_{streams}, prompts_{std::move(prompts)} {}

	Streams streams_;
	std::vector<Prompt> prompts_;
};

}
