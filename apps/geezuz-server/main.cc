// SPDX-License-Identifier: AGPL-3.0-only
//
// geezuz-server: the OpenAI Responses api over the engine on both cards.
//
//   geezuz-server <image-dir> <address>:<port>
//
// <image-dir> holds the three images geezuz-bake wrote; the server listens on the IPv4 address and port (port 0 takes
// an ephemeral one). The socket opens first, and the line `geezuz-server: listening on <address>:<port>` says so, so
// /health answers 503 while the engine loads. The engine then starts on a thread of its own, refusing in order: the
// machine, its memory clock, the images, the loops. A start that fails is printed and becomes /health's reason, and
// every other route is refused by name; the server keeps answering until SIGINT or SIGTERM, then exits 1. An engine
// that served exits 0 when stopped, 1 if it failed while serving. Exit status 2 is a command line that is not the above.
import std;
import engine.cards;
import engine.core;
import engine.net;
import engine.openai;
import engine.process;

namespace {

namespace cards = engine::cards;
namespace net = engine::net;
namespace openai = engine::openai;
namespace process = engine::process;

using cards::Engine;

/** Time and randomness for the service: the wall clock, and a generator seeded once from the system's entropy. */
class MachineSources {
public:
	explicit MachineSources(std::uint64_t seed) : random_{seed} {}

	[[nodiscard]] auto unix_seconds() const -> std::int64_t {
		return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}

	[[nodiscard]] auto draw() -> std::uint64_t {
		return random_();
	}

private:
	std::mt19937_64 random_;
};

static_assert(openai::Sources<MachineSources>);

/** The decimal number that is all of `text`, if it fits `T`. */
template<std::unsigned_integral T>
[[nodiscard]] auto number(std::string_view text) -> std::optional<T> {
	auto value = T{0};
	auto const read = std::from_chars(text.data(), std::to_address(text.end()), value);
	if (text.empty() || std::make_error_code(read.ec) || read.ptr != std::to_address(text.end())) {
		return std::nullopt;
	}
	return value;
}

/** `a.b.c.d:port`, each part decimal: the one way the server is told where to listen. */
[[nodiscard]] auto endpoint_of(std::string_view text) -> std::optional<net::Endpoint> {
	auto const colon = text.rfind(':');
	if (colon == std::string_view::npos) {
		return std::nullopt;
	}
	auto const port = number<std::uint16_t>(text.substr(colon + 1));
	auto address = net::Address{};
	auto octets = std::size_t{0};
	for (auto const part : std::views::split(text.substr(0, colon), '.')) {
		auto const octet = number<std::uint8_t>(std::string_view{part.begin(), part.end()});
		if (!octet || octets == address.size()) {
			return std::nullopt;
		}
		address.at(octets) = *octet;
		octets += 1;
	}
	if (!port || octets != address.size()) {
		return std::nullopt;
	}
	return net::Endpoint{.address = address, .port = *port};
}

[[nodiscard]] auto serve(std::filesystem::path const& directory, net::Endpoint endpoint) -> int {
	auto entropy = std::random_device{};
	auto const seed = (std::uint64_t{entropy()} << 32U) | entropy();
	auto sources = MachineSources{seed};
	auto server = openai::Server::open(openai::server_config(endpoint, sources));
	if (!server) {
		process::output_line(std::format("geezuz-server: cannot listen on the address (system error {})", server.error().code));
		return 1;
	}
	process::output_line(std::format("geezuz-server: listening on {}.{}.{}.{}:{}", endpoint.address[0], endpoint.address[1],
	                                 endpoint.address[2], endpoint.address[3], server->port()));
	// Before any other thread starts: on linux the signals are blocked for the whole process.
	if (auto const signals = server->stop_on_signals(); !signals) {
		process::output_line(std::format("geezuz-server: cannot take SIGINT and SIGTERM (system error {})", signals.error().code));
		return 1;
	}
	auto engine = Engine::start(directory, server->poster(), [](std::string_view reason) {
		process::output_line(std::format("geezuz-server: the engine could not start: {}", reason));
	});
	if (!engine) {
		process::output_line(std::format("geezuz-server: cannot start the engine's loader (system error {})", engine.error().code));
		return 1;
	}
	auto service = openai::Service<Engine, MachineSources>::open(std::move(*engine), std::move(sources));
	if (!service) {
		process::output_line("geezuz-server: the request decoder built into the program does not load");
		return 1;
	}
	auto const ran = server->run(*service);
	auto const ended = service->engine().stop();
	if (!ran) {
		process::output_line(std::format("geezuz-server: the server's wait failed (system error {})", ran.error().code));
	}
	if (!ended) {
		process::output_line(std::format("geezuz-server: stopped after a failure: {}", cards::describe(ended.error())));
	}
	return ran && ended ? 0 : 1;
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	auto const endpoint = arguments.size() == 3 ? endpoint_of(arguments[2]) : std::nullopt;
	if (!endpoint) {
		process::output_line("usage: geezuz-server <image-dir> <address>:<port>");
		return 2;
	}
	return serve(std::filesystem::path{arguments[1]}, *endpoint);
}
