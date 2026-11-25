// SPDX-License-Identifier: AGPL-3.0-only
//
// geezuz-server run as a program on loopback, on a machine without the cards (the development mac): it opens its
// socket before anything else, so /health answers while the engine starts; the engine's start fails with the driver's
// typed refusal, which /health then reports as its reason, and every other route is refused as unavailable; SIGTERM
// ends the server with a nonzero status, since it never served. The expected reason is the bring-up's own failure on
// this machine, rendered by its own description: the server must report exactly what starting the cards here returns.
import std;
import engine.cards;
import engine.core;
import engine.driver;
import engine.http_test_client;
import engine.json;
import engine.openai;
import engine.process;
import engine.testing;

namespace {

namespace cards = engine::cards;
namespace client = engine::http_test_client;
namespace openai = engine::openai;
namespace process = engine::process;

using Clock = std::chrono::steady_clock;

constexpr auto patience = std::chrono::seconds{30};

/** The server's port, from the line it prints once its socket is open. */
[[nodiscard]] auto listening_port(process::Child& server) -> std::optional<std::uint16_t> {
	constexpr auto prefix = std::string_view{"geezuz-server: listening on 127.0.0.1:"};
	auto const deadline = Clock::now() + patience;
	for (;;) {
		auto read = server.read_line_until(deadline);
		if (!read) {
			return std::nullopt;
		}
		auto const line = engine::core::alternative<process::Line>(*read);
		if (!line) {
			return std::nullopt;
		}
		if (line->text.starts_with(prefix)) {
			auto port = std::uint16_t{0};
			auto const digits = std::string_view{line->text}.substr(prefix.size());
			auto const parsed = std::from_chars(digits.data(), std::to_address(digits.end()), port);
			if (std::make_error_code(parsed.ec) || parsed.ptr != std::to_address(digits.end())) {
				return std::nullopt;
			}
			return port;
		}
	}
}

using Reader = engine::json::Reader<engine::json::request_bounds>;

/** An object's string members, read with the engine's own JSON reader: the oracle reads JSON, not text. */
template<class O>
[[nodiscard]] auto strings(O& object) -> std::optional<std::map<std::string, std::string>> {
	auto out = std::map<std::string, std::string>{};
	for (;;) {
		auto member = object.next();
		if (!member) {
			return std::nullopt;
		}
		if (!*member) {
			return out;
		}
		auto& [name, value] = **member;
		if (auto const text = value.as_string()) {
			out.emplace(name, *text);
		}
	}
}

/** /health's body: `{"status": ..., "reason": ...}`. */
[[nodiscard]] auto health_of(Reader& reader, std::string_view body) -> std::optional<std::map<std::string, std::string>> {
	auto root = reader.read_copy(body);
	if (!root) {
		return std::nullopt;
	}
	auto object = root->as_object();
	if (!object) {
		return std::nullopt;
	}
	auto members = strings(*object);
	if (!members || !reader.finish()) {
		return std::nullopt;
	}
	return members;
}

/** The `code` of an OpenAI error object, `{"error": {..., "code": ...}}`. */
[[nodiscard]] auto error_code(Reader& reader, std::string_view body) -> std::optional<std::string> {
	auto root = reader.read_copy(body);
	if (!root) {
		return std::nullopt;
	}
	auto outer = root->as_object();
	if (!outer) {
		return std::nullopt;
	}
	auto first = outer->next();
	if (!first || !*first) {
		return std::nullopt;
	}
	auto& [name, value] = **first;
	if (name != "error") {
		return std::nullopt;
	}
	auto inner = value.as_object();
	if (!inner) {
		return std::nullopt;
	}
	auto const members = strings(*inner);
	if (!members) {
		return std::nullopt;
	}
	auto const code = members->find("code");
	if (code == members->end()) {
		return std::nullopt;
	}
	return code->second;
}

/** One request on a fresh connection, and the server's whole response to it. */
[[nodiscard]] auto exchange(std::uint16_t port, std::string_view request) -> std::optional<client::Parsed> {
	auto connection = client::connect(port);
	if (!connection || connection->push(request) != request.size()) {
		return std::nullopt;
	}
	auto const deadline = Clock::now() + patience;
	while (Clock::now() < deadline) {
		connection->pull();
		auto parsed = client::parse_responses(connection->received);
		if (parsed.malformed) {
			return std::nullopt;
		}
		if (!parsed.complete.empty()) {
			return std::move(parsed.complete.front());
		}
		if (connection->peer == client::Peer::Closed) {
			return std::nullopt;
		}
		std::this_thread::sleep_until(Clock::now() + std::chrono::milliseconds{5});
	}
	return std::nullopt;
}

}

auto main(int argc, char** argv) -> int {
	using engine::testing::Case;
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	if (arguments.size() != 2) {
		std::println("FAIL: the test takes the server binary's path");
		return 1;
	}
	auto const binary = std::filesystem::path{arguments[1]};
	auto const images = std::filesystem::current_path() / "server.scratch";
	std::filesystem::create_directories(images);

	auto const expected = cards::start(images);
	if (expected) {
		std::println("FAIL: this test runs where the cards cannot start, and they started");
		return 1;
	}
	auto const reason = cards::describe(expected.error());

	auto server = process::Child::spawn(binary, std::vector<std::string>{images.string(), "127.0.0.1:0"});
	if (!server) {
		std::println("FAIL: the server could not be spawned");
		return 1;
	}
	auto const port = listening_port(*server);
	auto reader = Reader::make();
	if (!reader) {
		std::println("FAIL: no JSON reader");
		return 1;
	}
	// /health says `loading` until the engine's start ends; the start on this machine ends in milliseconds.
	auto health = std::optional<client::Parsed>{};
	auto health_body = std::optional<std::map<std::string, std::string>>{};
	auto const settled = Clock::now() + patience;
	while (port && Clock::now() < settled) {
		health = exchange(*port, "GET /health HTTP/1.1\r\nHost: geezuz\r\n\r\n");
		health_body = health.and_then([&](client::Parsed const& response) {
			return health_of(*reader, response.body);
		});
		if (!health_body || health_body->at("status") != "loading") {
			break;
		}
		std::this_thread::sleep_until(Clock::now() + std::chrono::milliseconds{20});
	}
	auto const models = port.and_then([&](std::uint16_t open) {
		return exchange(open, "GET /v1/models HTTP/1.1\r\nHost: geezuz\r\n\r\n");
	});
	auto const models_code = models.and_then([&](client::Parsed const& response) {
		return error_code(*reader, response.body);
	});
	auto const signalled = server->terminate();
	auto const exit = server->wait_until(Clock::now() + patience);

	auto const cases = std::array{
	    Case{.name = "the server prints its port once its socket is open", .holds = port.has_value()},
	    Case{.name = "the start failure on this machine is the driver's typed refusal",
		     .holds = std::holds_alternative<engine::driver::DeviceError>(expected.error().reason) &&
		              std::holds_alternative<engine::driver::DriverUnavailable>(
	                      std::get<engine::driver::DeviceError>(expected.error().reason).reason)},
	    Case{.name = "/health answers 503 with that failure as its reason",
		     .holds = health && health->status == 503 &&
		              health_body == std::map<std::string, std::string>{{"status", "failed"}, {"reason", reason}}},
	    Case{.name = "every other route is refused by name: the engine failed",
		     .holds = models && models->status == 503 && models_code == std::string{openai::spelling(openai::Code::EngineFailed)}},
	    Case{.name = "SIGTERM ends the server, with a nonzero status since it never served",
		     .holds = signalled && exit && exit->has_value() && std::holds_alternative<process::Exited>(**exit) &&
		              std::get<process::Exited>(**exit).status != 0},
	};
	return engine::testing::report(cases);
}
