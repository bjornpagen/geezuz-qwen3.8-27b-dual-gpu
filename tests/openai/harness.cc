// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai_harness;

import std;
import engine.core;
import engine.http;
import engine.http_test_client;
import engine.json;
import engine.net;
import engine.openai;

/**
 * The tests' side of the service: a server and its service turned by the test's thread between client operations,
 * requests written as a client writes them, and a reader of the bodies and events the server sends, which reads JSON
 * with the engine's reader by path, never by text.
 */
namespace engine::openai_harness {

namespace http = engine::http;
namespace net = engine::net;
namespace json = engine::json;
namespace openai = engine::openai;
namespace client = engine::http_test_client;
using namespace std::chrono_literals;

/** The one clock reading every test's responses carry. */
export inline constexpr std::int64_t now = 1'700'000'000;

/** A fixed clock and a splitmix64 stream: every id and seed the service makes replays. */
export struct FixedSources {
	std::uint64_t state;

	[[nodiscard]] auto unix_seconds() const -> std::int64_t {
		return now;
	}

	[[nodiscard]] auto draw() -> std::uint64_t {
		state += 0x9e37'79b9'7f4a'7c15ULL;
		auto value = state;
		value = (value ^ (value >> 30U)) * 0xbf58'476d'1ce4'e5b9ULL;
		value = (value ^ (value >> 27U)) * 0x94d0'49bb'1331'11ebULL;
		return value ^ (value >> 31U);
	}
};

static_assert(openai::Sources<FixedSources>);

export struct Timeouts {
	std::chrono::milliseconds read;
	std::chrono::milliseconds write;
};

export inline constexpr auto quick = Timeouts{.read = 5000ms, .write = 5000ms};

/** One server and its service, turned by the test's thread between client operations. */
export template<class Service>
struct Harness {
	openai::Server server;
	Service service;

	auto turn() -> void {
		std::ignore = server.turn(service, net::Clock::now() + 1ms);
	}

	/** Turns the server until `done` holds after the client has pulled what arrived, or until `budget` passes. */
	template<class Done>
	[[nodiscard]] auto pump(client::Client& peer, Done done, std::chrono::milliseconds budget) -> bool {
		auto const deadline = net::Clock::now() + budget;
		for (;;) {
			peer.pull();
			if (done()) {
				return true;
			}
			if (net::Clock::now() >= deadline) {
				return false;
			}
			turn();
		}
	}

	/** Turns the server until `done` holds, reading nothing. */
	template<class Done>
	[[nodiscard]] auto wait(Done done, std::chrono::milliseconds budget) -> bool {
		auto const deadline = net::Clock::now() + budget;
		while (!done()) {
			if (net::Clock::now() >= deadline) {
				return false;
			}
			turn();
		}
		return true;
	}

	auto idle(std::chrono::milliseconds span) -> void {
		auto const deadline = net::Clock::now() + span;
		while (net::Clock::now() < deadline) {
			turn();
		}
	}

	[[nodiscard]] auto push(client::Client& peer, std::string_view bytes) -> bool {
		auto sent = std::size_t{0};
		auto const deadline = net::Clock::now() + 5000ms;
		while (sent < bytes.size() && net::Clock::now() < deadline) {
			sent += peer.push(bytes.substr(sent));
			turn();
		}
		return sent == bytes.size();
	}

	/** Sends one request on `peer` (or nothing, for an answer already asked for) and waits for its whole response. */
	[[nodiscard]] auto ask(client::Client& peer, std::string_view request) -> std::optional<client::Parsed> {
		auto const before = client::parse_responses(peer.received).complete.size();
		auto const arrived = [&] {
			return client::parse_responses(peer.received).complete.size() > before;
		};
		if (!push(peer, request) || !pump(peer, arrived, 10000ms)) {
			return std::nullopt;
		}
		return client::parse_responses(peer.received).complete[before];
	}
};

/** Opens a server on an ephemeral loopback port, with its service over `make(poster)`'s engine. */
export template<class Service, class Make>
[[nodiscard]] auto harness(Timeouts timeouts, Make make, FixedSources sources) -> std::unique_ptr<Harness<Service>> {
	auto server = openai::Server::open(http::ServerConfig{
	    .endpoint = net::Endpoint{.address = net::loopback, .port = 0},
	    .read_timeout = timeouts.read,
	    .write_timeout = timeouts.write,
	    .id_seed = 11,
	});
	if (!server) {
		std::println("server open failed: errno {}", server.error().code);
		return nullptr;
	}
	auto service = Service::open(make(server->poster()), sources);
	if (!service) {
		return nullptr;
	}
	return std::make_unique<Harness<Service>>(Harness<Service>{.server = std::move(*server), .service = std::move(*service)});
}

[[nodiscard]] auto id_field(std::string_view id) -> std::string {
	return id.empty() ? std::string{} : std::format("X-Request-Id: {}\r\n", id);
}

export [[nodiscard]] auto get(std::string_view path, std::string_view id) -> std::string {
	return std::format("GET {} HTTP/1.1\r\nHost: test\r\n{}\r\n", path, id_field(id));
}

export [[nodiscard]] auto remove(std::string_view path, std::string_view id) -> std::string {
	return std::format("DELETE {} HTTP/1.1\r\nHost: test\r\n{}\r\n", path, id_field(id));
}

export [[nodiscard]] auto post(std::string_view path, std::string_view body, std::string_view id) -> std::string {
	return std::format("POST {} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}\r\n{}", path,
	                   body.size(), id_field(id), body);
}

export [[nodiscard]] auto create(std::string_view body, std::string_view id) -> std::string {
	return post("/v1/responses", body, id);
}

export [[nodiscard]] auto echoes(client::Parsed const& response, std::string_view id) -> bool {
	auto const field = response.fields.find("x-request-id");
	return field != response.fields.end() && field->second == id;
}

export using Step = std::variant<std::string_view, std::size_t>;

/** Reads members of a JSON document by path: the oracle reads JSON with the engine's reader, not by text. */
export class Json {
public:
	[[nodiscard]] static auto make() -> std::optional<Json> {
		auto reader = json::Reader<json::request_bounds>::make();
		if (!reader) {
			return std::nullopt;
		}
		return Json{*std::move(reader)};
	}

	[[nodiscard]] auto text(std::string_view document, std::initializer_list<Step> path) -> std::optional<std::string> {
		return read<std::string>(document, path, [](auto& value) -> std::optional<std::string> {
			auto const text = value.as_string();
			return text ? std::optional{std::string{*text}} : std::nullopt;
		});
	}

	[[nodiscard]] auto integer(std::string_view document, std::initializer_list<Step> path) -> std::optional<std::int64_t> {
		return read<std::int64_t>(document, path, [](auto& value) -> std::optional<std::int64_t> {
			auto const number = value.as_integer();
			return number ? std::optional{*number} : std::nullopt;
		});
	}

	[[nodiscard]] auto real(std::string_view document, std::initializer_list<Step> path) -> std::optional<double> {
		return read<double>(document, path, [](auto& value) -> std::optional<double> {
			auto const number = value.as_real();
			return number ? std::optional{*number} : std::nullopt;
		});
	}

	[[nodiscard]] auto boolean(std::string_view document, std::initializer_list<Step> path) -> std::optional<bool> {
		return read<bool>(document, path, [](auto& value) -> std::optional<bool> {
			auto const flag = value.as_boolean();
			return flag ? std::optional{*flag} : std::nullopt;
		});
	}

	[[nodiscard]] auto kind(std::string_view document, std::initializer_list<Step> path) -> std::optional<json::Kind> {
		return read<json::Kind>(document, path, [](auto& value) -> std::optional<json::Kind> {
			auto const kind = value.kind();
			return kind ? std::optional{*kind} : std::nullopt;
		});
	}

	[[nodiscard]] auto count(std::string_view document, std::initializer_list<Step> path) -> std::optional<std::size_t> {
		return read<std::size_t>(document, path, [](auto& value) -> std::optional<std::size_t> {
			auto array = value.as_array();
			if (!array) {
				return std::nullopt;
			}
			auto const size = array->size();
			return size ? std::optional{*size} : std::nullopt;
		});
	}

private:
	explicit Json(json::Reader<json::request_bounds> reader) : reader_{std::move(reader)} {}

	template<class T, class Value, class F>
	[[nodiscard]] static auto at(Value& value, std::span<Step const> path, F const& read) -> std::optional<T> {
		if (path.empty()) {
			return read(value);
		}
		if (auto const name = engine::core::alternative<std::string_view>(path.front())) {
			auto object = value.as_object();
			if (!object) {
				return std::nullopt;
			}
			auto member = object->find(*name);
			if (!member || !*member) {
				return std::nullopt;
			}
			return at<T>(**member, path.subspan(1), read);
		}
		auto array = value.as_array();
		if (!array) {
			return std::nullopt;
		}
		for (auto index = std::size_t{0};; ++index) {
			auto element = array->next();
			if (!element || !*element) {
				return std::nullopt;
			}
			if (index == std::get<std::size_t>(path.front())) {
				return at<T>(**element, path.subspan(1), read);
			}
		}
	}

	template<class T, class F>
	[[nodiscard]] auto read(std::string_view document, std::initializer_list<Step> path, F const& read) -> std::optional<T> {
		auto root = reader_.read_copy(document);
		if (!root) {
			return std::nullopt;
		}
		return at<T>(*root, std::span{path.begin(), path.size()}, read);
	}

	json::Reader<json::request_bounds> reader_;
};

}
