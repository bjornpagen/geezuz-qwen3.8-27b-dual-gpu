// SPDX-License-Identifier: AGPL-3.0-only
export module engine.throughput:request;

import std;
import engine.json;
import engine.net;
import :workload;

/**
 * The two requests the client sends, whole, each on a connection of its own that the server closes after its answer:
 * no connection sits idle through a run to be found closed when it is next needed.
 */
namespace engine::throughput {

[[nodiscard]] auto host_of(net::Endpoint server) -> std::string {
	return std::format("{}.{}.{}.{}:{}", server.address[0], server.address[1], server.address[2], server.address[3], server.port);
}

/**
 * The body that asks for one streamed response to `prompt`: the measurement's own choices and nothing else, so every
 * other member is the server's default, as a client that names none gets it. Temperature zero makes the reply the
 * same in every run.
 */
export [[nodiscard]] auto create_body(Prompt const& prompt, std::uint32_t max_output_tokens) -> std::string {
	return json::encode([&prompt, max_output_tokens](auto& out) {
		json::write_object(out, [&prompt, max_output_tokens](auto& members) {
			members.string("input", prompt.text());
			members.boolean("stream", true);
			members.real("temperature", 0.0);
			members.integer("max_output_tokens", max_output_tokens);
		});
	});
}

/** `POST /v1/responses` for `prompt`, as the bytes of one request. */
export [[nodiscard]] auto create_request(net::Endpoint server, Prompt const& prompt, std::uint32_t max_output_tokens) -> std::string {
	auto const body = create_body(prompt, max_output_tokens);
	return std::format("POST /v1/responses HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n"
	                   "Connection: close\r\n\r\n{}",
	                   host_of(server), body.size(), body);
}

/** `GET /status`, as the bytes of one request. */
export [[nodiscard]] auto status_request(net::Endpoint server) -> std::string {
	return std::format("GET /status HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", host_of(server));
}

}
