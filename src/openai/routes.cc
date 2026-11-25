// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:routes;

import std;
import engine.core;
import engine.http;
import engine.json;
import engine.pipeline;
import :boundary;
import :error;

/**
 * The answers that need no engine: the model list, readiness, the engine's snapshot, and the queries of the stored
 * response routes, each refused by name where the sdk's value is one the server does not serve.
 */
namespace engine::openai {

/** The engine is loading: nothing but `/health` is served. */
export struct Loading {
	[[nodiscard]] constexpr auto operator==(Loading const&) const -> bool = default;
};

/** Whether the engine serves: held once, by the service, and changed only by the updates posted to it. */
export using Readiness = std::variant<Loading, EngineFailed, Ready>;

/** The refusal of every route but `/health` while the engine does not serve. */
export [[nodiscard]] auto unready(Readiness const& readiness) -> std::optional<ApiError> {
	if (std::holds_alternative<Loading>(readiness)) {
		return ApiError{.code = Code::EngineLoading, .message = "the engine is still loading; retry shortly", .param = std::nullopt};
	}
	if (auto const failed = core::alternative<EngineFailed>(readiness)) {
		return ApiError{.code = Code::EngineFailed, .message = std::format("the engine failed: {}", failed->reason), .param = std::nullopt};
	}
	return std::nullopt;
}

/** GET /health: 200 once the engine serves; 503 with the reason until then, or for good once it failed. */
export [[nodiscard]] auto health_response(Readiness const& readiness) -> http::Response {
	auto const status = std::holds_alternative<Ready>(readiness) ? http::StatusCode::Ok : http::StatusCode::ServiceUnavailable;
	auto body = json::encode([&readiness](auto& out) {
		json::write_object(out, [&readiness](auto& members) {
			if (std::holds_alternative<Ready>(readiness)) {
				members.string("status", "ok");
			} else if (std::holds_alternative<Loading>(readiness)) {
				members.string("status", "loading");
				members.string("reason", "the engine is still loading");
			} else {
				members.string("status", "failed");
				members.string("reason", std::get<EngineFailed>(readiness).reason);
			}
		});
	});
	return http::Response{.status = status, .content_type = http::ContentType::Json, .body = std::move(body)};
}

/** GET /v1/models: the one model. `created` is when the server started, in Unix seconds. */
export [[nodiscard]] auto models_body(std::int64_t created) -> std::string {
	return json::encode([created](auto& out) {
		json::write_object(out, [created]<class O>(json::Members<O>& list) {
			list.string("object", "list");
			list.array("data", [created](json::Elements<O>& models) {
				models.object([created](json::Members<O>& entry) {
					entry.string("id", model_id);
					entry.string("object", "model");
					entry.integer("created", created);
					entry.string("owned_by", "geezuz");
				});
			});
		});
	});
}

template<json::Sink O>
struct SlotFields {
	json::Members<O>& members;

	auto operator()(pipeline::Vacant const& /*vacant*/) const -> void {
		members.string("phase", "vacant");
	}

	auto operator()(pipeline::Prefilling<std::string> const& slot) const -> void {
		members.string("phase", "prefill");
		members.string("request", slot.tag);
		members.integer("prompt_tokens", slot.prompt_tokens);
		members.integer("cached_tokens", slot.cached_tokens);
		members.integer("prefilled_tokens", slot.prefilled_tokens);
	}

	auto operator()(pipeline::Decoding<std::string> const& slot) const -> void {
		members.string("phase", "decode");
		members.string("request", slot.tag);
		members.integer("prompt_tokens", slot.prompt_tokens);
		members.integer("cached_tokens", slot.cached_tokens);
		members.integer("generated_tokens", slot.generated_tokens);
	}
};

/** GET /status: the engine's latest snapshot, or `null` before it posted one. */
export [[nodiscard]] auto status_body(std::optional<Snapshot> const& snapshot) -> std::string {
	return json::encode([&snapshot](auto& out) {
		json::write_object(out, [&snapshot]<class O>(json::Members<O>& members) {
			if (!snapshot) {
				members.null("engine");
				return;
			}
			members.object("engine", [&snapshot](json::Members<O>& engine) {
				engine.array("slots", [&snapshot](json::Elements<O>& slots) {
					for (auto const& slot : snapshot->slots) {
						slots.object([&slot](json::Members<O>& written) {
							std::visit(SlotFields<O>{.members = written}, slot);
						});
					}
				});
				engine.integer("queued", snapshot->queued);
				engine.object("kv_pool", [&snapshot](json::Members<O>& pool) {
					pool.integer("blocks", snapshot->pool.blocks);
					pool.integer("used_blocks", snapshot->pool.used_blocks);
					pool.integer("cached_blocks", snapshot->pool.cached_blocks);
				});
				auto const& counters = snapshot->counters;
				engine.object("counters", [&counters](json::Members<O>& written) {
					written.integer("cycles", counters.cycles);
					written.integer("drafted_tokens", counters.drafted_tokens);
					written.integer("accepted_tokens", counters.accepted_tokens);
					written.integer("prefill_chunks", counters.prefill_chunks);
					written.integer("prefill_tokens", counters.prefill_tokens);
					written.integer("preemptions", counters.preemptions);
					written.integer("snapshots_refused", counters.snapshots_refused);
				});
			});
		});
	});
}

[[nodiscard]] auto query_fault(Code code, std::string_view name, std::string message) -> std::unexpected<ApiError> {
	return std::unexpected(ApiError{.code = code, .message = std::move(message), .param = std::string{name}});
}

/** A query parameter the server takes at one value only: `served` is accepted, `unserved` refused by name. */
[[nodiscard]] auto two_valued(http::QueryParameter const& parameter, std::string_view served, std::string_view unserved)
    -> core::Result<void, ApiError> {
	if (parameter.value == served) {
		return {};
	}
	if (parameter.value == unserved) {
		return query_fault(Code::UnsupportedParameter, parameter.name,
		                   std::format("'{}' is not supported with the value {}", parameter.name, unserved));
	}
	return query_fault(Code::InvalidValue, parameter.name, std::format("'{}' must be true or false", parameter.name));
}

/**
 * GET /v1/responses/{id}'s query: `include` names output this server never makes, so any is refused; resuming a
 * stream (`stream=true`, `starting_after`) is refused, as is obfuscation; anything else is no parameter of the route.
 */
export [[nodiscard]] auto check_retrieve(http::Query const& query) -> core::Result<void, ApiError> {
	for (auto const& parameter : query.parameters) {
		auto checked = core::Result<void, ApiError>{};
		if (parameter.name == "include" || parameter.name == "include[]" || parameter.name == "starting_after") {
			checked = query_fault(Code::UnsupportedParameter, parameter.name, std::format("'{}' is not supported", parameter.name));
		} else if (parameter.name == "stream" || parameter.name == "include_obfuscation") {
			checked = two_valued(parameter, "false", "true");
		} else {
			checked =
			    query_fault(Code::UnknownParameter, parameter.name, std::format("'{}' is not a parameter of this route", parameter.name));
		}
		if (!checked) {
			return checked;
		}
	}
	return {};
}

export enum class Order : std::uint8_t {
	Ascending,
	Descending,
};

/** One page of a stored response's input items, as the query asks for it. */
export struct Paging {
	std::optional<std::string> after;
	std::size_t limit;
	Order order;

	[[nodiscard]] auto operator==(Paging const&) const -> bool = default;
};

/** The sdk's bounds on a page. */
inline constexpr std::size_t default_page = 20;
inline constexpr std::size_t max_page = 100;

/** GET /v1/responses/{id}/input_items's query: `after`, `limit` (1 to 100, default 20), `order` (default `desc`). */
export [[nodiscard]] auto paging_of(http::Query const& query) -> core::Result<Paging, ApiError> {
	auto paging = Paging{.after = std::nullopt, .limit = default_page, .order = Order::Descending};
	auto seen = std::vector<std::string_view>{};
	for (auto const& parameter : query.parameters) {
		auto const& name = parameter.name;
		auto const& value = parameter.value;
		if (std::ranges::contains(seen, std::string_view{name})) {
			return query_fault(Code::InvalidValue, name, std::format("'{}' is given more than once", name));
		}
		if (name == "after") {
			paging.after = value;
		} else if (name == "limit") {
			auto limit = std::size_t{0};
			auto const read = std::from_chars(value.data(), std::to_address(value.end()), limit);
			if (std::make_error_code(read.ec) || read.ptr != std::to_address(value.end()) || limit < 1 || limit > max_page) {
				return query_fault(Code::InvalidValue, name, std::format("'limit' must be an integer from 1 to {}", max_page));
			}
			paging.limit = limit;
		} else if (name == "order") {
			if (value != "asc" && value != "desc") {
				return query_fault(Code::InvalidValue, name, "'order' must be asc or desc");
			}
			paging.order = value == "asc" ? Order::Ascending : Order::Descending;
		} else if (name == "include" || name == "include[]") {
			return query_fault(Code::UnsupportedParameter, name, std::format("'{}' is not supported", name));
		} else {
			return query_fault(Code::UnknownParameter, name, std::format("'{}' is not a parameter of this route", name));
		}
		seen.push_back(name);
	}
	return paging;
}

/** The page `paging` asks for of `items`, held in input order; `after` must name one of them. */
export [[nodiscard]] auto page_of(std::span<json::StoredInput const> items, Paging const& paging) -> core::Result<std::string, ApiError> {
	auto ordered = std::vector<json::StoredInput>{items.begin(), items.end()};
	if (paging.order == Order::Descending) {
		std::ranges::reverse(ordered);
	}
	auto start = std::size_t{0};
	if (paging.after) {
		auto const found = std::ranges::find(ordered, *paging.after, &json::StoredInput::id);
		if (found == ordered.end()) {
			return query_fault(Code::InvalidValue, "after",
			                   std::format("'after' names no input item of this response: '{}'", *paging.after));
		}
		start = core::narrow<std::size_t>(found - ordered.begin()) + 1;
	}
	auto const page = std::span{ordered}.subspan(start, std::min(paging.limit, ordered.size() - start));
	return json::encode_input_items(page, start + page.size() < ordered.size());
}

}
