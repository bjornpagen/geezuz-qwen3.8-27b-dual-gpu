// SPDX-License-Identifier: AGPL-3.0-only
export module engine.openai:render;

import std;
import engine.core;
import engine.http;
import engine.json;
import :engine;
import :error;

namespace engine::openai {

/** The one model this server serves, as /v1/models lists it and every completion names it. */
export inline constexpr std::string_view model_id = "Qwen3.8-27B-FP8";

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

/** GET /health: 200 once the engine serves, 503 while it loads. */
export [[nodiscard]] auto health_response(Readiness readiness) -> http::Response {
	switch (readiness) {
	case Readiness::Loading:
		return http::Response{.status = http::StatusCode::ServiceUnavailable, .content_type = http::ContentType::Json, .body = R"({"status":"loading"})"};
	case Readiness::Serving:
		return http::Response{.status = http::StatusCode::Ok, .content_type = http::ContentType::Json, .body = R"({"status":"ok"})"};
	}
	std::unreachable();
}

[[nodiscard]] constexpr auto spelling(Readiness readiness) -> std::string_view {
	switch (readiness) {
	case Readiness::Loading:
		return "loading";
	case Readiness::Serving:
		return "serving";
	}
	std::unreachable();
}

template<json::Sink O>
struct SlotFields {
	json::Members<O>& members;

	auto operator()(Vacant const& /*vacant*/) const -> void {
		members.string("phase", "vacant");
	}

	auto operator()(Prefilling const& slot) const -> void {
		members.string("phase", "prefill");
		members.string("id", slot.id.text());
		members.integer("prompt_tokens", slot.prompt_tokens);
		members.integer("cached_tokens", slot.cached_tokens);
		members.integer("prefilled_tokens", slot.prefilled_tokens);
	}

	auto operator()(Decoding const& slot) const -> void {
		members.string("phase", "decode");
		members.string("id", slot.id.text());
		members.integer("prompt_tokens", slot.prompt_tokens);
		members.integer("cached_tokens", slot.cached_tokens);
		members.integer("generated_tokens", slot.generated_tokens);
	}
};

template<json::Sink O>
auto put_status(json::Members<O>& members, Status const& status) -> void {
	auto const busy = std::ranges::count_if(status.slots, [](Slot const& slot) { return !std::holds_alternative<Vacant>(slot); });
	members.string("readiness", spelling(status.readiness));
	members.integer("busy", busy);
	members.integer("queued", status.queued);
	members.integer("preemptions", status.preemptions);
	members.array("slots", [&status](json::Elements<O>& slots) {
		for (auto index = std::size_t{0}; index < status.slots.size(); ++index) {
			slots.object([&](json::Members<O>& slot) {
				slot.integer("slot", index);
				std::visit(SlotFields<O>{.members = slot}, status.slots[index]);
			});
		}
	});
	members.object("kv_pool", [&status](json::Members<O>& pool) {
		pool.integer("blocks", status.pool.blocks);
		pool.integer("used_blocks", status.pool.used_blocks);
		pool.integer("cached_blocks", status.pool.cached_blocks);
	});
	members.object("draft", [&status](json::Members<O>& draft) {
		draft.integer("cycles", status.cycles.cycles);
		draft.integer("drafted", status.cycles.drafted_tokens);
		draft.integer("accepted", status.cycles.accepted_tokens);
		// No draft yet has no rate, rather than a rate of zero.
		if (status.cycles.drafted_tokens == 0) {
			draft.null("accept_rate");
		} else {
			draft.real("accept_rate",
			           static_cast<double>(status.cycles.accepted_tokens) / static_cast<double>(status.cycles.drafted_tokens));
		}
	});
}

/** GET /status: the engine's latest snapshot. */
export [[nodiscard]] auto status_body(Status const& status) -> std::string {
	return json::encode([&status](auto& out) { json::write_object(out, [&status](auto& members) { put_status(members, status); }); });
}

/** The sentinel that ends every OpenAI stream: the last event's data, and not JSON. */
constexpr std::string_view done_sentinel = "[DONE]";

/** Renders a stream's events into one buffer kept for the server's life, of exactly one event's bound. */
export class EventWriter {
public:
	EventWriter() : buffer_{http::max_event_bytes} {}

	/** The chunk's event, or how far it overflows one event. */
	[[nodiscard]] auto chunk(json::ChunkHeader const& header, json::ChunkEvent const& event) -> core::Result<http::Event, json::Overflow> {
		buffer_.clear();
		if (auto appended = json::append(buffer_, header, event); !appended) {
			return std::unexpected(appended.error());
		}
		return http::Event{.data = std::string{buffer_.text()}};
	}

	[[nodiscard]] static auto done() -> http::Event {
		return http::Event{.data = std::string{done_sentinel}};
	}

	/** An error that ends a stream already under way, in OpenAI's error shape. */
	[[nodiscard]] static auto error(ApiError const& error) -> http::Event {
		auto body = error_body(error);
		core::invariant(body.size() <= http::max_event_bytes);
		return http::Event{.data = std::move(body)};
	}

private:
	json::Buffer buffer_;
};

}
