<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# API: the openai responses api, field by field

The server speaks the openai **Responses** api only; chat completions does not exist here. The reference is the
newest openai sdk's generated responses types: **openai-python 3.59.0**, git tag `v3.59.0` (commit
`d296ec3be3a98b225d391ef81717b20320f9b598`), sdist `openai-3.59.0.tar.gz` sha256
`e4e57d5c28bed6efe6483c9f61ba59be13fa16edde4b771cdaf5c7ca70bc9ff1`, published 2026-11-24 (the newest release on PyPI on
2026-11-25). Read: `types/responses/*` (create params, every input item param, tools, tool choices, text config,
`Response` and its output items, usage, every stream event), `types/shared_params/{reasoning,reasoning_effort,
custom_tool_input_format}.py`, `resources/responses/{responses,input_items,input_tokens}.py` (routes and their
result types), `types/conversations/*`, and `_streaming.py` (how the sdk reads a stream and its `error` events).

Each row says one of:

- **exact**: honoured exactly as the sdk documents it;
- **neutral `v`**: accepted only at the listed value (absence is always accepted); any other value is refused
  `unsupported_parameter` naming the member;
- **refused `code`**: refused when the request is decoded or validated, before the engine sees it;
- **deprecated**: the sdk marks it deprecated or being replaced: refused `deprecated_parameter`, never translated.

A member the sdk does not define is refused `unknown_parameter`; a value outside a member's vocabulary
`invalid_value`; a wrong JSON kind `invalid_type`; a missing required member `missing_required_parameter`. Statuses:
400 for request faults, 404 `model_not_found` and `response_not_found`, 503 `engine_loading` and `out_of_memory`
(retryable), 500 `response_too_large`. Every error body is `{"error": {"message", "type", "param", "code"}}`.

**D** marks an owner decision; **H** a change engine.http must make before the row is live.

## routes

| route | here | notes |
|---|---|---|
| `POST /v1/responses` | exact | whole, or streamed as the sdk's events **H1** |
| `GET /v1/responses/{id}` | exact for a stored response | query `include` neutral `[]`; `stream=true` (resume a stream) refused `unsupported_parameter`; unknown or evicted id 404 `response_not_found` **H1** |
| `DELETE /v1/responses/{id}` | exact | the sdk reads no body; the server answers `{"id", "object": "response.deleted", "deleted": true}` **H1** |
| `GET /v1/responses/{id}/input_items` | exact | `after`, `limit` (1 to 100, default 20), `order` (`asc`, `desc`, default `desc`); `include` neutral `[]` **H1** |
| `POST /v1/responses/{id}/cancel` | refused `unsupported_parameter` | the sdk cancels only background responses, and `background` is neutral `false` |
| `POST /v1/responses/compact` | refused (no such route: 404 `route_not_found`) | |
| `POST /v1/responses/input_tokens` | refused (no such route) | |
| `/v1/conversations/*` | refused (no such route) | conversations are not implemented; `conversation` is refused on create |
| `GET /v1/models` | exact | one model, `Qwen3.8-27B-FP8` |
| `GET /health`, `GET /status` | not sdk routes | readiness, and the engine's snapshot |

Every response carries `X-Request-Id`.

## `POST /v1/responses`: create params

| member | here | notes |
|---|---|---|
| `model` | exact: `Qwen3.8-27B-FP8` | any other id 404 `model_not_found`; absent is allowed, as the sdk allows it |
| `input` | exact | a string (one user message) or a list of input items (below); absent refused `missing_required_parameter` (there are no stored prompts) |
| `instructions` | exact | a system block before the input. With `previous_response_id` it must equal the stored response's instructions (both absent counts): the continuation is the stored tokens, which carry them, so anything else would be a re-render; otherwise `previous_response_mismatch` **D1** |
| `tools` | see tools | |
| `tool_choice` | see tool choice | absent: `auto` |
| `text.format` | `text`, `json_object`, `json_schema` exact | `json_schema`: `name` required, `schema` required, `description` rendered nowhere (accepted), `strict` at both values (every schema is enforced) |
| `text.verbosity` | neutral `medium` | the model has no verbosity control; `medium` is the sdk's default |
| `reasoning.effort` | `none` (thinking off), `low`, `medium`, `xhigh` exact | the chat template's three efforts; `minimal`, `high`, `max` refused `unsupported_parameter`; absent: thinking on at `xhigh`, as the template defaults |
| `reasoning.summary` | neutral `null` | the engine writes its reasoning text, never a summary |
| `reasoning.generate_summary` | deprecated | |
| `reasoning.context` | neutral `auto`, `all_turns` | past reasoning stays in the context, as the template's `preserve_thinking` default keeps it; `current_turn` refused |
| `reasoning.mode` | neutral `standard` | |
| `max_output_tokens` | exact | 1 to 262,144 (`max_output_tokens_below_one`, `max_output_tokens_above_context`); a prompt that leaves fewer positions is refused `context_length_exceeded` after rendering; reached, the response is `incomplete` with reason `max_output_tokens` |
| `max_tool_calls` | neutral `null` | |
| `parallel_tool_calls` | neutral `true` | |
| `temperature` | exact | 0 to 2 (`temperature_out_of_range`); absent: the model's recommended value for the thinking mode (1.0 thinking, 0.7 not) |
| `top_p` | exact | 0 to 1 (`top_p_out_of_range`); absent: 0.95 thinking, 0.8 not |
| `top_logprobs` | neutral `0` | |
| `stream` | exact | |
| `stream_options.include_obfuscation` | neutral `false` | **D2**: the sdk turns obfuscation on by default; this server never writes the `obfuscation` field, so absent behaves as `false` |
| `store` | exact | default `true`; a stored response lives in a bounded store and is evicted least recently used, so the sdk's 30-day retention is not promised **D3** |
| `previous_response_id` | exact | continues from the stored response's exact token sequence plus the new input rendered after it; unknown or evicted 404 `response_not_found`; a response that ended `incomplete` cannot be continued (`previous_response_incomplete`); `tools`, `instructions` and `reasoning.effort` must match the stored response's (`previous_response_mismatch`, param names the member) |
| `metadata` | exact | at most 16 pairs, keys at most 64 characters, values at most 512; echoed and stored |
| `include` | neutral `[]` | every includable names output this server never makes (`reasoning.encrypted_content`, logprobs, tool results) |
| `truncation` | neutral `disabled` | |
| `background`, `background_options` | neutral `false`, `null` | |
| `conversation` | refused `unsupported_parameter` | |
| `context_management` | neutral `null` | |
| `prompt` | refused `unsupported_parameter` | no stored prompts |
| `prompt_cache_key` | neutral `null` | the prefix cache is automatic and exact; a key would change nothing |
| `prompt_cache_options` | neutral `null` | |
| `prompt_cache_retention` | deprecated | |
| `safety_identifier` | neutral `null` | |
| `user` | deprecated (being replaced) | |
| `service_tier` | neutral `auto`, `default` | |
| `moderation` | neutral `null` | |
| `image`, `access_programs` | neutral `null` | |

There is no `stop`, no `seed`, no `top_k` and no `min_p`: the sdk defines none of them on responses. Every request
draws a fresh seed; top-k (20) and min-p (0) are the model's recommended sampling.

### input items

A turn of the conversation is rendered through the model's chat template exactly as the template renders the
equivalent chat: messages by role, a run of assistant items (reasoning, message, calls) as one assistant turn, call
outputs as tool responses.

| item | here | notes |
|---|---|---|
| message (`EasyInputMessage`, `type` optional) | exact | `role` `user`, `assistant`, `system`, `developer`; `content` a string or parts; `phase` neutral `null` |
| message (`Message`, input) | exact | `status` neutral `completed` |
| message (`ResponseOutputMessage`, assistant, with `id`) | exact | `output_text` parts (`annotations` neutral `[]`, `logprobs` neutral `[]`); `refusal` parts refused; `status` neutral `completed`; `phase` neutral `null` |
| content part `input_text` | exact | `prompt_cache_breakpoint` neutral `null` |
| content parts `input_image`, `input_file` | refused `unsupported_parameter` | text only |
| `function_call` | exact | `arguments` a string holding one JSON object; `call_id`, `name` required; `id` accepted; `status` neutral `completed`; `caller` neutral `direct`; `namespace` neutral absent |
| `function_call_output` | exact | `call_id` required (the sdk marks it optional; there is nothing to match without it); `output` a string or `input_text` parts; `caller`, `name`, `namespace` neutral absent |
| `custom_tool_call` | exact | `call_id`, `name`, `input`; rendered as a call with one string parameter `input` |
| `custom_tool_call_output` | exact | as `function_call_output` |
| `reasoning` | exact | `content` `reasoning_text` parts are the turn's reasoning; `summary` neutral `[]`; `encrypted_content` neutral `null`; `status` neutral `completed`; it must open an assistant turn (`reasoning_without_turn`) |
| `item_reference` | refused `unsupported_parameter` | |
| every other item (`file_search_call`, `computer_call`, `computer_call_output`, `web_search_call`, `tool_search_call`, `tool_search_output`, `additional_tools`, `configuration_update`, `compaction`, `compaction_trigger`, `image_generation_call`, `code_interpreter_call`, `local_shell_call`, `local_shell_call_output`, `shell_call`, `shell_call_output`, `apply_patch_call`, `apply_patch_call_output`, `mcp_list_tools`, `mcp_approval_request`, `mcp_approval_response`, `mcp_call`, `program`, `program_output`) | refused `unsupported_parameter` | |

The template's own rules hold: a system or developer message only first (`system_message_not_first`), at least one
user query (`no_user_query`), an input that is not empty (`input_empty`).

### tools

| tool | here | notes |
|---|---|---|
| `function` | exact | `name` `[a-zA-Z0-9_-]{1,64}` (`invalid_tool_name`), distinct across tools (`duplicate_tool_name`); `parameters` required (an object schema, or `null` for none); `strict` required, at both values (every schema is enforced); `description` rendered; `defer_loading` neutral `false`; `allowed_callers` neutral `["direct"]`; `output_schema` neutral `null`; an unsupported schema feature is refused by the grammar compiler (1.4b) |
| `custom` | exact | `format` `text` or `grammar` (`syntax` `lark` or `regex`, `definition`); reaches the model as a template tool with one string parameter `input`; `defer_loading`, `allowed_callers` as functions |
| `file_search`, `computer`, `computer_use_preview`, `web_search`, `web_search_preview`, `mcp`, `code_interpreter`, `programmatic_tool_calling`, `image_generation`, `local_shell`, `shell`, `namespace`, `tool_search`, `apply_patch` | refused `unsupported_parameter` | |

### tool choice

| choice | here |
|---|---|
| `none`, `auto`, `required` | exact (the grammar forbids, allows or demands calls) |
| `{type: function, name}` | exact; the name must be a function tool's (`unknown_tool`) |
| `{type: custom, name}` | exact; the name must be a custom tool's (`unknown_tool`) |
| `required` or a named tool without tools | refused `tool_choice_without_tools` |
| `allowed_tools`, the hosted-tool types, `mcp`, `apply_patch`, `shell`, `programmatic_tool_calling` | refused `unsupported_parameter` |

## the `Response` object

| member | here |
|---|---|
| `id` | `resp_` + 32 hex digits |
| `object` | `response` |
| `created_at`, `completed_at` | Unix seconds |
| `status` | `completed`, `incomplete`; `in_progress` in the stream's opening events; `failed` never (the engine has no failure output) |
| `error` | `null` |
| `incomplete_details` | `{"reason": "max_output_tokens"}` when the budget or the context ran out, else `null` |
| `instructions`, `metadata`, `model`, `previous_response_id`, `max_output_tokens`, `store` | as requested |
| `tools`, `tool_choice` | as requested (`tool_choice` `auto` when absent) |
| `temperature`, `top_p` | the values used |
| `text` | `{"format": ..., "verbosity": "medium"}` |
| `reasoning` | `{"effort": ..., "summary": null}` |
| `parallel_tool_calls` `true`, `truncation` `"disabled"`, `background` `false`, `top_logprobs` `0`, `service_tier` `"default"` | fixed |
| `output` | the items below, in order |
| `usage` | `input_tokens`, `input_tokens_details.cached_tokens` (served from the prefix cache, including a continuation's stored tokens), `input_tokens_details.cache_write_tokens` `0` (no explicit cache breakpoints exist), `output_tokens`, `output_tokens_details.reasoning_tokens`, `total_tokens` |
| `conversation`, `prompt`, `prompt_cache_*`, `safety_identifier`, `user`, `moderation`, `tool_usage`, `image`, `input_transformations`, `access_programs` | `null` or omitted |

### output items

| item | here |
|---|---|
| `reasoning` | `{id: "rs_…", type, summary: [], content: [{type: "reasoning_text", text}], status}` |
| `message` | `{id: "msg_…", type, role: "assistant", status, content: [{type: "output_text", text, annotations: [], logprobs: []}]}`; `phase` omitted |
| `function_call` | `{id: "fc_…", type, call_id: "call_…", name, arguments, status}` |
| `custom_tool_call` | `{id: "ctc_…", type, call_id: "call_…", name, input}` |
| refusals, annotations, every hosted-tool item | never |

## stream events

Each is one server-sent event whose data is the event object; `sequence_number` counts from 0. A stream opens only
once the engine accepts the request, so a refusal is a whole 4xx. **H2**: the terminal events carry the whole
`Response`, which can outgrow the http layer's per-event bound.

| event | here |
|---|---|
| `response.created`, `response.in_progress` | the response with `status` `in_progress` and empty `output` |
| `response.output_item.added`, `.done` | each output item, at its `output_index` |
| `response.content_part.added`, `.done` | the message's `output_text` part, the reasoning item's `reasoning_text` part |
| `response.output_text.delta`, `.done` | `logprobs` `[]` |
| `response.reasoning_text.delta`, `.done` | |
| `response.function_call_arguments.delta`, `.done` | the arguments arrive whole from the engine: one delta (or as many as one event's bound needs) |
| `response.custom_tool_call_input.delta`, `.done` | likewise |
| `response.completed`, `response.incomplete` | the final response |
| `error` | `{type, code, message, param, sequence_number}`: a client that read too slowly (`stream_backlog`); the sdk raises it |
| `response.queued`, `response.failed`, refusal, audio, summary, annotation, and every hosted-tool event | never |

## decisions for the owner

- **D1** instructions with `previous_response_id`: the sdk drops the previous response's instructions; the stored
  token sequence carries them. Proposal (implemented): a continuation must repeat them exactly (then the result is
  the same sequence the sdk would build), else `previous_response_mismatch`.
- **D2** `include_obfuscation` defaults to on in the sdk; this server never obfuscates.
- **D3** stored responses are evicted lru from a bounded store, not kept 30 days.

## changes engine.http must make

- **H1** the routes: `POST /v1/responses` with a body; `GET` and `DELETE /v1/responses/{id}`; `GET
  /v1/responses/{id}/input_items` with its query; `POST /v1/responses/{id}/cancel`; each carrying the path id and
  query as owned text; `/v1/chat/completions` deleted.
- **H2** an event larger than `max_event_bytes`: the terminal `response.completed` and `response.incomplete` carry the
  whole response (bounded by `max_response_body_bytes`); the per-connection window must admit one such event.
