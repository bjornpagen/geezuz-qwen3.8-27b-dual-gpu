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
`invalid_value`; a wrong JSON kind `invalid_type`; a missing required member `missing_required_parameter`; a body that
is not one JSON document within the request's bounds `invalid_json`; members that exclude each other
`conflicting_parameters`; a list or a text past its bound `too_many_items` and `string_too_long`. Statuses: 400 for
request faults; 404 `model_not_found` and `response_not_found`; 503 `engine_loading`, `engine_busy` and
`out_of_memory` (retryable), and `engine_failed` (not: the engine serves nothing more until it is restarted); 500
`response_too_large`, and `server_error` for a request the engine failed under. A request whose grammar outgrows the
engine's bounds while its reply is decoded ends there: a whole response is refused 400 `grammar_overrun`, a stream
ends with that `error` event; neither is stored. Every error body is `{"error": {"message", "type", "param", "code"}}`,
`type` `invalid_request_error` for a 4xx and `server_error` for a 5xx.

**D** marks an owner decision.

Live today: engine.openai serves every route below over engine.http: it decodes and validates each create request
whole at the boundary (engine.json's decoder, then the served model, tools, tool choice, the output grammar, budget,
sampling, the chat template's rules and the stored response a continuation names), hands the engine a typed submission
carrying its compiled grammar, and answers whole or streamed from the engine's typed outputs; engine.store holds stored
responses. The request's tools, tool choice, `text.format` and reasoning are compiled by engine.grammar during
validation, so a schema or grammar feature the engine does not implement is a named 400 (below) before the engine sees
the request, and every reply decodes under its grammar: calls always parse, arguments always match their schema, and
the tool choice and format are honoured. The engine's front (`openai::Front`) renders each submission with the chat
format, starts the grammar's matcher, and runs the reply's tokens through the reply parser.

### readiness and bounds

The service holds the engine's readiness once: loading, failed (with a reason), or ready. Until it is ready, `GET
/health` answers 503 `{"status": "loading" | "failed", "reason"}` and every other route is refused 503
`engine_loading` or `engine_failed`; ready, `/health` answers 200 `{"status": "ok"}`. Whoever starts the engine posts
that it serves, or that it could not start (no driver on this machine, a card that failed its checks); an engine that
fails while serving fails closed: every stream under way ends with `response.failed`, every whole response with a 500
`server_error`, and new requests are refused 503 `engine_failed`.

The engine holds at most half its intake's capacity of requests at once, counting each from its submission to its
terminal output (a refusal, its finish, or its answer to a cancel): a create past that is refused 503 `engine_busy`. A
streaming client that leaves 4 MiB of events unread ends its stream with an `error` event `stream_backlog`, and its
request is cancelled; a client that disconnects cancels its request.

## routes

| route | here | notes |
|---|---|---|
| `POST /v1/responses` | exact | whole, or streamed as the sdk's events |
| `GET /v1/responses/{id}` | exact for a stored response | query `include` neutral `[]` (any `include` refused `unsupported_parameter`); `stream` neutral `false` (`stream=true`, resuming a stream, refused `unsupported_parameter`), as are `starting_after` and `include_obfuscation=true`; another parameter `unknown_parameter`; unknown or evicted id 404 `response_not_found` |
| `DELETE /v1/responses/{id}` | exact | the sdk reads no body; the server answers `{"id", "object": "response.deleted", "deleted": true}`; unknown 404 `response_not_found` |
| `GET /v1/responses/{id}/input_items` | exact | `after` (an item id of this response, else `invalid_value`), `limit` (1 to 100, default 20), `order` (`asc`, `desc`, default `desc`); a parameter given twice `invalid_value`; `include` neutral `[]` |
| `POST /v1/responses/{id}/cancel` | refused `unsupported_parameter`, param `background` | the sdk cancels only background responses, and `background` is neutral `false` |
| `POST /v1/responses/compact`, `POST /v1/responses/input_tokens` | refused 405 `method_not_allowed` | the paths match `/v1/responses/{id}`, which takes `GET` and `DELETE` (the `Allow` field lists them); neither route is implemented |
| `/v1/conversations/*` | refused (no such route: 404 `route_not_found`) | conversations are not implemented; `conversation` is refused on create |
| `GET /v1/models` | exact | one model, `Qwen3.8-27B-FP8` |
| `GET /health`, `GET /status` | not sdk routes | readiness (above), and the engine's latest snapshot: `{"engine": null}` before it posted one, else `{"engine": {"slots", "queued", "kv_pool", "counters"}}`, each running request named by its request id |

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
| `reasoning.effort` | `none` (thinking off), `low`, `medium`, `xhigh` exact | the chat template's three efforts; `minimal`, `high`, `max` refused `unsupported_parameter`, the message listing the four served values **D4**; absent: thinking on at `xhigh`, as the template defaults |
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
| `stream_options.include_obfuscation` | neutral `false` | **D2**: this server never writes the `obfuscation` field, so `true` is refused `unsupported_parameter` and `false` and absent are accepted (absent means off here, though the sdk defaults it on) |
| `store` | exact | default `true`; stored responses live in memory, bounded (4096 responses, 2 GiB) and evicted least recently used, and are lost on restart: the deviation from the sdk's 30-day retention **D3** |
| `previous_response_id` | exact | continues from the stored response's exact token sequence plus the new input rendered after it; unknown, evicted, unstored (`store: false`) or not an id this server issues: 404 `response_not_found`; a response that ended `incomplete` cannot be continued (`previous_response_incomplete`); `tools`, `instructions` and `reasoning.effort` must equal the stored response's (`previous_response_mismatch`, param names the member) **D1**; a system or developer message in the new input is `system_message_not_first` |
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
| `reasoning` | exact | `content` `reasoning_text` parts are the turn's reasoning; `summary` neutral `[]`; `encrypted_content` neutral `null`; `status` neutral `completed`; it opens an assistant turn, which the following assistant message and calls of the same run join |
| `item_reference` | refused `unsupported_parameter` | |
| every other item (`file_search_call`, `computer_call`, `computer_call_output`, `web_search_call`, `tool_search_call`, `tool_search_output`, `additional_tools`, `configuration_update`, `compaction`, `compaction_trigger`, `image_generation_call`, `code_interpreter_call`, `local_shell_call`, `local_shell_call_output`, `shell_call`, `shell_call_output`, `apply_patch_call`, `apply_patch_call_output`, `mcp_list_tools`, `mcp_approval_request`, `mcp_approval_response`, `mcp_call`, `program`, `program_output`) | refused `unsupported_parameter` | |

The template's own rules hold: a system or developer message only first (`system_message_not_first`), at least one
user query (`no_user_query`, which an empty input also fails).

### tools

| tool | here | notes |
|---|---|---|
| `function` | exact | `name` `[a-zA-Z0-9_-]{1,64}` (`invalid_tool_name`), distinct across tools (`duplicate_tool_name`); `parameters` required (an object schema, or `null` for none); `strict` required, at both values (every schema is enforced); `description` rendered; `defer_loading` neutral `false`; `allowed_callers` neutral `["direct"]`; `output_schema` neutral `null`; a definition longer than the 4 MiB the chat format reads back as one JSON text `tool_too_large`; an unsupported schema feature is refused by the grammar compiler with its own code (see the output grammar) |
| `custom` | exact | `format` `text` or `grammar` (`syntax` `lark` or `regex`, `definition`); reaches the model as a template tool with one string parameter `input`; `defer_loading`, `allowed_callers` as functions; its name and definition bounded as a function's |
| `file_search`, `computer`, `computer_use_preview`, `web_search`, `web_search_preview`, `mcp`, `code_interpreter`, `programmatic_tool_calling`, `image_generation`, `local_shell`, `shell`, `namespace`, `tool_search`, `apply_patch` | refused `unsupported_parameter` | |

### tool choice

| choice | here |
|---|---|
| `none`, `auto`, `required` | exact (the grammar forbids, allows or demands calls) |
| `{type: function, name}` | exact; the name must be a function tool's (`unknown_tool`) |
| `{type: custom, name}` | exact; the name must be a custom tool's (`unknown_tool`) |
| `required` or a named tool without tools | refused `tool_choice_without_tools` |
| `allowed_tools`, the hosted-tool types, `mcp`, `apply_patch`, `shell`, `programmatic_tool_calling` | refused `unsupported_parameter` |

## the output grammar

Every request decodes under a grammar compiled from its tools, tool choice, `text.format` and reasoning effort
(`parallel_tool_calls` is always `true`), during validation, under fixed bounds. What the compiler does not implement,
or what passes a bound, is a 400 with its own `code`; `param` names the member (`tools[i].parameters` for a function's
schema, `tools[i].format` for a custom tool's regex, `tools` for a custom tool's lark grammar, `text.format.schema` for
the format's schema, `tool_choice`), `null` for a bound of the grammar as a whole, and the message gives the compiler's
location in the request.

| code | refused |
|---|---|
| `schema_shape` | a schema that is neither an object nor a boolean |
| `schema_keyword` | a keyword the engine does not implement |
| `schema_value` | a keyword's value of the wrong shape (a `required` that is no list of strings, a negative `maxLength`) |
| `schema_type` | a `type` naming no JSON type |
| `schema_reference` | a `$ref` that is not `#`, `#/$defs/<name>` or `#/definitions/<name>`, or names no definition |
| `schema_format` | a string `format` the engine does not implement |
| `schema_one_of` | `oneOf` whose choices the engine cannot prove disjoint |
| `schema_multiple_of` | `multipleOf` |
| `schema_bound` | a numeric bound the engine cannot hold exactly (an exclusive real bound past 15 significant digits) |
| `grammar_empty` | a grammar that admits no output: a schema no value satisfies, a lark grammar with no finite derivation |
| `schema_open_argument` | a tool argument that may hold an object with additional properties (the chat format refuses repeated keys) |
| `schema_string_union` | a tool argument whose type list mixes `string` with a type the chat format would read the text as |
| `schema_recursive_argument` | a tool argument's recursive schema |
| `tool_parameters` | a function's `parameters` that is not an object schema |
| `tool_name`, `tool_duplicate`, `tool_unknown` | a tool or parameter name the chat format cannot write, two tools of one name, a choice naming no tool (the api's own checks refuse the last two first, as `duplicate_tool_name` and `unknown_tool`) |
| `lark_syntax`, `lark_undefined`, `lark_recursive_terminal`, `lark_no_start`, `lark_priority`, `lark_template`, `lark_directive`, `lark_token`, `lark_option`, `lark_operator` | a custom tool's lark grammar: text that does not parse, a rule or terminal undefined or defined twice, a recursive terminal, no `start`, priorities, templates, a directive other than `%import common`, a special-token or grammar reference, a lexeme option other than `lazy` and `suffix`, `&` or `~` outside a terminal |
| `regex_unexpected_end`, `regex_unbalanced_group`, `regex_nothing_to_repeat`, `regex_bad_repetition`, `regex_repetition_too_large`, `regex_bad_escape`, `regex_bad_class`, `regex_bad_character`, `regex_too_deep`, `regex_too_large`, `regex_lookaround`, `regex_lazy_quantifier`, `regex_anchor`, `regex_word_boundary`, `regex_backreference`, `regex_unicode_property`, `regex_unicode_case`, `regex_class_syntax`, `regex_group_syntax` | a regular expression (a custom tool's regex format, or a schema's `pattern`), each fault by name: malformed, or lookaround, lazy quantifiers, anchors without an exact meaning, word boundaries, backreferences, Unicode property classes, case folding past ascii, class set syntax and group flags, which the engine does not implement |
| `schema_too_large` | a schema text past 256 KiB or nested past 32 levels, a `pattern` past 16 KiB, or a `minLength`/`maxLength` past 100,000 |
| `schema_too_many` | more than 5,000 properties or 2,000 enum and const values across the request, or too many definitions |
| `lark_too_large` | a lark grammar past 64 KiB or 1,000 rules and terminals; a custom tool's regex past 16 KiB |
| `automaton_too_large` | a lexeme whose automaton passes 10,000 states, or all together 120,000 |
| `grammar_too_large` | a grammar past 20,000 lexemes, 100,000 rules, 400,000 symbols or 2^20 expression nodes |

A string's `minLength` and `maxLength` are counted while it decodes, beside its automaton, so they reach 100,000 at no
cost in states, in a function's arguments and a `json_schema` format alike, a string-or-`null` type included. A
string with a `pattern` or `format` has its bound written into its automaton instead, whose states grow with it: as
measured here, `^[a-z]+$` compiles up to `maxLength` 9,996 at the top level and 9,984 as a function argument; past
that the request is refused `automaton_too_large`. Patterns and enums of many long values meet the same bound.

A request whose grammar outgrows the matcher's own bounds while the reply decodes (`grammar_overrun`) is described
above. A caller without stored responses (the cli) refuses `previous_response_id` with `previous_response_unavailable`.

## the `Response` object

| member | here |
|---|---|
| `id` | `resp_` + 32 hex digits |
| `object` | `response` |
| `created_at`, `completed_at` | Unix seconds |
| `status` | `completed`, `incomplete`; `in_progress` in the stream's opening events; `failed` only in a stream's `response.failed`, when the engine failed under it (such a response is never stored) |
| `error` | `null`; in `response.failed`, `{"code": "server_error", "message"}` with the engine's reason |
| `incomplete_details` | `{"reason": "max_output_tokens"}` when the budget, the context or the pool ran out, else `null`; an item the end cut off has `status` `incomplete` |
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
once the engine accepts the request, so a refusal is a whole 4xx. The terminal events carry the whole `Response`; an
event longer than one http event may be ends the stream with an `error` event `response_too_large` instead. While
the client reads slower than the model writes, the pending deltas of one item merge into one event.

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
| `response.failed` | the response as it stood, `status` `failed`, when the engine failed under the stream |
| `error` | `{type, code, message, param, sequence_number}`: a client that read too slowly (`stream_backlog`, its request cancelled), a response that outgrew one event or one body (`response_too_large`), or a grammar that outgrew its bounds while decoding (`grammar_overrun`); the sdk raises it |
| `response.queued`, refusal, audio, summary, annotation, and every hosted-tool event | never |

## owner decisions

- **D1** instructions with `previous_response_id`: the sdk drops the previous response's instructions; the stored
  token sequence carries them. A continuation must repeat them exactly, and its tools and effort too (then the result
  is the same sequence the sdk would build), else `previous_response_mismatch` naming the member.
- **D2** `include_obfuscation` defaults to on in the sdk; this server never obfuscates: `true` is refused by name,
  `false` and absent are accepted.
- **D3** stored responses live in memory, bounded and evicted least recently used, and are lost on restart; they are
  not kept 30 days. Disk is experiment 2.9.
- **D4** `reasoning.effort` honours exactly the template's levels (`none` = thinking off, `low`, `medium`, `xhigh`)
  and refuses the sdk's others by name, listing the four.
