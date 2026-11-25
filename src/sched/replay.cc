// SPDX-License-Identifier: AGPL-3.0-only
export module engine.sched:replay;

import std;
import engine.core;
import engine.sha256;
import engine.tokenizer;
import :request;

namespace engine::sched {

/** Finished turns kept until a later prompt meets them: only a conversation's next request is expected to. */
export inline constexpr Count replay_recent = 64;

/** Tokens the stored (rendered, generated) pairs hold at most, keys included; the least recently used go first. */
export inline constexpr Count replay_tokens = Count{1} << 22U;

/** Positions of the prefix of a prompt whose last tokens a finished turn checks before hashing the whole prefix. */
inline constexpr Count replay_tail = 16;

/** A prompt after replay, with its marks moved to where their blocks now start. */
export struct Replayed {
	std::vector<Token> prompt;
	Marks marks;

	[[nodiscard]] auto operator==(Replayed const&) const -> bool = default;
};

/** A request that ended its turn, waiting for the prompt of its conversation's next turn. */
struct Turn {
	Count length;
	std::vector<Token> tail;
	sha256::Digest digest;
	/** The prompt from its last `<|im_start|>`: the generation prompt the turn followed. */
	std::vector<Token> generation_prompt;
	/** The tokens generated, the last one `<|im_end|>`, as the chat format closes a turn. */
	std::vector<Token> generated;
};

struct Key {
	std::vector<Token> generation_prompt;
	std::vector<Token> rendered;

	[[nodiscard]] auto operator<=>(Key const&) const = default;
	[[nodiscard]] auto operator==(Key const&) const -> bool = default;
};

struct Stored {
	std::vector<Token> generated;
	std::uint64_t used;
};

/** Tokens [start, end) of a prompt become `replacement`. */
struct Splice {
	Count start;
	Count end;
	std::vector<Token> replacement;
};

[[nodiscard]] auto digest_of(std::span<Token const> tokens) -> sha256::Digest {
	return sha256::digest(std::as_bytes(tokens));
}

[[nodiscard]] auto ascii_space(char letter) -> bool {
	return letter == ' ' || letter == '\t' || letter == '\n' || letter == '\r' || letter == '\f' || letter == '\v';
}

/**
 * Generated assistant turns, put back into later prompts in place of their rendering. A client sends a turn back as
 * parsed text and the chat format renders it again; whitespace the parser dropped and markup it normalized (a tool
 * call opened inside thinking gets a `</think>`) make that rendering differ from the tokens the model generated, and
 * on this hybrid model a prompt that differs from the cache once is prefilled from there on, because the deltanet
 * state cannot be cut back. A finished turn waits for the next prompt that starts with its own prompt; when that
 * prompt renders the turn with the same text (whitespace and that markup aside), the generated tokens replace the
 * rendering there and, through the stored (rendered, generated) pair, in every later prompt of the conversation. A
 * turn whose text the client changed keeps its rendering.
 */
export class Replay {
public:
	/** A request ended its turn: `prompt` as it ran, `generated` its tokens through the one that ended the turn. */
	auto finished(std::span<Token const> prompt, std::span<Token const> generated) -> void {
		core::invariant(!generated.empty() && ends_turn(generated.back()));
		auto const start = std::ranges::find_last(prompt, im_start);
		if (start.empty()) {
			return;
		}
		auto turn = Turn{
		    .length = prompt.size(),
		    .tail = {prompt.end() - core::narrow<std::ptrdiff_t>(std::min(replay_tail, prompt.size())), prompt.end()},
		    .digest = digest_of(prompt),
		    .generation_prompt = {start.begin(), prompt.end()},
		    .generated = {generated.begin(), generated.end()},
		};
		turn.generated.back() = im_end;
		pending_.push_back(std::move(turn));
		if (pending_.size() > replay_recent) {
			pending_.pop_front();
		}
	}

	/** The prompt with every stored turn replayed, and the finished turn it continues, if its text is unchanged. */
	[[nodiscard]] auto apply(tokenizer::Tokenizer const& tokenizer, std::vector<Token> prompt, Marks marks) -> Replayed {
		auto const splices = substitutions(prompt);
		auto replayed = splice(Replayed{.prompt = std::move(prompt), .marks = marks}, splices);
		auto& tokens = replayed.prompt;
		for (auto at = pending_.size(); at > 0; --at) {
			if (!continues(tokens, pending_[at - 1])) {
				continue;
			}
			auto const taken = pending_[at - 1];
			pending_.erase(pending_.begin() + core::narrow<std::ptrdiff_t>(at - 1));
			auto const end = std::ranges::find(std::span{tokens}.subspan(taken.length), im_end);
			if (end == std::span{tokens}.subspan(taken.length).end()) {
				break;
			}
			auto const close = taken.length + core::narrow<Count>(end - std::span{tokens}.subspan(taken.length).begin()) + 1;
			auto const rendered = std::vector<Token>{tokens.begin() + core::narrow<std::ptrdiff_t>(taken.length),
			                                         tokens.begin() + core::narrow<std::ptrdiff_t>(close)};
			if (rendered == taken.generated || normal(tokenizer, rendered) != normal(tokenizer, taken.generated)) {
				break;
			}
			store(Key{.generation_prompt = taken.generation_prompt, .rendered = rendered}, taken.generated);
			replayed = splice(std::move(replayed), std::array{Splice{.start = taken.length, .end = close, .replacement = taken.generated}});
			break;
		}
		return replayed;
	}

	/** Tokens the stored pairs hold, keys included. */
	[[nodiscard]] auto stored_tokens() const -> Count {
		return stored_tokens_;
	}

	[[nodiscard]] auto pending() const -> Count {
		return pending_.size();
	}

private:
	static constexpr Token im_start = tokenizer::id(tokenizer::Added::im_start);
	static constexpr Token im_end = tokenizer::id(tokenizer::Added::im_end);

	[[nodiscard]] static auto continues(std::span<Token const> tokens, Turn const& turn) -> bool {
		if (turn.length >= tokens.size()) {
			return false;
		}
		auto const prefix = tokens.first(turn.length);
		return std::ranges::equal(prefix.last(turn.tail.size()), turn.tail) && digest_of(prefix) == turn.digest;
	}

	/** The text of a turn with whitespace and the markup a parser normalizes taken out. */
	[[nodiscard]] static auto normal(tokenizer::Tokenizer const& tokenizer, std::span<Token const> tokens) -> std::string {
		auto text = tokenizer.decode(tokens);
		// Every id here is one the tokenizer produced or the sampler drew, and neither yields a padding id.
		core::invariant(text.has_value());
		std::erase_if(*text, ascii_space);
		for (auto const markup : {std::string_view{"<think>"}, std::string_view{"</think>"}, std::string_view{"<tool_call>"}}) {
			for (auto found = text->find(markup); found != std::string::npos; found = text->find(markup, found)) {
				text->erase(found, markup.size());
			}
		}
		return std::move(*text);
	}

	/** Each rendered turn of a stored pair in the prompt, to be replaced by its generated tokens, in prompt order. */
	[[nodiscard]] auto substitutions(std::span<Token const> tokens) -> std::vector<Splice> {
		auto splices = std::vector<Splice>{};
		if (pairs_.empty()) {
			return splices;
		}
		auto at = Count{0};
		for (auto index = Count{0}; index < tokens.size(); ++index) {
			if (tokens[index] != im_start || index < at) {
				continue;
			}
			for (auto const& [generation_prompt, pairs] : prompts_) {
				if (!std::ranges::starts_with(tokens.subspan(index), generation_prompt)) {
					continue;
				}
				auto const start = index + generation_prompt.size();
				auto const end = std::ranges::find(tokens.subspan(start), im_end);
				if (end == tokens.subspan(start).end()) {
					break;
				}
				auto const close = start + core::narrow<Count>(end - tokens.subspan(start).begin()) + 1;
				auto const found = pairs_.find(Key{
				    .generation_prompt = generation_prompt,
				    .rendered = {tokens.begin() + core::narrow<std::ptrdiff_t>(start), tokens.begin() + core::narrow<std::ptrdiff_t>(close)},
				});
				if (found != pairs_.end()) {
					clock_ += 1;
					found->second.used = clock_;
					splices.push_back({.start = start, .end = close, .replacement = found->second.generated});
					at = close;
				}
				break;
			}
		}
		return splices;
	}

	/** Applies `splices` (disjoint, in order) to the prompt and moves each mark past a splice by its change in length. */
	[[nodiscard]] static auto splice(Replayed replayed, std::span<Splice const> splices) -> Replayed {
		if (splices.empty()) {
			return replayed;
		}
		auto const& tokens = replayed.prompt;
		auto out = std::vector<Token>{};
		auto at = Count{0};
		for (auto const& each : splices) {
			out.insert(out.end(), tokens.begin() + core::narrow<std::ptrdiff_t>(at), tokens.begin() + core::narrow<std::ptrdiff_t>(each.start));
			out.insert(out.end(), each.replacement.begin(), each.replacement.end());
			at = each.end;
		}
		out.insert(out.end(), tokens.begin() + core::narrow<std::ptrdiff_t>(at), tokens.end());
		auto const moved = [&](std::optional<Count> mark) {
			return mark.transform([&](Count position) {
				auto shifted = position;
				for (auto const& each : splices) {
					// A mark is the offset of an `<|im_start|>`, and a replaced turn runs from after its generation
					// prompt through its `<|im_end|>`, so no mark falls inside one.
					core::invariant(position <= each.start || position >= each.end);
					if (position >= each.end) {
						shifted = shifted + each.replacement.size() - (each.end - each.start);
					}
				}
				return shifted;
			});
		};
		return Replayed{
		    .prompt = std::move(out),
		    .marks =
		        {
		            .system_end = moved(replayed.marks.system_end),
		            .last_user_start = moved(replayed.marks.last_user_start),
		            .generation_start = moved(replayed.marks.generation_start),
		        },
		};
	}

	[[nodiscard]] static auto weight(Key const& key, std::span<Token const> generated) -> Count {
		return key.generation_prompt.size() + key.rendered.size() + generated.size();
	}

	auto store(Key key, std::span<Token const> generated) -> void {
		if (weight(key, generated) > replay_tokens) {
			return;
		}
		clock_ += 1;
		if (auto const found = pairs_.find(key); found != pairs_.end()) {
			stored_tokens_ -= weight(found->first, found->second.generated);
			found->second = Stored{.generated = {generated.begin(), generated.end()}, .used = clock_};
			stored_tokens_ += weight(found->first, generated);
		} else {
			stored_tokens_ += weight(key, generated);
			prompts_[key.generation_prompt] += 1;
			pairs_.emplace(std::move(key), Stored{.generated = {generated.begin(), generated.end()}, .used = clock_});
		}
		while (stored_tokens_ > replay_tokens) {
			auto const oldest = std::ranges::min_element(pairs_, {}, [](auto const& pair) {
				return pair.second.used;
			});
			stored_tokens_ -= weight(oldest->first, oldest->second.generated);
			auto const uses = prompts_.find(oldest->first.generation_prompt);
			uses->second -= 1;
			if (uses->second == 0) {
				prompts_.erase(uses);
			}
			pairs_.erase(oldest);
		}
	}

	std::deque<Turn> pending_;
	std::map<Key, Stored> pairs_;
	/** The generation prompts of the stored pairs, with how many pairs have each. */
	std::map<std::vector<Token>, Count> prompts_;
	Count stored_tokens_ = 0;
	std::uint64_t clock_ = 0;
};

}
