// SPDX-License-Identifier: AGPL-3.0-only
//
// geezuz-ref: the cpu reference and the verification harness. Every computation runs the reference under the sm_86
// tensor core's accumulation, the engine's own arithmetic; only `noise-floor` also runs the ieee one, because measuring
// the difference between the two is what it is for.
//
//   geezuz-ref logits <checkpoint> <out.dump> <input>              teacher-forced dump of every row
//   geezuz-ref greedy <checkpoint> <out.dump> <count> <input>      greedy continuation, dumped
//   geezuz-ref noise-floor <checkpoint> <input>                    the rulers between sm_86 and ieee accumulation
//   geezuz-ref rulers <a.dump> <b.dump>                            top-1 agreement, kl and perplexity
//   geezuz-ref decode-ruler <checkpoint> <run.dump>                a run's chosen tokens against the reference
//   geezuz-ref golden record <checkpoint> <golden> <prompts> <reason>
//   geezuz-ref golden compare <golden> <candidate>
//
// <input> is `ids <id>...` or `text <file>` (plain text, special-token spellings read as characters). A prompts file
// holds one prompt per line: a name without spaces, one space, the text.
import std;
import engine.core;
import engine.file;
import engine.file_write;
import engine.model;
import engine.ref;
import engine.sha256;
import engine.tokenizer;

namespace {

namespace core = engine::core;
namespace ref = engine::ref;
namespace model = engine::model;

using model::Count;
using ref::Token;

/** The subcommands; nothing else selects what the program does. */
enum class Command : std::uint8_t {
	Logits,
	Greedy,
	NoiseFloor,
	Rulers,
	DecodeRuler,
	GoldenRecord,
	GoldenCompare,
};

/** Rows fed to the trunk at once: batch invariance makes any piece size compute the same bits. */
constexpr Count piece_rows = 16;

using Arguments = std::span<std::string_view const>;

[[nodiscard]] auto command_of(Arguments words) -> std::optional<Command> {
	if (words.empty()) {
		return std::nullopt;
	}
	auto const& first = words[0];
	if (first == "golden" && words.size() >= 2) {
		return words[1] == "record" ? std::optional{Command::GoldenRecord} : words[1] == "compare" ? std::optional{Command::GoldenCompare} : std::nullopt;
	}
	constexpr auto named = std::array{
	    std::pair{std::string_view{"logits"}, Command::Logits},       std::pair{std::string_view{"greedy"}, Command::Greedy},
	    std::pair{std::string_view{"noise-floor"}, Command::NoiseFloor}, std::pair{std::string_view{"rulers"}, Command::Rulers},
	    std::pair{std::string_view{"decode-ruler"}, Command::DecodeRuler},
	};
	auto const found = std::ranges::find(named, first, &std::pair<std::string_view, Command>::first);
	return found == named.end() ? std::nullopt : std::optional{found->second};
}

/** A failure the program reports before it exits non-zero. */
struct Failure {
	std::string message;
};

template<class T>
using Outcome = std::expected<T, Failure>;

[[nodiscard]] auto failure(std::string message) -> std::unexpected<Failure> {
	return std::unexpected(Failure{.message = std::move(message)});
}

[[nodiscard]] auto read_bytes(std::string_view path) -> Outcome<std::vector<std::byte>> {
	auto const mapping = engine::file::Mapping::open(std::filesystem::path{path});
	if (!mapping) {
		return failure(std::format("cannot read {}", path));
	}
	auto const bytes = mapping->bytes();
	return std::vector<std::byte>{bytes.begin(), bytes.end()};
}

[[nodiscard]] auto read_text(std::string_view path) -> Outcome<std::string> {
	return read_bytes(path).transform([](std::vector<std::byte> const& bytes) {
		auto text = std::string{};
		for (auto const byte : bytes) {
			text.push_back(std::to_integer<char>(byte));
		}
		return text;
	});
}

[[nodiscard]] auto write_bytes(std::string_view path, std::span<std::byte const> bytes) -> Outcome<void> {
	auto output = engine::file_write::Output::create(std::filesystem::path{path}, bytes.size());
	if (!output || !output->write(0, bytes) || !std::move(*output).commit()) {
		return failure(std::format("cannot write {}", path));
	}
	return {};
}

[[nodiscard]] auto encode_text(std::string_view text) -> Outcome<std::vector<Token>> {
	auto const tokenizer = engine::tokenizer::load();
	if (!tokenizer) {
		return failure("the built-in tokenizer does not load");
	}
	auto const pieces = std::array{engine::tokenizer::Piece{engine::tokenizer::Text{.utf8 = text, .spelling = engine::tokenizer::Spelling::Characters}}};
	auto ids = tokenizer->encode(pieces);
	if (!ids) {
		return failure("the text is not utf-8");
	}
	if (ids->empty()) {
		return failure("the text encodes to no tokens");
	}
	return std::move(*ids);
}

/** A decimal count, all of `text`. */
[[nodiscard]] auto whole_number(std::string_view text) -> std::optional<Count> {
	auto value = Count{0};
	auto const read = std::from_chars(text.data(), std::to_address(text.end()), value);
	if (text.empty() || std::make_error_code(read.ec) || read.ptr != std::to_address(text.end())) {
		return std::nullopt;
	}
	return value;
}

/** `ids <id>...` or `text <file>`. */
[[nodiscard]] auto input_of(Arguments words) -> Outcome<std::vector<Token>> {
	if (words.size() >= 2 && words[0] == "ids") {
		auto ids = std::vector<Token>{};
		for (auto const word : words.subspan(1)) {
			auto const id = whole_number(word);
			if (!id || *id >= model::sampled_vocabulary) {
				return failure(std::format("{} is not a token id", word));
			}
			ids.push_back(core::narrow<Token>(*id));
		}
		return ids;
	}
	if (words.size() == 2 && words[0] == "text") {
		return read_text(words[1]).and_then([](std::string const& text) {
			return encode_text(text);
		});
	}
	return failure("the input is `ids <id>...` or `text <file>`");
}

[[nodiscard]] auto open_model(std::string_view directory) -> Outcome<ref::Model> {
	auto const started = std::chrono::steady_clock::now();
	auto opened = ref::Model::open(std::filesystem::path{directory});
	if (!opened) {
		return failure(std::format("the checkpoint is refused: failure {} at {}", std::to_underlying(opened.error().failure), opened.error().subject));
	}
	auto const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	std::println("checkpoint mapped and checked in {:.1f} s, {} threads", seconds, opened->pool().threads());
	return std::move(*opened);
}

/** A teacher-forced run's every row: its logits summarized, and their canonical hash. */
struct Run {
	std::vector<ref::DumpRow> rows;
	std::vector<float> next;
	engine::sha256::Digest digest;
};

template<ref::Accumulation A>
[[nodiscard]] auto teacher_forced(ref::Model& model, std::span<Token const> tokens) -> Run {
	auto sequence = ref::Sequence::empty();
	auto hasher = ref::LogitsHasher{};
	auto run = Run{};
	auto const started = std::chrono::steady_clock::now();
	for (auto first = Count{0}; first < tokens.size(); first += piece_rows) {
		auto const piece = tokens.subspan(first, std::min(piece_rows, tokens.size() - first));
		auto const logits = ref::extend<A>(model, sequence, piece);
		hasher.update(logits);
		for (auto row = Count{0}; row < piece.size(); ++row) {
			run.rows.push_back(ref::summarize(logits.row(row)));
			if (first + row + 1 < tokens.size()) {
				run.next.push_back(logits.row(row)[tokens[first + row + 1]]);
			}
		}
	}
	auto const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	std::println("{}: {} rows in {:.1f} s, {:.2f} s per row", A::name, tokens.size(), seconds, seconds / static_cast<double>(tokens.size()));
	run.digest = std::move(hasher).finish();
	return run;
}

[[nodiscard]] auto dump_from(ref::Source source, Count prompt, std::span<Token const> tokens, Run run) -> ref::Dump {
	return {.source = source, .prompt = prompt, .tokens = {tokens.begin(), tokens.end()}, .next = std::move(run.next), .rows = std::move(run.rows)};
}

auto print_comparison(ref::Comparison const& compared) -> void {
	std::println("rows {}", compared.rows);
	std::println("top-1 agreement {:.6f}", compared.top1_agreement);
	std::println("kl mean {:.3e}, max {:.3e} at row {}, {} rows bounded (a candidate outside the other's list)", compared.mean_kl,
	             compared.max_kl, compared.max_kl_row, compared.bounded_rows);
	std::println("perplexity {:.6f} vs {:.6f}", compared.perplexity_a, compared.perplexity_b);
}

auto print_candidates(std::string_view label, std::span<ref::Ranked const> best) -> void {
	std::print("{}:", label);
	for (auto const& each : best.first(std::min<Count>(best.size(), 8))) {
		std::print(" {}({:.4f})", each.id, each.logit);
	}
	std::println("");
}

[[nodiscard]] auto run_logits(Arguments words) -> Outcome<void> {
	if (words.size() < 4) {
		return failure("logits <checkpoint> <out.dump> <input>");
	}
	auto const tokens = input_of(words.subspan(3));
	auto model = open_model(words[1]);
	if (!tokens || !model) {
		return failure(!tokens ? tokens.error().message : model.error().message);
	}
	auto run = teacher_forced<ref::Tensor>(*model, *tokens);
	std::println("logits sha256 {}", run.digest);
	return write_bytes(words[2], ref::encode(dump_from(ref::Source::Tensor, tokens->size(), *tokens, std::move(run))));
}

[[nodiscard]] auto run_greedy(Arguments words) -> Outcome<void> {
	auto const count = words.size() < 5 ? std::nullopt : whole_number(words[3]);
	if (!count || *count == 0) {
		return failure("greedy <checkpoint> <out.dump> <count> <input>");
	}
	auto const prompt = input_of(words.subspan(4));
	auto model = open_model(words[1]);
	if (!prompt || !model) {
		return failure(!prompt ? prompt.error().message : model.error().message);
	}
	auto sequence = ref::Sequence::empty();
	auto tokens = *prompt;
	auto rows = std::vector<ref::DumpRow>{};
	auto next = std::vector<float>{};
	auto feed = std::vector<Token>{tokens};
	auto const started = std::chrono::steady_clock::now();
	for (auto step = Count{0}; step <= *count; ++step) {
		auto const logits = ref::extend<ref::Tensor>(*model, sequence, feed);
		for (auto row = Count{0}; row < feed.size(); ++row) {
			rows.push_back(ref::summarize(logits.row(row)));
			if (row + 1 < feed.size()) {
				next.push_back(logits.row(row)[feed[row + 1]]);
			}
		}
		if (step == *count) {
			break;
		}
		auto const chosen = rows.back().best.front();
		next.push_back(chosen.logit);
		tokens.push_back(chosen.id);
		feed = {chosen.id};
		std::println("token {}: {}", step, chosen.id);
	}
	auto const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	std::println("{} rows in {:.1f} s", tokens.size(), seconds);
	auto const dump = ref::Dump{.source = ref::Source::Tensor, .prompt = prompt->size(), .tokens = tokens, .next = std::move(next), .rows = std::move(rows)};
	return write_bytes(words[2], ref::encode(dump));
}

[[nodiscard]] auto run_noise_floor(Arguments words) -> Outcome<void> {
	if (words.size() < 3) {
		return failure("noise-floor <checkpoint> <input>");
	}
	auto const tokens = input_of(words.subspan(2));
	auto model = open_model(words[1]);
	if (!tokens || !model) {
		return failure(!tokens ? tokens.error().message : model.error().message);
	}
	auto const tensor = dump_from(ref::Source::Tensor, tokens->size(), *tokens, teacher_forced<ref::Tensor>(*model, *tokens));
	auto const ieee = dump_from(ref::Source::Ieee, tokens->size(), *tokens, teacher_forced<ref::Ieee>(*model, *tokens));
	auto const compared = ref::compare(tensor, ieee);
	if (!compared) {
		return failure("the two runs saw different tokens");
	}
	std::println("noise floor: sm86 (a) against ieee (b) accumulation");
	print_comparison(*compared);
	return {};
}

[[nodiscard]] auto read_dump(std::string_view path) -> Outcome<ref::Dump> {
	return read_bytes(path).and_then([path](std::vector<std::byte> const& bytes) -> Outcome<ref::Dump> {
		auto dump = ref::decode(bytes);
		if (!dump) {
			return failure(std::format("{} is not a dump: failure {}", path, std::to_underlying(dump.error().failure)));
		}
		return std::move(*dump);
	});
}

[[nodiscard]] auto run_rulers(Arguments words) -> Outcome<void> {
	if (words.size() != 3) {
		return failure("rulers <a.dump> <b.dump>");
	}
	auto const a = read_dump(words[1]);
	auto const b = read_dump(words[2]);
	if (!a || !b) {
		return failure(!a ? a.error().message : b.error().message);
	}
	auto const compared = ref::compare(*a, *b);
	if (!compared) {
		return failure("the dumps are of different token sequences");
	}
	print_comparison(*compared);
	return {};
}

[[nodiscard]] auto run_decode_ruler(Arguments words) -> Outcome<void> {
	if (words.size() != 3) {
		return failure("decode-ruler <checkpoint> <run.dump>");
	}
	auto const run = read_dump(words[2]);
	auto model = open_model(words[1]);
	if (!run || !model) {
		return failure(!run ? run.error().message : model.error().message);
	}
	auto const reference = dump_from(ref::Source::Tensor, run->prompt, run->tokens, teacher_forced<ref::Tensor>(*model, run->tokens));
	auto const found = ref::first_divergence(reference, *run);
	if (!found) {
		return failure("the dumps are of different token sequences");
	}
	if (!*found) {
		std::println("no divergence: the reference ranks first every one of the {} chosen tokens", run->tokens.size() - run->prompt);
		return {};
	}
	auto const& divergence = **found;
	std::println("first divergence at row {} (position {}): the run chose {}", divergence.row, divergence.row, divergence.chosen);
	print_candidates("reference", divergence.reference);
	print_candidates("run", divergence.run);
	return failure("the run diverges from the reference");
}

struct Prompt {
	std::string name;
	std::string text;
};

[[nodiscard]] auto read_prompts(std::string_view path) -> Outcome<std::vector<Prompt>> {
	auto const text = read_text(path);
	if (!text) {
		return std::unexpected(text.error());
	}
	auto prompts = std::vector<Prompt>{};
	for (auto const line : std::views::split(std::string_view{*text}, '\n')) {
		auto const view = std::string_view{line.begin(), line.end()};
		if (view.empty()) {
			continue;
		}
		auto const space = view.find(' ');
		if (space == std::string_view::npos || space == 0 || space + 1 == view.size()) {
			return failure(std::format("a prompt line is `<name> <text>`: {}", view));
		}
		prompts.push_back({.name = std::string{view.substr(0, space)}, .text = std::string{view.substr(space + 1)}});
	}
	return prompts;
}

[[nodiscard]] auto read_golden(std::string_view path) -> Outcome<ref::Golden> {
	return read_text(path).and_then([path](std::string const& text) -> Outcome<ref::Golden> {
		auto golden = ref::parse_golden(text);
		if (!golden) {
			return failure(std::format("{} is not a golden file", path));
		}
		return std::move(*golden);
	});
}

[[nodiscard]] auto run_golden_record(Arguments words) -> Outcome<void> {
	if (words.size() != 6) {
		return failure("golden record <checkpoint> <golden> <prompts> <reason>");
	}
	auto const prompts = read_prompts(words[4]);
	if (!prompts) {
		return std::unexpected(prompts.error());
	}
	auto const previous = std::filesystem::exists(std::filesystem::path{words[3]}) ? std::optional{read_golden(words[3])} : std::nullopt;
	if (previous && !*previous) {
		return std::unexpected(previous->error());
	}
	auto model = open_model(words[2]);
	if (!model) {
		return std::unexpected(model.error());
	}
	auto entries = std::vector<ref::GoldenEntry>{};
	for (auto const& prompt : *prompts) {
		auto const tokens = encode_text(prompt.text);
		if (!tokens) {
			return std::unexpected(tokens.error());
		}
		auto const run = teacher_forced<ref::Tensor>(*model, *tokens);
		std::println("{}: {} tokens, {}", prompt.name, tokens->size(), run.digest);
		entries.push_back({.prompt = prompt.name, .live_slots = 1, .digest = run.digest});
	}
	auto const earlier = previous ? std::optional{**previous} : std::nullopt;
	auto const golden = ref::regolden(earlier, std::move(entries), std::string{words[5]});
	if (!golden) {
		return failure("the recording is refused: a reason is one line, a prompt name one word, each prompt once");
	}
	if (earlier && *golden == *earlier) {
		std::println("unchanged: every hash reproduces");
		return {};
	}
	auto const text = ref::render(*golden);
	return write_bytes(words[3], std::as_bytes(std::span{text}));
}

[[nodiscard]] auto run_golden_compare(Arguments words) -> Outcome<void> {
	if (words.size() != 4) {
		return failure("golden compare <golden> <candidate>");
	}
	auto const golden = read_golden(words[2]);
	auto const candidate = read_golden(words[3]);
	if (!golden || !candidate) {
		return failure(!golden ? golden.error().message : candidate.error().message);
	}
	auto const mismatches = ref::check(*golden, *candidate);
	for (auto const& mismatch : mismatches) {
		if (mismatch.found) {
			std::println("{} at {} live slots: {} instead of {}", mismatch.prompt, mismatch.live_slots, *mismatch.found, mismatch.golden);
		} else {
			std::println("{}: no entry", mismatch.prompt);
		}
	}
	if (!mismatches.empty()) {
		return failure(std::format("{} mismatches", mismatches.size()));
	}
	std::println("every entry reproduces its golden hash");
	return {};
}

[[nodiscard]] auto run(Command command, Arguments words) -> Outcome<void> {
	switch (command) {
	case Command::Logits:
		return run_logits(words);
	case Command::Greedy:
		return run_greedy(words);
	case Command::NoiseFloor:
		return run_noise_floor(words);
	case Command::Rulers:
		return run_rulers(words);
	case Command::DecodeRuler:
		return run_decode_ruler(words);
	case Command::GoldenRecord:
		return run_golden_record(words);
	case Command::GoldenCompare:
		return run_golden_compare(words);
	}
	std::unreachable();
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	auto words = std::vector<std::string_view>{};
	for (auto const argument : arguments.subspan(1)) {
		words.emplace_back(argument);
	}
	auto const command = command_of(words);
	if (!command) {
		std::println("usage: geezuz-ref logits | greedy | noise-floor | rulers | decode-ruler | golden record | golden compare ...");
		return 2;
	}
	auto const done = run(*command, words);
	if (!done) {
		std::println("{}", done.error().message);
		return 1;
	}
	return 0;
}
