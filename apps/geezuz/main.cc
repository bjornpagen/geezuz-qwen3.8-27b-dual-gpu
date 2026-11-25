// SPDX-License-Identifier: AGPL-3.0-only
//
// geezuz: one request through the same engine the server runs, from the command line.
//
//   geezuz respond <image-dir> <request.json>   one Responses create body, exactly as POST /v1/responses takes it;
//                                               the finished Response object is printed as JSON (a streaming request
//                                               is answered whole all the same)
//   geezuz say <image-dir> <prompt-file>        the file's text as one user message with the api's defaults; the
//                                               answer's text is printed as the engine commits it
//
// The request is validated (its grammar compiled) before the cards are touched, then the engine starts as the server's
// does and runs it under its grammar, through the api's own front. Nothing is stored, so `previous_response_id` is
// refused. Exit status 0 when the reply finished (completed, or cut short by its budget); 1 on a refusal, a failed
// start or a failure while running, each printed; 2 for a command line that is not one of the above.
//
// For the decode-path ruler, `respond` with `"temperature": 0` gives the engine's greedy continuation token for token
// against `geezuz-ref greedy`.
import std;
import engine.cards;
import engine.core;
import engine.file;
import engine.http;
import engine.json;
import engine.net;
import engine.openai;
import engine.process;

namespace {

namespace cards = engine::cards;
namespace core = engine::core;
namespace http = engine::http;
namespace json = engine::json;
namespace net = engine::net;
namespace openai = engine::openai;
namespace process = engine::process;

/** The subcommands; nothing else selects what the program does. */
enum class Command : std::uint8_t {
	Respond,
	Say,
};

/** What the program prints while the reply grows: nothing until the whole response, or the answer as it commits. */
enum class Printing : std::uint8_t {
	WholeResponse,
	AnswerText,
};

/** Time and randomness for the api: the wall clock, and a generator seeded once from the system's entropy. */
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

/** The cli's one request, as the pipeline names it: no connection carries it, so any fixed exchange will do. */
constexpr auto the_exchange = http::Exchange{.connection = {.bits = 1}, .sequence = 1};

/** The tag status snapshots would name the request by. */
constexpr auto the_tag = std::string_view{"cli"};

using Mailbox = net::Mailbox<openai::Update, http::max_posted_messages>;

[[nodiscard]] auto read_text(std::filesystem::path const& path) -> std::optional<std::string> {
	auto const mapping = engine::file::Mapping::open(path);
	if (!mapping) {
		return std::nullopt;
	}
	auto text = std::string{};
	text.reserve(mapping->bytes().size());
	for (auto const byte : mapping->bytes()) {
		text.push_back(std::to_integer<char>(byte));
	}
	return text;
}

/** How the request ended, once its terminal output or the engine's failure arrives. */
struct Running {};

struct Over {
	int status;
};

using Progress = std::variant<Running, Over>;

/** Follows the one request's outputs into its response, printing as `printing` says. */
class Follow {
public:
	Follow(json::Response response, Printing printing, MachineSources& sources)
	    : assembly_{std::move(response)}, printing_{printing}, sources_{sources} {}

	[[nodiscard]] auto take(openai::Update update) -> Progress {
		if (auto const failed = core::alternative<openai::EngineFailed>(update)) {
			process::output_line(std::format("geezuz: the engine failed: {}", failed->reason));
			return Over{.status = 1};
		}
		auto const outputs = core::alternative<openai::Outputs>(update);
		if (!outputs) {
			return Running{};
		}
		for (auto const& item : outputs->items) {
			// The cli submits one request, so every output is that request's.
			core::invariant(item.exchange == the_exchange);
			if (auto const over = output(item.output); std::holds_alternative<Over>(over)) {
				return over;
			}
		}
		return Running{};
	}

private:
	[[nodiscard]] auto output(openai::Output const& output) -> Progress {
		if (std::holds_alternative<openai::Accepted>(output)) {
			return Running{};
		}
		if (auto const refused = core::alternative<openai::PromptTooLong>(output)) {
			process::output_line(std::format("geezuz: the prompt's {} tokens leave no room in the context", refused->prompt_tokens));
			return Over{.status = 1};
		}
		if (std::holds_alternative<openai::Overran>(output)) {
			process::output_line("geezuz: the request's grammar outgrew its bounds; the reply ended there");
			return Over{.status = 1};
		}
		// Only a client that left is cancelled, and the cli never leaves its request.
		core::invariant(!std::holds_alternative<openai::Cancelled>(output));
		for (auto const& step : assembly_.take(output, sources_.get())) {
			print(step);
		}
		if (!std::holds_alternative<openai::Finished>(output)) {
			return Running{};
		}
		if (printing_ == Printing::WholeResponse) {
			process::output_line(json::encode(assembly_.response()));
		} else {
			process::output_line("");
		}
		return Over{.status = 0};
	}

	auto print(openai::Step const& step) -> void {
		if (printing_ != Printing::AnswerText) {
			return;
		}
		auto const delta = core::alternative<openai::Delta>(step);
		auto const answer = delta && std::holds_alternative<json::MessageOutput>(assembly_.item(delta->index));
		if (answer) {
			process::output(delta->text);
		}
	}

	openai::Assembly assembly_;
	Printing printing_;
	std::reference_wrapper<MachineSources> sources_;
};

[[nodiscard]] auto run(Command command, std::filesystem::path const& images, std::filesystem::path const& input) -> int {
	auto const text = read_text(input);
	if (!text) {
		process::output_line(std::format("geezuz: cannot read {}", input.string()));
		return 1;
	}
	auto entropy = std::random_device{};
	auto sources = MachineSources{(std::uint64_t{entropy()} << 32U) | entropy()};
	auto admitted = [&] -> core::Result<openai::Admitted, openai::ApiError> {
		if (command == Command::Say) {
			return openai::say(*text, the_tag, sources);
		}
		auto gate = openai::Gate::make();
		// The decoder's readers are built into the program; one that does not load is a broken build.
		core::invariant(gate.has_value());
		auto storeless = openai::Storeless{};
		return gate->admit_text(*text, the_tag, sources, storeless);
	}();
	if (!admitted) {
		process::output_line(std::format("geezuz: refused: {}", openai::error_body(admitted.error())));
		return 1;
	}

	auto poller = net::Poller::open();
	if (!poller) {
		process::output_line(std::format("geezuz: cannot open a wait (system error {})", poller.error().code));
		return 1;
	}
	auto mailbox = std::make_unique<Mailbox>(poller->waker());
	auto pipeline = cards::load(images, http::Poster<openai::Update>{*mailbox});
	if (!pipeline) {
		process::output_line(std::format("geezuz: the engine could not start: {}", cards::describe(pipeline.error())));
		return 1;
	}
	auto follow = Follow{std::move(admitted->response), command == Command::Say ? Printing::AnswerText : Printing::WholeResponse, sources};
	pipeline->submit(the_exchange, std::move(admitted->submission));
	auto progress = Progress{Running{}};
	while (std::holds_alternative<Running>(progress)) {
		if (auto const waited = poller->wait_until(std::nullopt); !waited) {
			process::output_line(std::format("geezuz: the wait failed (system error {})", waited.error().code));
			progress = Over{.status = 1};
			break;
		}
		mailbox->drain([&](openai::Update&& update) {
			if (std::holds_alternative<Running>(progress)) {
				progress = follow.take(std::move(update));
			}
		});
	}
	auto const stopped = pipeline->stop();
	if (!stopped) {
		process::output_line(std::format("geezuz: {}", cards::describe(cards::ServingError{.reason = stopped.error()})));
		return 1;
	}
	return std::get<Over>(progress).status;
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	auto const command = [&] -> std::optional<Command> {
		if (arguments.size() != 4) {
			return std::nullopt;
		}
		auto const name = std::string_view{arguments[1]};
		return name == "respond" ? std::optional{Command::Respond} : name == "say" ? std::optional{Command::Say} : std::nullopt;
	}();
	if (!command) {
		process::output_line("usage: geezuz respond <image-dir> <request.json> | geezuz say <image-dir> <prompt-file>");
		return 2;
	}
	return run(*command, std::filesystem::path{arguments[2]}, std::filesystem::path{arguments[3]});
}
