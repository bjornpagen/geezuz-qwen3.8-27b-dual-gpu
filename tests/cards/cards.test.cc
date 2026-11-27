// SPDX-License-Identifier: AGPL-3.0-only
//
// The cards adapter over the decode engine and prefill's chunks, on the driver double with small synthetic images and
// a model of the resident loop on each card. The model is a test double of the loop's protocol, not its arithmetic: it
// logs command tickets, and answers a cycle or a prompt's end with tokens that are a fixed function of what the
// descriptor asked (so both cards agree, and the test knows the answer). The model follows the driver's launches: each
// time the cards launch the loops it starts serving the new rings, on the same run loop the cards' senders poll.
import std;
import engine.card_scheduler;
import engine.cards;
import engine.core;
import engine.decode;
import engine.decode_device;
import engine.device;
import engine.driver_double;
import engine.exec;
import engine.file_write;
import engine.image;
import engine.image_synthetic;
import engine.kv;
import engine.model;
import engine.pipeline;
import engine.sampling;
import engine.sched;
import engine.testing;
import engine.vocabulary;

namespace {

namespace cards = engine::cards;
namespace core = engine::core;
namespace decode = engine::decode;
namespace dd = engine::decode_device;
namespace device = engine::device;
namespace ex = engine::exec;
namespace fake = engine::driver_double;
namespace image = engine::image;
namespace kv = engine::kv;
namespace model = engine::model;
namespace pipeline = engine::pipeline;
namespace sched = engine::sched;
namespace synthetic = engine::image_synthetic;

struct SyntheticCatalogs {
	using CardZero = synthetic::SyntheticCatalog<image::Target::CardZero>;
	using CardOne = synthetic::SyntheticCatalog<image::Target::CardOne>;
	using Host = synthetic::SyntheticCatalog<image::Target::Host>;
};

using Engine = decode::Engine<SyntheticCatalogs>;
using Cards = cards::BasicCards<SyntheticCatalogs>;
using Error = decode::DecodeError;
using engine::model::Card;

static_assert(pipeline::Cards<Cards, kv::model_geometry>);

auto const deadline_after = [] {
	return std::chrono::steady_clock::now() + std::chrono::seconds{30};
};

template<image::Catalog C>
[[nodiscard]] auto write_image(std::filesystem::path const& path, std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	auto const baked = synthetic::bake<C>(synthetic::synthetic_checkpoint, seed, random);
	auto output = engine::file_write::Output::create(path, baked.file.size());
	return output && output->write(0, baked.file).has_value() && std::move(*output).commit().has_value();
}

[[nodiscard]] auto images() -> std::optional<decode::Paths> {
	auto const directory = std::filesystem::current_path() / "cards.scratch";
	std::filesystem::remove_all(directory);
	std::filesystem::create_directories(directory);
	auto const paths =
	    decode::Paths{.card_zero = directory / "card0.image", .card_one = directory / "card1.image", .host = directory / "host.image"};
	auto const written = write_image<SyntheticCatalogs::CardZero>(paths.card_zero, 1) &&
	                     write_image<SyntheticCatalogs::CardOne>(paths.card_one, 2) && write_image<SyntheticCatalogs::Host>(paths.host, 3);
	return written ? std::optional{paths} : std::nullopt;
}

/** The token the model loop commits at sequence position `position`. */
[[nodiscard]] constexpr auto truth(std::uint64_t position) -> unsigned {
	return 1000 + static_cast<unsigned>(position);
}

/** The model loop's `index`-th draft after a row that committed its last token at `position`. */
[[nodiscard]] constexpr auto proposal(std::uint64_t position, std::size_t index) -> unsigned {
	return 50'000 + static_cast<unsigned>(position * 8 + index);
}

/**
 * The logit the model loop gives `token` after the input `input` at `position`: an exact fp32 integer, distinct for
 * nearly every token, so a row computed for the wrong position or input, or put back in the wrong order, reads wrong.
 */
[[nodiscard]] constexpr auto forced_logit(std::uint64_t token, std::uint64_t position, std::uint64_t input) -> float {
	return static_cast<float>(((token * 131) + (position * 17) + (input * 7)) % 1'000'003);
}

/** What one launch of the loop on one card saw, read only after its thread has been joined. */
struct Seen {
	std::vector<dd::Command> applied;
	std::vector<dd::Work> work;
	/** Tickets served, the exit's included: a ticket's number picks its half of the logits buffer. */
	std::uint64_t tickets = 0;
	/** Each logits ticket's rows, as (slot, position, token). */
	std::vector<std::vector<std::array<unsigned, 3>>> forced;
};

/**
 * One card's loop, as a protocol double: a cycle accepts every draft it verifies and commits the truth at each
 * position; a prompt's end commits the truth after the prompt; both propose the next drafts from the last position.
 */
struct ModelLoop {
	std::reference_wrapper<Seen> seen;
	/** The card's logits buffer, at the address its loop is given: under the fake, a mapped buffer's host address. */
	dd::LogitsBuffer* logits;
	/** The token each of the card's lm_head rows holds. */
	std::reference_wrapper<std::vector<unsigned> const> row_tokens;

	[[nodiscard]] auto operator()(dd::Descriptor const& descriptor) const -> dd::Result {
		auto result = dd::Result{};
		seen.get().tickets += 1;
		seen.get().work.push_back(descriptor.work);
		std::ranges::copy(std::span{descriptor.command}.first(descriptor.commands), std::back_inserter(seen.get().applied));
		if (descriptor.work == dd::Work::Apply) {
			return result;
		}
		if (descriptor.work == dd::Work::Logits) {
			force(descriptor);
			return result;
		}
		for (auto index = std::size_t{0}; index < descriptor.live; ++index) {
			auto const& run = std::span{descriptor.slots}[index];
			auto& outcome = std::span{result.slots}[index];
			outcome.accepted = run.drafts;
			for (auto verify = std::size_t{0}; verify <= run.drafts; ++verify) {
				std::span{outcome.committed}[verify] = truth(run.position + 1 + verify);
			}
			auto const last = run.position + 1 + run.drafts;
			for (auto draft = std::size_t{0}; draft < dd::max_drafts; ++draft) {
				std::span{outcome.drafts}[draft] = proposal(last, draft);
			}
		}
		result.started = 1000;
		result.finished = 2000;
		return result;
	}

	/** Writes each row's logits into this card's half of the buffer the ticket's number picks, in the card's row order. */
	auto force(dd::Descriptor const& descriptor) const -> void {
		auto& half = std::span{logits->half}[dd::logits_half(seen.get().tickets)];
		auto& rows = seen.get().forced.emplace_back();
		for (auto index = std::size_t{0}; index < descriptor.live; ++index) {
			auto const& run = std::span{descriptor.slots}[index];
			rows.push_back({run.slot, run.position, run.token});
			auto values = std::span{std::span{half.rows}[index]};
			for (auto const [row, token] : std::views::enumerate(row_tokens.get())) {
				values[core::narrow<std::size_t>(row)] = forced_logit(token, run.position, run.token);
			}
		}
	}
};

using Kernel = fake::RingKernel<dd::Descriptor, dd::Result, dd::ring_slots, ModelLoop>;

/** The loops on both cards as the driver launched them: a model loop per launch, serving its ring. */
class Loops {
public:
	/** Starts a model loop for each loop launch not yet followed; how many have been launched so far. */
	auto follow(fake::Fake const& driver) -> std::size_t {
		auto const launches = driver.launches();
		for (; seen_launches_ < launches.size(); ++seen_launches_) {
			auto const& launch = launches[seen_launches_];
			if (launch.entry != "decode_loop") {
				continue;
			}
			auto bytes = std::array<std::byte, sizeof(dd::Params)>{};
			core::invariant(launch.parameters.size() == bytes.size());
			std::ranges::copy(launch.parameters, bytes.begin());
			auto const params = std::bit_cast<dd::Params>(bytes);
			auto& seen = seen_.emplace_back();
			auto const card = params.card == 0 ? Card::Zero : Card::One;
			kernels_.push_back(std::make_unique<Kernel>(params.ring, ModelLoop{.seen = seen,
			                                                                    .logits = std::bit_cast<dd::LogitsBuffer*>(params.logits),
			                                                                    .row_tokens = row_tokens_.at(std::to_underlying(card))}));
			order_.push_back(seen_launches_);
		}
		return kernels_.size();
	}

	/** Joins every model loop; their logs may be read after. */
	auto join() -> void {
		kernels_.clear();
	}

	[[nodiscard]] auto seen(std::size_t loop) const -> Seen const& {
		return seen_.at(loop);
	}

	/** The driver's launch index of the `loop`-th loop launch. */
	[[nodiscard]] auto launch_of(std::size_t loop) const -> std::size_t {
		return order_.at(loop);
	}

private:
	/** Each card's lm_head row-to-token map, as the cards are given it. */
	std::array<std::vector<unsigned>, model::cards> row_tokens_ = {engine::vocabulary::row_tokens(Card::Zero), engine::vocabulary::row_tokens(Card::One)};
	std::size_t seen_launches_ = 0;
	std::deque<Seen> seen_;
	std::vector<std::unique_ptr<Kernel>> kernels_;
	std::vector<std::size_t> order_;
};

/** Follows the driver's launches, polled on the cards' own loop, until `count` loops have been launched. */
struct Follow {
	std::reference_wrapper<Loops> loops;
	std::reference_wrapper<fake::Fake const> driver;
	std::size_t count;

	[[nodiscard]] auto operator()() const -> std::optional<core::Result<void, Error>> {
		if (loops.get().follow(driver.get()) >= count) {
			return core::Result<void, Error>{};
		}
		return std::nullopt;
	}
};

/** A sender that follows the launches until `count` loops have been launched. */
[[nodiscard]] auto until(Loops& loops, fake::Fake const& driver, std::size_t count) {
	return engine::card_scheduler::polled(Follow{.loops = loops, .driver = driver, .count = count});
}

[[nodiscard]] constexpr auto rule() -> engine::sampling::Rule {
	return {.seed = 7,
	        .filter = {.temperature = 1.0, .top_p = 1.0, .min_p = 0.0, .top_k = 1, .nucleus = engine::sampling::Nucleus::Off},
	        .request = 1};
}

/** A mask allowing every row of both cards. */
[[nodiscard]] auto everything() -> pipeline::RowMask {
	auto mask = pipeline::RowMask{};
	mask.words.fill(~std::uint64_t{0});
	return mask;
}

[[nodiscard]] auto masks_for(sched::Batch const& batch) -> pipeline::Masks {
	return pipeline::Masks{.rows = std::vector<pipeline::RowMask>(batch.rows.size() * (1 + model::drafts(batch.depth)), everything())};
}

template<class S>
[[nodiscard]] auto run(S sender) {
	auto waited = ex::wait<Error>(std::move(sender));
	return std::move(waited).value();
}

[[nodiscard]] auto row(std::size_t slot, std::uint64_t position, unsigned token) -> sched::Row {
	return {.slot = kv::Slot::at(slot),
	        .request = sched::RequestHandle{.bits = slot + 1},
	        .position = position,
	        .token = token,
	        .rule = rule(),
	        .most = 8};
}

/** Runs one batch through draft and verify: its drafts as the pipeline sees them, and its outcome. */
struct Cycled {
	pipeline::Drafts drafts;
	sched::BatchDone done;
	std::size_t masks;
};

[[nodiscard]] auto cycle(Cards& cards, sched::Batch const& batch) -> std::optional<Cycled> {
	auto drafted = run(cards.draft(sched::Batch{batch}));
	if (!drafted) {
		return std::nullopt;
	}
	auto verified = run(cards.verify(masks_for(batch)));
	if (!verified) {
		return std::nullopt;
	}
	auto& [value] = *verified;
	return Cycled{.drafts = std::get<0>(*drafted), .done = std::move(value.done), .masks = value.masks.rows.size()};
}

/**
 * The whole life of the cards: assembled and started, a fresh row and a continued one, a pool command and a chunk
 * that ends its prompt, the continued slot with the prompt end's drafts, a chunk that reads on, and the teardown.
 */
[[nodiscard]] auto serves_batches_and_chunks(decode::Paths const& paths) -> bool {
	auto const driver = fake::Fake{fake::Machine{}};
	auto loops = Loops{};
	auto held = true;
	auto const check = [&](bool fact) {
		held = held && fact;
	};
	{
		auto devices = device::Devices::open(driver.driver());
		if (!devices) {
			return false;
		}
		auto engine = Engine::open(std::move(*devices), paths, synthetic::synthetic_checkpoint, deadline_after());
		if (!engine) {
			return false;
		}
		auto assembled = Cards::assemble(std::move(*engine));
		if (!assembled) {
			return false;
		}
		auto& cards = *assembled;
		check(loops.follow(driver) == 2);

		// A fresh row verifies no drafts, whatever the depth; it commits the truth after its token.
		auto const fresh = cycle(cards, sched::Batch{.depth = model::DraftDepth::Two, .rows = {row(0, 10, 42)}});
		check(fresh && fresh->drafts.rows.size() == 1 && fresh->drafts.rows[0].count == 0 && fresh->masks == 3 &&
		      fresh->done.rows.size() == 1 && fresh->done.rows[0].view().size() == 1 && fresh->done.rows[0].tokens[0] == truth(11) &&
		      fresh->done.took > std::chrono::nanoseconds{0});

		// Continued, the row verifies the drafts the last cycle proposed, cut to the depth.
		auto const continued = cycle(cards, sched::Batch{.depth = model::DraftDepth::Two, .rows = {row(0, 11, truth(11))}});
		check(continued && continued->drafts.rows[0].count == 2 &&
		      std::ranges::equal(continued->drafts.rows[0].view(), std::array{proposal(11, 0), proposal(11, 1)}) &&
		      continued->done.rows[0].view().size() == 3 && continued->done.rows[0].tokens[2] == truth(14));

		// A pool command queued before a chunk reaches the cards before the loops yield to it.
		cards.apply(kv::SetTableEntry{.slot = kv::Slot::at(1), .entry = 0, .block = 5});
		auto const chunk = sched::Chunk{.slot = kv::Slot::at(1),
		                                .request = sched::RequestHandle{.bits = 2},
		                                .from = 0,
		                                .tokens = {1, 2, 3},
		                                .read = sched::Read::Last,
		                                .rule = rule()};
		auto const prompted = run(ex::when_all(cards.prefill(pipeline::Prefill{pipeline::Sampling{.chunk = chunk, .mask = everything()}}),
		                                       until(loops, driver, 4)));
		check(prompted.has_value() && std::get<0>(*prompted).sampled == std::optional{truth(3)} &&
		      std::get<0>(*prompted).took > std::chrono::nanoseconds{0});

		// The slot continues from its sampled token with the prompt end's drafts.
		auto const after_prompt =
		    cycle(cards, sched::Batch{.depth = model::DraftDepth::Three,
			                          .rows = {row(0, 14, truth(14)), sched::Row{.slot = kv::Slot::at(1),
			                                                                     .request = sched::RequestHandle{.bits = 2},
			                                                                     .position = 3,
			                                                                     .token = truth(3),
			                                                                     .rule = rule(),
			                                                                     .most = 8}}});
		check(after_prompt && after_prompt->drafts.rows.size() == 2 && after_prompt->drafts.rows[1].count == 3 &&
		      std::ranges::equal(after_prompt->drafts.rows[1].view(), std::array{proposal(3, 0), proposal(3, 1), proposal(3, 2)}) &&
		      after_prompt->masks == 8);

		auto const reading = sched::Chunk{.slot = kv::Slot::at(2),
		                                  .request = sched::RequestHandle{.bits = 3},
		                                  .from = 0,
		                                  .tokens = {4, 5},
		                                  .read = sched::Read::Part,
		                                  .rule = rule()};
		auto const read = run(ex::when_all(cards.prefill(pipeline::Prefill{pipeline::Reading{.chunk = reading}}), until(loops, driver, 6)));

		// A logits ticket over a continuing slot and a fresh one: every token's logit in token order, from both cards'
		// halves, each row's at its own position and input; and the continuing slot's drafts are gone afterwards.
		auto const forced = sched::Batch{.depth = model::DraftDepth::One, .rows = {row(0, 18, truth(18)), row(3, 5, 9)}};
		auto const logits = run(cards.logits(forced));
		auto const expected_row = [](sched::Row const& asked) {
			auto values = std::vector<float>(model::sampled_vocabulary);
			for (auto token = std::size_t{0}; token < values.size(); ++token) {
				values[token] = forced_logit(token, asked.position, asked.token);
			}
			return values;
		};
		check(logits.has_value() && std::get<0>(*logits).rows.size() == 2 && std::get<0>(*logits).rows[0] == expected_row(forced.rows[0]) &&
		      std::get<0>(*logits).rows[1] == expected_row(forced.rows[1]));
		auto const after_logits = cycle(cards, sched::Batch{.depth = model::DraftDepth::Two, .rows = {row(0, 19, 77)}});
		check(after_logits && after_logits->drafts.rows[0].count == 0 && after_logits->done.rows[0].tokens[0] == truth(20));
		check(read.has_value() && !std::get<0>(*read).sampled.has_value());
	}
	loops.join();
	// The first loops applied the queued command before they exited for the chunk, and the chunk's kernels ran between
	// those loops and the next.
	auto const lowered = std::ranges::any_of(loops.seen(0).applied, [](dd::Command const& command) {
		return command.action == dd::Action::SetTableEntry && command.slot == 1 && command.first == 5 && command.second == 0;
	});
	auto const launches = driver.launches();
	auto const between = std::ranges::any_of(std::views::iota(loops.launch_of(1) + 1, loops.launch_of(2)), [&](std::size_t index) {
		return launches[index].entry != "decode_loop";
	});
	// Both cards ran the logits ticket on the loops the last chunk relaunched, each row as asked.
	auto const asked = std::vector<std::vector<std::array<unsigned, 3>>>{{{0, 18, truth(18)}, {3, 5, 9}}};
	auto const ran = loops.seen(4).forced == asked && loops.seen(5).forced == asked;
	return held && lowered && between && ran && driver.violations().empty() && driver.nothing_live();
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const paths = images();
	if (!paths) {
		std::println("FAIL: the synthetic images could not be written");
		return 1;
	}
	auto const cases = std::array{
	    Case{.name = "the cards serve batches and chunks over decode and prefill, and tear down clean",
		     .holds = serves_batches_and_chunks(*paths)},
	};
	return engine::testing::report(cases);
}
