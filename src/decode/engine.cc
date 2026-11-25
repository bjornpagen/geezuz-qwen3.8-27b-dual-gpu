// SPDX-License-Identifier: AGPL-3.0-only
//
// The decode engine on both cards: the baked images opened and verified, each card's image copied whole into its
// memory and the host image into pinned memory both cards map, the row-to-token tables, the kv pool, the slots'
// deltanet state and the workspace allocated, the decode image loaded, and one resident loop started per card. Then
// cycles run through the session, and `stop` ends both loops. Every device size is a compile-time constant; the
// images' catalogs are a parameter only so the tests can stand small images in for the real ones.
export module engine.decode:engine;

import std;
import engine.attn;
import engine.core;
import engine.decode_device;
import engine.device;
import engine.image;
import engine.model;
import engine.sched;
import engine.vocabulary;
import :commands;
import :error;
import :session;

namespace engine::decode {

namespace device_layout = decode_device;

/** The three images the engine loads, by catalog: each card's and the host's. */
export template<class C>
concept Catalogs = image::Catalog<typename C::CardZero> && image::Catalog<typename C::CardOne> && image::Catalog<typename C::Host>;

/** The model's images, as the baker writes them. */
export struct ModelCatalogs {
	using CardZero = image::CardCatalog<model::Card::Zero>;
	using CardOne = image::CardCatalog<model::Card::One>;
	using Host = image::HostCatalog;
};

static_assert(Catalogs<ModelCatalogs>);

/** Where the three images are. */
export struct Paths {
	std::filesystem::path card_zero;
	std::filesystem::path card_one;
	std::filesystem::path host;
};

/** An opened image as the host bytes a copy reads: its payload, which is copied to its memory whole. */
template<image::Catalog C>
class Payload {
public:
	explicit Payload(image::Image<C> opened) : image_{std::move(opened)} {}

	[[nodiscard]] auto bytes() const -> std::span<std::byte const> {
		return image_.payload();
	}

private:
	image::Image<C> image_;
};

template<image::Catalog C>
using Weights = device::DeviceArray<std::byte, image::payload_bytes<C>>;

using RowTokens = device::DeviceArray<std::uint32_t, model::vocab_per_card>;
using Workspace = device::DeviceArray<std::byte, device_layout::workspace.bytes>;
using Pool = device::DeviceArray<std::byte, model::pool_bytes>;
using States = device::DeviceArray<std::byte, model::slots * model::slot_state_bytes>;
using Control = device::DeviceArray<device_layout::Control, 1>;

/** One card's memories, its stream, and its loaded program. */
template<image::Catalog C>
struct Card {
	device::Stream stream;
	Weights<C> weights;
	RowTokens row_tokens;
	Workspace workspace;
	Pool pool;
	States states;
	Control control;
	device_layout::Program program;
	std::optional<device::Event> launched;
};

/** Waits for a transfer and hands back its destination, or the copy's failure, or `late` past `deadline`. */
template<class Destination, class Source>
[[nodiscard]] auto finish(device::Transfer<Destination, Source> transfer, std::chrono::steady_clock::time_point deadline, TimedOut late)
    -> core::Result<Destination, DecodeError> {
	if (transfer.wait_until(deadline) == device::EventState::Pending) {
		// The copy is still reading its source; the transfer may not be dropped while it does.
		core::invariant(transfer.wait_until(std::chrono::steady_clock::time_point::max()) == device::EventState::Complete);
		return fail(late);
	}
	if (auto delivered = transfer.try_finish(); delivered.has_value()) {
		if (!delivered->copied) {
			return fail(delivered->copied.error());
		}
		return std::move(delivered->destination);
	}
	// Not reached: a finished copy hands its buffers back once, and this is that once.
	core::invariant(false);
	std::unreachable();
}

/** Copies a pinned staging array of `T` to a new array on the card. */
template<device::Wire T, std::size_t Count>
[[nodiscard]] auto upload_staged(device::Context const& context, device::Stream const& stream, device::MappedArray<T, Count> staged,
                                 std::chrono::steady_clock::time_point deadline, TimedOut late) -> core::Result<device::DeviceArray<T, Count>, DecodeError> {
	auto array = context.allocate<T, Count>();
	if (!array) {
		return fail(array.error());
	}
	return finish(context.upload(stream, std::move(*array), std::move(staged), 0), deadline, late);
}

/**
 * Attention's rope table in the workspace at `table`, filled once by its own kernel, which evaluates the model's
 * rotations; decode and prefill both read it.
 */
[[nodiscard]] auto fill_rotations(device::Context const& context, model::Card card, device::Stream const& stream, std::uint64_t table,
                                  std::chrono::steady_clock::time_point deadline) -> core::Result<void, DecodeError> {
	auto program = context.load<attn::RopeTable>();
	auto event = program.and_then([&](auto const&) {
		return context.create_event();
	});
	auto launched = event.and_then([&](auto const&) {
		return context.launch<attn::RopeTable>(*program, stream, table).and_then([&] {
			return event->record(stream);
		});
	});
	if (!launched) {
		return fail(launched.error());
	}
	auto const state = event->wait_until(deadline);
	if (!state) {
		return fail(state.error());
	}
	if (*state == device::EventState::Pending) {
		return fail(TimedOut{.card = card, .waiting = Waiting::Rotations});
	}
	return {};
}

/** Zeroes a card's control block, the words its loop must find zero when it starts. */
[[nodiscard]] auto zero_control(device::Context const& context, model::Card card, device::Stream const& stream, Control control,
                                std::chrono::steady_clock::time_point deadline) -> core::Result<Control, DecodeError> {
	auto staged = context.allocate_mapped<device_layout::Control, 1>();
	if (!staged) {
		return fail(staged.error());
	}
	return finish(context.upload(stream, std::move(control), std::move(*staged), 0), deadline, TimedOut{.card = card, .waiting = Waiting::Control});
}

/** Opens, verifies and copies one card's image, and allocates and fills the rest of the card's memories. */
template<image::Catalog C>
[[nodiscard]] auto load_card(device::Context const& context, model::Card card, std::filesystem::path const& path, image::Checkpoint const& checkpoint,
                             std::chrono::steady_clock::time_point deadline) -> core::Result<Card<C>, DecodeError> {
	auto opened = image::Image<C>::open(path, checkpoint);
	if (!opened) {
		return fail(Unloadable{.image = card == model::Card::Zero ? ImageOf::CardZero : ImageOf::CardOne, .error = opened.error()});
	}
	auto stream = context.create_stream();
	if (!stream) {
		return fail(stream.error());
	}
	auto pinned = context.pin(Payload<C>{std::move(*opened)});
	auto weights = pinned.and_then([&](auto const&) {
		return context.allocate<std::byte, image::payload_bytes<C>>();
	});
	if (!weights) {
		return fail(weights.error());
	}
	auto copied = finish(context.upload(*stream, std::move(*weights), std::move(*pinned), 0), deadline, TimedOut{.card = card, .waiting = Waiting::Weights});
	if (!copied) {
		return std::unexpected(copied.error());
	}
	auto tokens = context.allocate_mapped<std::uint32_t, model::vocab_per_card>();
	if (!tokens) {
		return fail(tokens.error());
	}
	std::ranges::copy(vocabulary::row_tokens(card), tokens->host().begin());
	auto row_tokens = upload_staged(context, *stream, std::move(*tokens), deadline, TimedOut{.card = card, .waiting = Waiting::RowTokens});
	if (!row_tokens) {
		return std::unexpected(row_tokens.error());
	}
	auto allocated = context.allocate<device_layout::Control, 1>();
	if (!allocated) {
		return fail(allocated.error());
	}
	auto control = zero_control(context, card, *stream, std::move(*allocated), deadline);
	if (!control) {
		return std::unexpected(control.error());
	}
	auto workspace = context.allocate<std::byte, device_layout::workspace.bytes>();
	auto pool = workspace.and_then([&](auto const&) {
		return context.allocate<std::byte, model::pool_bytes>();
	});
	auto states = pool.and_then([&](auto const&) {
		return context.allocate<std::byte, model::slots * model::slot_state_bytes>();
	});
	if (!states) {
		return fail(states.error());
	}
	if (auto rotations = fill_rotations(context, card, *stream, workspace->address() + device_layout::workspace.exchange.rope_table, deadline); !rotations) {
		return std::unexpected(rotations.error());
	}
	auto program = context.load<device_layout::Loop>();
	if (!program) {
		return fail(program.error());
	}
	return Card<C>{
	    .stream = std::move(*stream),
	    .weights = std::move(*copied),
	    .row_tokens = std::move(*row_tokens),
	    .workspace = std::move(*workspace),
	    .pool = std::move(*pool),
	    .states = std::move(*states),
	    .control = std::move(*control),
	    .program = std::move(*program),
	    .launched = std::nullopt,
	};
}

/** The host image's bytes in pinned memory both cards map: the embedding table each cycle gathers rows from. */
template<image::Catalog Host>
using HostImage = device::MappedArray<std::byte, image::payload_bytes<Host>>;

/**
 * Keeps `owner` from its destructor for the rest of the process: what it owns may still be written by a kernel no one
 * can stop, and freed memory would be handed to whatever allocates next. Never freed, by design; the process is
 * expected to exit.
 */
template<class Owner>
auto lose(Owner owner) -> void {
	[[clang::no_destroy]] static auto lost = std::pair<std::mutex, std::vector<std::shared_ptr<void>>>{};
	auto const held = std::scoped_lock{lost.first};
	lost.second.push_back(std::make_shared<Owner>(std::move(owner)));
}

/**
 * Where the resident loops stand. Halted loops used up their rings (each carried an exit ticket), so the next start
 * takes fresh ones; unlaunched loops still have theirs. Failed loops ended on an error and their kernels have finished;
 * abandoned loops were given up on at a deadline, and may still run, so the engine never frees what they could write.
 * Nothing restarts either.
 */
enum class Loops : std::uint8_t {
	Unlaunched,
	Running,
	Halted,
	Failed,
	Abandoned,
};

/**
 * One card's context and stream, borrowed from the engine for one synchronous call: valid until the engine's next
 * non-const call, never stored.
 */
export struct CardAccess {
	device::Context const& context;
	device::Stream const& stream;
};

/**
 * The memories of one card that prefill borrows while the loops are stopped: the image, the mapped embedding table,
 * the kv pool and the block tables, the slots' deltanet state images, the rope table, and the prompt row where it
 * leaves a prompt's last final-normed row for the prompt's end.
 */
export struct Memories {
	std::uint64_t image;
	std::uint64_t embedding;
	std::uint64_t pool;
	std::uint64_t tables;
	std::uint64_t states;
	std::uint64_t rotations;
	std::uint64_t prompt_row;
};

export template<Catalogs C>
class Engine {
public:
	/**
	 * Opens and verifies the images and loads them onto `devices`, which the engine keeps: every allocation it makes
	 * dies before the contexts it came from. Nothing runs on the cards until `start`.
	 */
	[[nodiscard]] static auto open(device::Devices devices, Paths const& paths, image::Checkpoint const& checkpoint,
	                               std::chrono::steady_clock::time_point deadline) -> core::Result<Engine, DecodeError> {
		auto host_image = image::Image<typename C::Host>::open(paths.host, checkpoint);
		if (!host_image) {
			return fail(Unloadable{.image = ImageOf::Host, .error = host_image.error()});
		}
		auto host = devices.context(model::Card::Zero).template allocate_mapped<std::byte, image::payload_bytes<typename C::Host>>();
		if (!host) {
			return fail(host.error());
		}
		std::ranges::copy(host_image->payload(), host->host().begin());
		auto zero = load_card<typename C::CardZero>(devices.context(model::Card::Zero), model::Card::Zero, paths.card_zero, checkpoint, deadline);
		if (!zero) {
			return std::unexpected(zero.error());
		}
		auto one = load_card<typename C::CardOne>(devices.context(model::Card::One), model::Card::One, paths.card_one, checkpoint, deadline);
		if (!one) {
			return std::unexpected(one.error());
		}
		auto session = Session::create(devices);
		if (!session) {
			return std::unexpected(session.error());
		}
		return Engine{std::move(devices), std::move(*host), std::move(*zero), std::move(*one), std::move(*session)};
	}

	Engine(Engine&& other) noexcept
	    : devices_{std::move(other.devices_)},
	      host_{std::move(other.host_)},
	      zero_{std::move(other.zero_)},
	      one_{std::move(other.one_)},
	      session_{std::move(other.session_)},
	      loops_{std::exchange(other.loops_, Loops::Unlaunched)},
	      launches_{other.launches_},
	      failure_{std::move(other.failure_)} {}

	auto operator=(Engine&&) noexcept -> Engine& = delete;
	Engine(Engine const&) = delete;
	auto operator=(Engine const&) -> Engine& = delete;

	~Engine() {
		// A running loop reads every allocation here: it must have been shut down first.
		core::invariant(loops_ != Loops::Running);
		if (loops_ == Loops::Abandoned) {
			// Abandoned loops may still write any of it, the rings and mask buffers in host memory included: fail closed,
			// keeping every owner (the contexts too, whose destruction frees what they hold) unfreed. What is left of the
			// members below is moved-from, and frees nothing.
			lose(Remains{.devices = std::move(devices_),
			             .host = std::move(host_),
			             .zero = std::move(zero_),
			             .one = std::move(one_),
			             .session = std::move(session_)});
		}
	}

	/** The parameters card `card`'s loop starts with: its memories, and the peer's workspace and control block. */
	[[nodiscard]] auto parameters(model::Card card) const -> device_layout::Params {
		auto const own = [&](auto const& zero, auto const& one) {
			return card == model::Card::Zero ? zero.address() : one.address();
		};
		auto const peer = [&](auto const& zero, auto const& one) {
			return card == model::Card::Zero ? one.address() : zero.address();
		};
		auto const host = session_.addresses(card);
		return {
		    .image = own(zero_.weights, one_.weights),
		    .embedding = host_.device_address() + image::layout<typename C::Host>[0].offset,
		    .row_tokens = own(zero_.row_tokens, one_.row_tokens),
		    .workspace = own(zero_.workspace, one_.workspace),
		    .peer_workspace = peer(zero_.workspace, one_.workspace),
		    .control = own(zero_.control, one_.control),
		    .peer_control = peer(zero_.control, one_.control),
		    .pool = own(zero_.pool, one_.pool),
		    .states = own(zero_.states, one_.states),
		    .ring = host.ring,
		    .masks = host.masks,
		    .launch = launches_,
		    .card = static_cast<unsigned>(std::to_underlying(card)),
		    .unused = 0,
		};
	}

	/**
	 * Launches both cards' resident loops; each waits for the other before it serves its first ticket. A prefill chunk
	 * runs between a stop and the next start: the loops yield the SMs by exiting, because prefill's kernels need every
	 * SM with their own shared memory, and a relaunch costs microseconds against a chunk's second, with no second way out
	 * of the loop to keep. Returns once both launches are enqueued.
	 */
	[[nodiscard]] auto start() -> core::Result<void, DecodeError> {
		core::invariant(loops_ == Loops::Unlaunched || loops_ == Loops::Halted);
		if (loops_ == Loops::Halted) {
			if (auto renewed = session_.renew(devices_); !renewed) {
				return renewed;
			}
		}
		++launches_;
		auto launch = [&]<image::Catalog Of>(Card<Of>& card, model::Card which) -> core::Result<void, DecodeError> {
			auto const& context = devices_.context(which);
			auto event = context.create_event();
			if (!event) {
				return fail(event.error());
			}
			auto launched = context.template launch<device_layout::Loop>(card.program, card.stream, parameters(which)).and_then([&] {
				return event->record(card.stream);
			});
			if (!launched) {
				return fail(launched.error());
			}
			card.launched = std::move(*event);
			return {};
		};
		return launch(zero_, model::Card::Zero).and_then([&] {
			loops_ = Loops::Running;
			return launch(one_, model::Card::One);
		});
	}

	/** What prefill borrows of card `card` while the loops are stopped. */
	[[nodiscard]] auto memories(model::Card card) const -> Memories {
		auto const params = parameters(card);
		return {
		    .image = params.image,
		    .embedding = params.embedding,
		    .pool = params.pool,
		    .tables = params.workspace + device_layout::workspace.tables,
		    .states = params.states,
		    .rotations = params.workspace + device_layout::workspace.exchange.rope_table,
		    .prompt_row = params.workspace + device_layout::workspace.prompt,
		};
	}

	/**
	 * Card `which`'s context and stream, borrowed for the caller's synchronous call (prefill's chunks run on them while
	 * the loops are stopped).
	 */
	[[nodiscard]] auto card(model::Card which) const -> CardAccess {
		auto const& stream = which == model::Card::Zero ? zero_.stream : one_.stream;
		return {.context = devices_.context(which), .stream = stream};
	}

	auto apply(PoolCommand const& command) -> void {
		session_.apply(command);
	}

	/** Submits the batch's cycle (and the queued commands ahead of it) and returns at once. */
	[[nodiscard]] auto begin(sched::Batch batch) -> core::Result<Pending, DecodeError> {
		core::invariant(loops_ == Loops::Running);
		return session_.begin(std::move(batch));
	}

	/** Submits a prompt's end (and the queued commands ahead of it) and returns at once. */
	[[nodiscard]] auto begin_prompt_end(sched::Row prompt, model::DraftDepth depth) -> core::Result<Pending, DecodeError> {
		core::invariant(loops_ == Loops::Running);
		return session_.begin_prompt_end(std::move(prompt), depth);
	}

	template<MaskRows M>
	[[nodiscard]] auto publish(Pending const& pending, M const& masks, std::size_t count) -> core::Result<void, DecodeError> {
		return session_.publish(pending, masks, count);
	}

	/** The batch's outcome once both cards have answered; nullopt while either has not. Never blocks. */
	[[nodiscard]] auto try_complete(Pending const& pending) -> std::optional<core::Result<sched::BatchDone, DecodeError>> {
		return session_.try_complete(pending);
	}

	/** The prompt's sampled token once both cards have answered; nullopt while either has not. Never blocks. */
	[[nodiscard]] auto try_complete_prompt_end(Pending const& pending) -> std::optional<core::Result<sched::ChunkDone, DecodeError>> {
		return session_.try_complete_prompt_end(pending);
	}

	/**
	 * Stops both loops with nothing begun, after applying every queued command, so a prefill that follows sees what the
	 * scheduler set up; done once both have exited and their kernels have finished. nullopt until then. Never blocks. A
	 * failure leaves the loops for `try_shutdown`, the one call that ends them after an error.
	 */
	[[nodiscard]] auto try_stop() -> std::optional<core::Result<void, DecodeError>> {
		core::invariant(loops_ == Loops::Running);
		auto exited = session_.try_stop();
		if (!exited) {
			return std::nullopt;
		}
		if (!*exited) {
			return exited;
		}
		auto done = finished();
		if (done && *done) {
			loops_ = Loops::Halted;
		}
		return done;
	}

	/**
	 * Ends the loops from wherever they stand, for teardown, and is the one way out after an error: never launched or
	 * halted is an immediate success; running loops are shut down as the session's `try_shutdown` says, and then their
	 * kernels awaited. Past `give_up` they are abandoned with `Abandoned`. A failure is kept and returned again. nullopt
	 * while it is still waiting. Never blocks.
	 */
	[[nodiscard]] auto try_shutdown(std::chrono::steady_clock::time_point give_up) -> std::optional<core::Result<void, DecodeError>> {
		switch (loops_) {
		case Loops::Unlaunched:
		case Loops::Halted:
			return core::Result<void, DecodeError>{};
		case Loops::Failed:
		case Loops::Abandoned:
			return std::unexpected(*failure_);
		case Loops::Running:
			break;
		}
		auto ended = session_.try_shutdown(give_up);
		if (!ended) {
			return std::nullopt;
		}
		auto done = finished();
		if (done && *done) {
			if (*ended) {
				loops_ = Loops::Halted;
				return ended;
			}
			// The kernels have finished, so a ticket the session gave up on can no longer be written to.
			auto error = ended->error();
			if (auto const abandoned = core::alternative<Abandoned>(error.reason)) {
				abandoned->kernels = Kernels::Finished;
			}
			return end(Loops::Failed, std::move(error));
		}
		if (!done && std::chrono::steady_clock::now() < give_up) {
			return std::nullopt;
		}
		// A kernel never finished past the deadline, or whether it has cannot be told: it may still run, and the engine
		// keeps everything it could write. The ticket the session gave up on, if it did, is named.
		auto const gave_up = *ended ? std::nullopt : core::alternative<Abandoned>(ended->error().reason);
		return end(Loops::Abandoned,
		           DecodeError{.reason = Abandoned{.ticket = gave_up ? gave_up->ticket : std::nullopt, .kernels = Kernels::MayRun}});
	}

	[[nodiscard]] auto session() const -> Session const& {
		return session_;
	}

private:
	Engine(device::Devices devices, HostImage<typename C::Host> host, Card<typename C::CardZero> zero, Card<typename C::CardOne> one, Session session)
	    : devices_{std::move(devices)}, host_{std::move(host)}, zero_{std::move(zero)}, one_{std::move(one)}, session_{std::move(session)} {}

	/** The loops end as `loops`, failed with `error`, which later shutdowns return. */
	[[nodiscard]] auto end(Loops loops, DecodeError error) -> std::optional<core::Result<void, DecodeError>> {
		loops_ = loops;
		failure_ = error;
		return std::unexpected(std::move(error));
	}

	/** Success once both kernels have finished, nullopt while either runs. */
	[[nodiscard]] auto finished() const -> std::optional<core::Result<void, DecodeError>> {
		// A kernel has finished when the event recorded after its launch has completed.
		auto const done = [](std::optional<device::Event> const& launched) -> std::optional<core::Result<void, DecodeError>> {
			if (!launched) {
				return core::Result<void, DecodeError>{};
			}
			auto const state = launched->query();
			if (!state) {
				return fail(state.error());
			}
			return *state == device::EventState::Pending ? std::nullopt : std::optional{core::Result<void, DecodeError>{}};
		};
		if (auto const zero = done(zero_.launched); !zero || !*zero) {
			return zero;
		}
		if (auto const one = done(one_.launched); !one || !*one) {
			return one;
		}
		return core::Result<void, DecodeError>{};
	}

	/** Everything the engine owns on the cards and in pinned memory: what an abandoned engine keeps unfreed. */
	struct Remains {
		device::Devices devices;
		HostImage<typename C::Host> host;
		Card<typename C::CardZero> zero;
		Card<typename C::CardOne> one;
		Session session;
	};

	// Declared first, so it dies last: every member below lives in its contexts.
	device::Devices devices_;
	HostImage<typename C::Host> host_;
	Card<typename C::CardZero> zero_;
	Card<typename C::CardOne> one_;
	Session session_;
	Loops loops_ = Loops::Unlaunched;
	/** Launches of the loops so far: the epoch the cards meet on at each start. */
	std::uint64_t launches_ = 0;
	/** Why the loops failed or were abandoned, once they have. */
	std::optional<DecodeError> failure_;
};

/** The engine over the model's images. */
export using Decode = Engine<ModelCatalogs>;

}
