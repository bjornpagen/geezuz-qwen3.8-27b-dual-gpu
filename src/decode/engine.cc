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

/** Waits for a transfer and hands back its destination, or the copy's or the wait's failure. */
template<class Destination, class Source>
[[nodiscard]] auto finish(device::Transfer<Destination, Source> transfer, std::chrono::steady_clock::time_point deadline)
    -> core::Result<Destination, DecodeError> {
	if (transfer.wait_until(deadline) == device::EventState::Pending) {
		// The copy is still reading its source; the transfer may not be dropped while it does.
		core::invariant(transfer.wait_until(std::chrono::steady_clock::time_point::max()) == device::EventState::Complete);
		return fail(TimedOut{.waiting = Waiting::Answer, .ticket = 0});
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
                                 std::chrono::steady_clock::time_point deadline) -> core::Result<device::DeviceArray<T, Count>, DecodeError> {
	auto array = context.allocate<T, Count>();
	if (!array) {
		return fail(array.error());
	}
	return finish(context.upload(stream, std::move(*array), std::move(staged), 0), deadline);
}

/**
 * Attention's rope table in the workspace at `table`, filled once by its own kernel, which evaluates the model's
 * rotations; decode and prefill both read it.
 */
[[nodiscard]] auto fill_rotations(device::Context const& context, device::Stream const& stream, std::uint64_t table,
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
		return fail(TimedOut{.waiting = Waiting::Answer, .ticket = 0});
	}
	return {};
}

/** Zeroes a card's control block, the words its loop must find zero when it starts. */
[[nodiscard]] auto zero_control(device::Context const& context, device::Stream const& stream, Control control,
                                std::chrono::steady_clock::time_point deadline) -> core::Result<Control, DecodeError> {
	auto staged = context.allocate_mapped<device_layout::Control, 1>();
	if (!staged) {
		return fail(staged.error());
	}
	return finish(context.upload(stream, std::move(control), std::move(*staged), 0), deadline);
}

/** Opens, verifies and copies one card's image, and allocates and fills the rest of the card's memories. */
template<image::Catalog C>
[[nodiscard]] auto load_card(device::Context const& context, model::Card card, std::filesystem::path const& path, image::Checkpoint const& checkpoint,
                             std::chrono::steady_clock::time_point deadline) -> core::Result<Card<C>, DecodeError> {
	auto opened = image::Image<C>::open(path, checkpoint);
	if (!opened) {
		return fail(opened.error());
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
	auto copied = finish(context.upload(*stream, std::move(*weights), std::move(*pinned), 0), deadline);
	if (!copied) {
		return std::unexpected(copied.error());
	}
	auto tokens = context.allocate_mapped<std::uint32_t, model::vocab_per_card>();
	if (!tokens) {
		return fail(tokens.error());
	}
	std::ranges::copy(vocabulary::row_tokens(card), tokens->host().begin());
	auto row_tokens = upload_staged(context, *stream, std::move(*tokens), deadline);
	if (!row_tokens) {
		return std::unexpected(row_tokens.error());
	}
	auto control = context.allocate<device_layout::Control, 1>();
	auto workspace = control.and_then([&](auto const&) {
		return context.allocate<std::byte, device_layout::workspace.bytes>();
	});
	auto pool = workspace.and_then([&](auto const&) {
		return context.allocate<std::byte, model::pool_bytes>();
	});
	auto states = pool.and_then([&](auto const&) {
		return context.allocate<std::byte, model::slots * model::slot_state_bytes>();
	});
	if (!states) {
		return fail(states.error());
	}
	if (auto rotations = fill_rotations(context, *stream, workspace->address() + device_layout::workspace.exchange.rope_table, deadline); !rotations) {
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
 * Where the resident loops stand. Stopped loops have used up their rings (each carried an exit ticket), so the next
 * start takes fresh ones; loops never started still have theirs.
 */
enum class Loops : std::uint8_t {
	NeverStarted,
	Running,
	Stopped,
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
			return fail(host_image.error());
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
	      loops_{std::exchange(other.loops_, Loops::NeverStarted)} {}

	auto operator=(Engine&&) noexcept -> Engine& = delete;
	Engine(Engine const&) = delete;
	auto operator=(Engine const&) -> Engine& = delete;

	~Engine() {
		// A loop still running reads every allocation here: it must have been stopped.
		core::invariant(loops_ != Loops::Running);
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
		    .card = static_cast<unsigned>(std::to_underlying(card)),
		    .unused = 0,
		};
	}

	/**
	 * Launches both cards' resident loops; each waits for the other before it serves its first ticket. A prefill chunk
	 * runs between `stop` and the next `start`: the loops yield the SMs by exiting, because prefill's kernels need every
	 * SM with their own shared memory, and a relaunch costs microseconds against a chunk's second, with no second way out
	 * of the loop to keep.
	 */
	[[nodiscard]] auto start(std::chrono::steady_clock::time_point deadline) -> core::Result<void, DecodeError> {
		core::invariant(loops_ != Loops::Running);
		if (loops_ == Loops::Stopped) {
			if (auto renewed = session_.renew(devices_); !renewed) {
				return renewed;
			}
		}
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
		// Both control blocks read zero before either loop starts: a loop stores into its peer's at once.
		auto zeroed = [&]<image::Catalog Of>(Card<Of>& card, model::Card which) -> core::Result<void, DecodeError> {
			auto cleared = zero_control(devices_.context(which), card.stream, std::move(card.control), deadline);
			if (!cleared) {
				return std::unexpected(cleared.error());
			}
			card.control = std::move(*cleared);
			return {};
		};
		if (auto cleared = zeroed(zero_, model::Card::Zero).and_then([&] { return zeroed(one_, model::Card::One); }); !cleared) {
			return cleared;
		}
		auto started = launch(zero_, model::Card::Zero).and_then([&] {
			loops_ = Loops::Running;
			return launch(one_, model::Card::One);
		});
		return started;
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

	[[nodiscard]] auto begin_prompt_end(sched::Row prompt, model::DraftDepth depth, std::chrono::steady_clock::time_point deadline)
	    -> core::Result<Pending, DecodeError> {
		core::invariant(loops_ == Loops::Running);
		return session_.begin_prompt_end(std::move(prompt), depth, deadline);
	}

	[[nodiscard]] auto complete_prompt_end(Pending pending, std::chrono::steady_clock::time_point deadline)
	    -> core::Result<sched::ChunkDone, DecodeError> {
		return session_.complete_prompt_end(std::move(pending), deadline);
	}

	auto apply(PoolCommand const& command) -> void {
		session_.apply(command);
	}

	[[nodiscard]] auto flush(std::chrono::steady_clock::time_point deadline) -> core::Result<void, DecodeError> {
		return session_.flush(deadline);
	}

	[[nodiscard]] auto begin(sched::Batch batch, std::chrono::steady_clock::time_point deadline) -> core::Result<Pending, DecodeError> {
		core::invariant(loops_ == Loops::Running);
		return session_.begin(std::move(batch), deadline);
	}

	template<MaskRows M>
	[[nodiscard]] auto publish(Pending const& pending, M const& masks, std::size_t count) -> core::Result<void, DecodeError> {
		return session_.publish(pending, masks, count);
	}

	[[nodiscard]] auto complete(Pending pending, std::chrono::steady_clock::time_point deadline) -> core::Result<sched::BatchDone, DecodeError> {
		return session_.complete(std::move(pending), deadline);
	}

	/** Ends both loops and waits until their kernels have finished, after which nothing on the cards reads the engine. */
	[[nodiscard]] auto stop(std::chrono::steady_clock::time_point deadline) -> core::Result<void, DecodeError> {
		core::invariant(loops_ == Loops::Running);
		if (auto exited = session_.stop(deadline); !exited) {
			return exited;
		}
		auto const finished = [&](std::optional<device::Event> const& launched) -> core::Result<void, DecodeError> {
			if (!launched) {
				return {};
			}
			auto const state = launched->wait_until(deadline);
			if (!state) {
				return fail(state.error());
			}
			if (*state == device::EventState::Pending) {
				return fail(TimedOut{.waiting = Waiting::Answer, .ticket = 0});
			}
			return {};
		};
		if (auto both = finished(zero_.launched).and_then([&] { return finished(one_.launched); }); !both) {
			return both;
		}
		loops_ = Loops::Stopped;
		return {};
	}

	[[nodiscard]] auto session() const -> Session const& {
		return session_;
	}

private:
	Engine(device::Devices devices, HostImage<typename C::Host> host, Card<typename C::CardZero> zero, Card<typename C::CardOne> one, Session session)
	    : devices_{std::move(devices)}, host_{std::move(host)}, zero_{std::move(zero)}, one_{std::move(one)}, session_{std::move(session)} {}

	// Declared first, so it dies last: every member below lives in its contexts.
	device::Devices devices_;
	HostImage<typename C::Host> host_;
	Card<typename C::CardZero> zero_;
	Card<typename C::CardOne> one_;
	Session session_;
	Loops loops_ = Loops::NeverStarted;
};

/** The engine over the model's images. */
export using Decode = Engine<ModelCatalogs>;

}
