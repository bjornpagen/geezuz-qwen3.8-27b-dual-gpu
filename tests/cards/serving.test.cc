// SPDX-License-Identifier: AGPL-3.0-only
//
// The engine as the programs start it, on the machine without the cards (the development mac): loading fails with
// the bring-up's own refusal, the driver's, and the server's engine slot says so three ways that must agree: the
// reason it reports at once, the `EngineFailed` it posts to the service (and nothing else), and the failure its stop
// returns.
import std;
import engine.cards;
import engine.core;
import engine.driver;
import engine.http;
import engine.net;
import engine.openai;
import engine.testing;

namespace {

namespace cards = engine::cards;
namespace net = engine::net;
namespace openai = engine::openai;

/** The driver's refusal on a machine without it, as the bring-up reports it. */
[[nodiscard]] auto driver_unavailable(cards::LoadError const& error) -> bool {
	auto const start = engine::core::alternative<cards::StartError>(error.reason);
	auto const device = start.and_then([](cards::StartError const& failed) {
		return engine::core::alternative<engine::driver::DeviceError>(failed.reason);
	});
	return device && std::holds_alternative<engine::driver::DriverUnavailable>(device->reason);
}

/** Every update posted until `deadline`. */
[[nodiscard]] auto updates_until(net::Poller& poller, net::Mailbox<openai::Update, engine::http::max_posted_messages>& mailbox,
                                 net::Deadline deadline) -> std::vector<openai::Update> {
	auto updates = std::vector<openai::Update>{};
	while (net::Clock::now() < deadline) {
		mailbox.drain([&](openai::Update&& update) {
			updates.push_back(std::move(update));
		});
		if (!updates.empty()) {
			return updates;
		}
		if (!poller.wait_until(deadline)) {
			return updates;
		}
	}
	return updates;
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const images = std::filesystem::current_path() / "serving.scratch";
	std::filesystem::create_directories(images);
	auto poller = net::Poller::open();
	if (!poller) {
		std::println("FAIL: no poller");
		return 1;
	}
	auto mailbox = net::Mailbox<openai::Update, engine::http::max_posted_messages>{poller->waker()};
	auto const poster = engine::http::Poster<openai::Update>{mailbox};

	auto const loaded = cards::load(images, poster);
	auto const refused = !loaded && driver_unavailable(loaded.error());
	auto const expected = loaded ? std::string{} : cards::describe(loaded.error());

	auto reported = std::string{};
	auto engine = cards::Engine::start(images, poster, [&reported](std::string_view reason) {
		reported = reason;
	});
	auto const updates =
	    engine ? updates_until(*poller, mailbox, net::Clock::now() + std::chrono::seconds{30}) : std::vector<openai::Update>{};
	auto const ended = engine ? engine->stop() : engine::core::Result<void, cards::ServingError>{};
	auto later = std::size_t{0};
	mailbox.drain([&later](openai::Update const& /*update*/) {
		later += 1;
	});
	auto const posted = updates.size() == 1 && later == 0 && updates.front() == openai::Update{openai::EngineFailed{.reason = expected}};
	auto const stopped = !loaded && !ended && ended.error() == cards::ServingError{.reason = loaded.error()};

	auto const cases = std::array{
	    Case{.name = "loading fails with the driver's typed refusal", .holds = refused},
	    Case{.name = "the slot reports that failure at once, as /health will give it", .holds = refused && reported == expected},
	    Case{.name = "the slot posts the failure to the service, and nothing else", .holds = refused && posted},
	    Case{.name = "stopping the slot returns the same failure", .holds = refused && stopped},
	};
	return engine::testing::report(cases);
}
