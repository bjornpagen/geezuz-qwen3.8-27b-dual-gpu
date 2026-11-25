// SPDX-License-Identifier: AGPL-3.0-only
//
// The pool's commands as the kernel applies them, over random command lists whose blocks, slots and entries collide
// often: lowering keeps every command's meaning and order (a snapshot becomes its parts in order); the waves keep the
// order, never put two commands that touch one block, state part or table entry in one wave, and open a wave only
// where the next command clashes with the current one; and the tickets cut the list without losing or moving a
// command, each counting its waves from zero.
import std;
import engine.decode;
import engine.decode_device;
import engine.kv;
import engine.model;
import engine.testing;

namespace {

namespace decode = engine::decode;
namespace dd = engine::decode_device;
namespace kv = engine::kv;
namespace model = engine::model;

using dd::Action;

[[nodiscard]] auto pool_command(std::mt19937_64& random) -> decode::PoolCommand {
	auto const block = [&] {
		return static_cast<kv::BlockIndex>(random() % 24);
	};
	auto const slot = [&] {
		return kv::Slot::at(random() % model::slots);
	};
	switch (random() % 5) {
	case 0:
		return kv::ZeroBlock{.block = block()};
	case 1:
		return kv::CopyPrefix{.source = block(), .target = block(), .positions = random() % (model::kv_block_tokens + 1)};
	case 2:
		return kv::SetTableEntry{.slot = slot(), .entry = random() % 16, .block = block()};
	case 3: {
		auto save = kv::SaveSnapshot<kv::model_geometry>{.slot = slot(), .blocks = {}};
		std::ranges::generate(save.blocks, block);
		return save;
	}
	default: {
		auto load = kv::LoadSnapshot<kv::model_geometry>{.slot = slot(), .blocks = {}};
		std::ranges::generate(load.blocks, block);
		return load;
	}
	}
}

/** What the test takes a command to touch: its own reading of each action, not the lowering's. */
[[nodiscard]] auto touched(dd::Command const& command) -> std::vector<std::tuple<int, unsigned, unsigned>> {
	switch (command.action) {
	case Action::ZeroBlock:
		return {{0, 0U, command.first}};
	case Action::CopyPrefix:
		return {{0, 0U, command.first}, {0, 0U, command.second}};
	case Action::SetTableEntry:
		return {{2, unsigned{command.slot}, command.second}};
	case Action::SaveSnapshotPart:
	case Action::LoadSnapshotPart:
		return {{0, 0U, command.first}, {1, unsigned{command.slot}, command.second}};
	case Action::Replay:
		return {};
	}
	std::unreachable();
}

/** Whether `lowered` is the one pool command it came from. */
struct Faithful {
	std::span<dd::Command const> lowered;

	[[nodiscard]] auto one(Action action) const -> bool {
		return lowered.size() == 1 && lowered[0].action == action;
	}

	[[nodiscard]] auto operator()(kv::ZeroBlock const& zero) const -> bool {
		return one(Action::ZeroBlock) && lowered[0].first == zero.block;
	}

	[[nodiscard]] auto operator()(kv::CopyPrefix const& copy) const -> bool {
		return one(Action::CopyPrefix) && lowered[0].first == copy.source && lowered[0].second == copy.target && lowered[0].third == copy.positions;
	}

	[[nodiscard]] auto operator()(kv::SetTableEntry const& entry) const -> bool {
		return one(Action::SetTableEntry) && lowered[0].slot == entry.slot.index() && lowered[0].second == entry.entry &&
		       lowered[0].first == entry.block;
	}

	[[nodiscard]] auto operator()(kv::SaveSnapshot<kv::model_geometry> const& save) const -> bool {
		return parts(Action::SaveSnapshotPart, save.slot, save.blocks);
	}

	[[nodiscard]] auto operator()(kv::LoadSnapshot<kv::model_geometry> const& load) const -> bool {
		return parts(Action::LoadSnapshotPart, load.slot, load.blocks);
	}

	[[nodiscard]] auto parts(Action action, kv::Slot slot, std::span<kv::BlockIndex const> blocks) const -> bool {
		auto holds = lowered.size() == blocks.size();
		for (auto part = std::size_t{0}; holds && part < lowered.size(); ++part) {
			holds = lowered[part].action == action && lowered[part].slot == slot.index() && lowered[part].first == blocks[part] &&
			        lowered[part].second == part;
		}
		return holds;
	}
};

[[nodiscard]] auto clash(dd::Command const& left, dd::Command const& right) -> bool {
	// A replay takes the whole of its slot's state and runs alone.
	if (left.action == Action::Replay || right.action == Action::Replay) {
		return true;
	}
	auto const a = touched(left);
	auto const b = touched(right);
	return std::ranges::any_of(a, [&](auto const& key) {
		return std::ranges::contains(b, key);
	});
}

[[nodiscard]] auto lowered_faithfully(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 2'000; ++trial) {
		auto const command = pool_command(random);
		auto const lowered = decode::lower(command);
		if (!std::visit(Faithful{.lowered = lowered}, command)) {
			return false;
		}
	}
	return true;
}

[[nodiscard]] auto random_list(std::mt19937_64& random) -> std::vector<dd::Command> {
	auto commands = std::vector<dd::Command>{};
	auto const count = random() % 60;
	for (auto index = std::uint64_t{0}; index < count; ++index) {
		if (random() % 8 == 0) {
			commands.push_back({.action = Action::Replay, .slot = static_cast<unsigned char>(random() % model::slots), .wave = 0,
			                    .first = static_cast<unsigned>(1 + (random() % 4)), .second = 0, .third = 0});
		}
		std::ranges::copy(decode::lower(pool_command(random)), std::back_inserter(commands));
	}
	return commands;
}

[[nodiscard]] auto same_command(dd::Command const& left, dd::Command const& right) -> bool {
	return left.action == right.action && left.slot == right.slot && left.first == right.first && left.second == right.second &&
	       left.third == right.third;
}

[[nodiscard]] auto waves_are_sound(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 300; ++trial) {
		auto const commands = random_list(random);
		auto const waved = decode::waves(commands);
		if (waved.size() != commands.size()) {
			return false;
		}
		for (auto index = std::size_t{0}; index < waved.size(); ++index) {
			if (!same_command(waved[index].command, commands[index])) {
				return false;
			}
			if (index == 0) {
				if (waved[0].wave != 0) {
					return false;
				}
				continue;
			}
			auto const step = waved[index].wave - waved[index - 1].wave;
			if (step > 1) {
				return false;
			}
			// Within its wave, nothing earlier clashes; a new wave opens only on a clash with the wave it closes.
			auto clashes = false;
			for (auto earlier = index; earlier > 0 && waved[earlier - 1].wave == waved[index - 1].wave; --earlier) {
				clashes = clashes || clash(waved[earlier - 1].command, waved[index].command);
			}
			if ((step == 0) == clashes) {
				return false;
			}
		}
	}
	return true;
}

[[nodiscard]] auto tickets_cut_cleanly(std::uint64_t seed) -> bool {
	auto random = engine::testing::generator(seed);
	for (auto trial = 0; trial < 40; ++trial) {
		auto commands = random_list(random);
		// Long lists too, so lists span several tickets.
		for (auto more = random() % 3; more > 0; --more) {
			auto const extra = random_list(random);
			for (auto repeat = 0; repeat < 10; ++repeat) {
				std::ranges::copy(extra, std::back_inserter(commands));
			}
		}
		auto const waved = decode::waves(commands);
		auto const cut = decode::tickets(waved);
		auto flat = std::size_t{0};
		for (auto const& ticket : cut) {
			if (ticket.empty() || ticket.size() > dd::max_commands || ticket.front().wave != 0) {
				return false;
			}
			for (auto index = std::size_t{0}; index < ticket.size(); ++index) {
				auto const& original = waved[flat + index];
				if (!same_command(ticket[index], original.command)) {
					return false;
				}
				if (index > 0 && (ticket[index].wave - ticket[index - 1].wave) != (original.wave - waved[flat + index - 1].wave)) {
					return false;
				}
			}
			flat += ticket.size();
		}
		if (flat != commands.size()) {
			return false;
		}
	}
	return true;
}

}

auto main() -> int {
	using engine::testing::Case;
	auto const cases = std::array{
	    Case{.name = "every pool command lowers to its meaning, snapshots part by part (seed 1)", .holds = lowered_faithfully(1)},
	    Case{.name = "waves keep order, never hold a clash, and open only on one (seed 2)", .holds = waves_are_sound(2)},
	    Case{.name = "waves keep order, never hold a clash, and open only on one (seed 3)", .holds = waves_are_sound(3)},
	    Case{.name = "tickets cut the list whole and in order, waves counted from zero (seed 4)", .holds = tickets_cut_cleanly(4)},
	};
	return engine::testing::report(cases);
}
