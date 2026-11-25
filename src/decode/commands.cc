// SPDX-License-Identifier: AGPL-3.0-only
//
// The kv pool's commands as the decode kernel applies them. The kernel, not the host, applies them: the pool and the
// block tables are device memory the resident kernel reads every cycle, and the driver's copies move whole
// allocations only, so no host-issued copy can address one block of the pool or one table entry; a command list in the
// ticket is applied by every SM before the cycle that follows it, ordered with the kernel's own reads by construction.
// A snapshot becomes one command per block of it, and the list is cut into waves of commands that touch nothing in
// common, so the kernel spreads a wave over every SM and puts a grid barrier only between waves.
export module engine.decode:commands;

import std;
import engine.core;
import engine.decode_device;
import engine.kv;
import engine.model;

namespace engine::decode {

export using PoolCommand = kv::Command<kv::model_geometry>;

using decode_device::Action;
using DeviceCommand = decode_device::Command;

[[nodiscard]] constexpr auto command(Action action, std::size_t slot, std::uint64_t first, std::uint64_t second, std::uint64_t third)
    -> DeviceCommand {
	return {
	    .action = action,
	    .slot = core::narrow<unsigned char>(slot),
	    .wave = 0,
	    .first = core::narrow<unsigned>(first),
	    .second = core::narrow<unsigned>(second),
	    .third = core::narrow<unsigned>(third),
	};
}

/** The device commands one pool command becomes, in order. */
/** Lowers each pool command; a snapshot is one command per block, in part order. */
struct Lowering {
	[[nodiscard]] auto operator()(kv::ZeroBlock const& zero) const -> std::vector<DeviceCommand> {
		return {command(Action::ZeroBlock, 0, zero.block, 0, 0)};
	}

	[[nodiscard]] auto operator()(kv::CopyPrefix const& copy) const -> std::vector<DeviceCommand> {
		core::invariant(copy.positions <= model::kv_block_tokens);
		return {command(Action::CopyPrefix, 0, copy.source, copy.target, copy.positions)};
	}

	[[nodiscard]] auto operator()(kv::SetTableEntry const& entry) const -> std::vector<DeviceCommand> {
		core::invariant(entry.entry < model::kv_table_blocks);
		return {command(Action::SetTableEntry, entry.slot.index(), entry.block, entry.entry, 0)};
	}

	[[nodiscard]] auto operator()(kv::SaveSnapshot<kv::model_geometry> const& save) const -> std::vector<DeviceCommand> {
		return parts(Action::SaveSnapshotPart, save.slot, save.blocks);
	}

	[[nodiscard]] auto operator()(kv::LoadSnapshot<kv::model_geometry> const& load) const -> std::vector<DeviceCommand> {
		return parts(Action::LoadSnapshotPart, load.slot, load.blocks);
	}

	[[nodiscard]] static auto parts(Action action, kv::Slot slot, std::span<kv::BlockIndex const> blocks) -> std::vector<DeviceCommand> {
		auto lowered = std::vector<DeviceCommand>{};
		for (auto const [part, block] : std::views::enumerate(blocks)) {
			lowered.push_back(command(action, slot.index(), block, core::narrow<std::uint64_t>(part), 0));
		}
		return lowered;
	}
};

/** The device commands one pool command becomes, in order. */
export [[nodiscard]] auto lower(PoolCommand const& pool) -> std::vector<DeviceCommand> {
	return std::visit(Lowering{}, pool);
}

/** What a command reads or writes, as one key: a pool block, one part of a slot's state image, or one table entry. */
enum class Touched : std::uint8_t { Block, State, Table };

[[nodiscard]] constexpr auto key(Touched what, std::uint64_t slot, std::uint64_t index) -> std::uint64_t {
	return (std::uint64_t{std::to_underlying(what)} << 56U) | (slot << 48U) | index;
}

[[nodiscard]] constexpr auto touches(DeviceCommand const& command) -> std::array<std::optional<std::uint64_t>, 2> {
	switch (command.action) {
	case Action::ZeroBlock:
		return {key(Touched::Block, 0, command.first), std::nullopt};
	case Action::CopyPrefix:
		return {key(Touched::Block, 0, command.first), key(Touched::Block, 0, command.second)};
	case Action::SetTableEntry:
		return {key(Touched::Table, command.slot, command.second), std::nullopt};
	case Action::SaveSnapshotPart:
	case Action::LoadSnapshotPart:
		return {key(Touched::Block, 0, command.first), key(Touched::State, command.slot, command.second)};
	case Action::Replay:
		// A replay runs alone in its wave (`waves` sees to it): it touches every part of the slot's state.
		return {std::nullopt, std::nullopt};
	}
	std::unreachable();
}

/** A command and the wave it runs in, counted over the whole list. */
export struct Waved {
	DeviceCommand command;
	std::uint64_t wave;
};

/**
 * The commands in order, each given the first wave after every earlier command it shares something with: a greedy cut
 * that keeps the order, so a command that reads what an earlier one writes runs a grid barrier after it.
 */
export [[nodiscard]] auto waves(std::span<DeviceCommand const> commands) -> std::vector<Waved> {
	auto waved = std::vector<Waved>{};
	waved.reserve(commands.size());
	auto wave = std::uint64_t{0};
	auto current = std::set<std::uint64_t>{};
	auto after_replay = false;
	for (auto const& each : commands) {
		auto const touched = touches(each);
		auto const replay = each.action == Action::Replay;
		auto const clashes = std::ranges::any_of(touched, [&](std::optional<std::uint64_t> const& what) {
			return what && current.contains(*what);
		});
		if (clashes || ((replay || after_replay) && !waved.empty())) {
			++wave;
			current.clear();
		}
		for (auto const& what : touched) {
			if (what) {
				current.insert(*what);
			}
		}
		after_replay = replay;
		waved.push_back({.command = each, .wave = wave});
	}
	return waved;
}

/**
 * The waved list cut into tickets of at most `max_commands`, each ticket's waves counted from its own first, so the
 * count fits a ticket whatever the list's length. A wave cut by a ticket's end loses nothing: tickets are applied in
 * order, a grid barrier between them.
 */
export [[nodiscard]] auto tickets(std::span<Waved const> waved) -> std::vector<std::vector<DeviceCommand>> {
	auto cut = std::vector<std::vector<DeviceCommand>>{};
	for (auto start = std::size_t{0}; start < waved.size(); start += decode_device::max_commands) {
		auto const piece = waved.subspan(start, std::min(decode_device::max_commands, waved.size() - start));
		auto ticket = std::vector<DeviceCommand>{};
		ticket.reserve(decode_device::max_commands);
		auto const first = piece.front().wave;
		for (auto const& each : piece) {
			auto command = each.command;
			command.wave = core::narrow<unsigned short>(each.wave - first);
			ticket.push_back(command);
		}
		cut.push_back(std::move(ticket));
	}
	return cut;
}

}
