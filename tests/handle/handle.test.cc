// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.handle;
import engine.testing;

namespace {

struct ProbeHandle {
	std::uint64_t bits;

	[[nodiscard]] constexpr auto operator==(ProbeHandle const&) const -> bool = default;
};

static_assert(engine::handle::Handle<ProbeHandle>);
static_assert(!engine::handle::Handle<std::uint64_t>);

/**
 * The oracle: what a generational table promises, stated as plain containers. Live handles map to their values;
 * every handle ever released stays dead; a slot is reused until its generation is exhausted.
 */
struct Model {
	std::map<std::uint64_t, int> live;
	std::set<std::uint64_t> dead;
	std::size_t capacity;
	std::size_t retired;
};

/** `handles()` lists exactly the model's live handles, in slot order, each once. */
template<class Table>
[[nodiscard]] auto handles_match(Table const& table, Model const& model) -> bool {
	auto listed = std::vector<std::uint64_t>{};
	for (auto const handle : table.handles()) {
		listed.push_back(handle.bits);
	}
	auto const index = [](std::uint64_t bits) {
		return bits & ((std::uint64_t{1} << Table::index_bits) - 1);
	};
	auto const live = model.live | std::views::keys | std::ranges::to<std::set>();
	return listed.size() == live.size() && std::ranges::to<std::set>(listed) == live && std::ranges::is_sorted(listed, std::less{}, index);
}

template<std::size_t Capacity, std::uint8_t GenerationBits>
[[nodiscard]] auto agrees_with_model(std::uint64_t seed, std::size_t steps) -> bool {
	using Table = engine::handle::Table<int, ProbeHandle, Capacity, GenerationBits>;
	auto table = std::make_unique<Table>();
	auto model = Model{.live = {}, .dead = {}, .capacity = Capacity, .retired = 0};
	auto rng = engine::testing::generator(seed);
	auto holds = true;
	for (auto step = std::size_t{0}; step < steps && holds; ++step) {
		auto const choice = rng() % 3;
		if (choice == 0) {
			auto const value = static_cast<int>(rng() % 1000);
			auto const handle = table->insert(value);
			auto const room = model.live.size() + model.retired < model.capacity;
			holds = holds && handle.has_value() == room;
			if (handle) {
				holds = holds && !model.live.contains(handle->bits) && !model.dead.contains(handle->bits);
				model.live.emplace(handle->bits, value);
			}
		} else if (choice == 1 && !model.live.empty()) {
			auto const victim = std::next(model.live.begin(), static_cast<std::ptrdiff_t>(rng() % model.live.size()));
			auto const bits = victim->first;
			auto const generation = bits >> Table::index_bits;
			table->release(ProbeHandle{bits});
			model.live.erase(victim);
			model.dead.insert(bits);
			model.retired += generation == Table::generation_limit ? 1 : 0;
		} else if (!model.dead.empty()) {
			auto const stale = std::next(model.dead.begin(), static_cast<std::ptrdiff_t>(rng() % model.dead.size()));
			holds = holds && !table->try_get(ProbeHandle{*stale}).has_value();
		}
		for (auto const& [bits, value] : model.live) {
			auto const found = std::as_const(*table).try_get(ProbeHandle{bits});
			holds = holds && found.has_value() && *found == value;
		}
		holds = holds && table->live_count() == model.live.size() && table->retired_count() == model.retired;
		holds = holds && handles_match(*table, model);
	}
	return holds;
}

[[nodiscard]] auto exhausted_slot_retires() -> bool {
	auto table = engine::handle::Table<int, ProbeHandle, 1, 2>{};
	auto handles = std::vector<ProbeHandle>{};
	while (auto const handle = table.insert(7)) {
		handles.push_back(*handle);
		table.release(*handle);
	}
	auto const none_valid = std::ranges::none_of(handles, [&](ProbeHandle handle) {
		return table.try_get(handle).has_value();
	});
	return handles.size() == 3 && none_valid && table.retired_count() == 1 && !table.insert(8).has_value();
}

[[nodiscard]] auto forged_index_past_capacity_is_rejected() -> bool {
	auto table = engine::handle::Table<int, ProbeHandle, 3, 8>{};
	std::ignore = table.insert(1);
	std::ignore = table.insert(2);
	std::ignore = table.insert(3);
	return !table.try_get(ProbeHandle{(std::uint64_t{1} << std::uint64_t{2}) | std::uint64_t{3}}).has_value();
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "a full-width table agrees with the model over 20000 random operations (seed 1)",
	        .holds = agrees_with_model<5, 40>(1, 20000),
	    },
	    engine::testing::Case{
	        .name = "a table with 3-bit generations agrees with the model, through retirement (seed 2)",
	        .holds = agrees_with_model<4, 3>(2, 20000),
	    },
	    engine::testing::Case{
	        .name = "a table with 2-bit generations agrees with the model until every slot retires (seed 3)",
	        .holds = agrees_with_model<8, 2>(3, 5000),
	    },
	    engine::testing::Case{
	        .name = "a power-of-two table agrees with the model (seed 4)",
	        .holds = agrees_with_model<16, 4>(4, 20000),
	    },
	    engine::testing::Case{
	        .name = "an exhausted slot retires and never validates any of its handles",
	        .holds = exhausted_slot_retires(),
	    },
	    engine::testing::Case{
	        .name = "a handle naming an index past the capacity finds nothing",
	        .holds = forged_index_past_capacity_is_rejected(),
	    },
	};
	return engine::testing::report(cases);
}
