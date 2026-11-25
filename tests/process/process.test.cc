// SPDX-License-Identifier: AGPL-3.0-only
//
// Child processes against programs every darwin and linux machine has, and this test run as its own child: output
// comes back line by line in order, the exit status and the signal that ended a child are reported exactly, a deadline
// ends every wait, a line past the bound is refused, a missing program is refused by name, dropping a running child
// does not wait for it, and this process's own output reaches a pipe as it is written.
import std;
import engine.core;
import engine.process;
import engine.testing;

namespace {

namespace process = engine::process;

using Clock = std::chrono::steady_clock;

[[nodiscard]] auto after(std::chrono::milliseconds wait) -> Clock::time_point {
	return Clock::now() + wait;
}

constexpr auto patience = std::chrono::milliseconds{10'000};

[[nodiscard]] auto spawn(std::string_view program, std::vector<std::string> const& arguments)
    -> engine::core::Result<process::Child, process::ProcessError> {
	return process::Child::spawn(std::filesystem::path{program}, arguments);
}

[[nodiscard]] auto shell(std::string const& script) -> engine::core::Result<process::Child, process::ProcessError> {
	return spawn("/bin/sh", {"-c", script});
}

/** Every read until the output closes, or nothing if a read fails or the deadline passes first. */
[[nodiscard]] auto lines(process::Child& child) -> std::optional<std::vector<std::string>> {
	auto out = std::vector<std::string>{};
	auto const deadline = after(patience);
	for (;;) {
		auto read = child.read_line_until(deadline);
		if (!read || std::holds_alternative<process::Quiet>(*read)) {
			return std::nullopt;
		}
		if (std::holds_alternative<process::Closed>(*read)) {
			return out;
		}
		out.push_back(std::get<process::Line>(std::move(*read)).text);
	}
}

[[nodiscard]] auto exited_with(process::Child& child, int status) -> bool {
	auto const exit = child.wait_until(after(patience));
	return exit && *exit == std::optional<process::Exit>{process::Exited{.status = status}};
}

[[nodiscard]] auto arguments_reach_the_child() -> bool {
	auto child = spawn("/bin/echo", {"one  two", "three"});
	return child && lines(*child) == std::vector<std::string>{"one  two three"} && exited_with(*child, 0);
}

[[nodiscard]] auto lines_arrive_in_order() -> bool {
	auto child = shell(R"(printf 'a\n\nb\nlast')");
	return child && lines(*child) == std::vector<std::string>{"a", "", "b", "last"} && exited_with(*child, 0);
}

[[nodiscard]] auto exit_status_is_reported() -> bool {
	return std::ranges::all_of(std::array{0, 3, 77}, [](int status) {
		auto child = shell(std::format("exit {}", status));
		// The same exit again: a reaped child is not waited for twice.
		return child && exited_with(*child, status) && exited_with(*child, status);
	});
}

[[nodiscard]] auto deadlines_end_waits_and_terminate_signals() -> bool {
	auto child = spawn("/bin/sleep", {"30"});
	if (!child) {
		return false;
	}
	auto const quiet = child->read_line_until(after(std::chrono::milliseconds{50}));
	auto const running = child->wait_until(after(std::chrono::milliseconds{50}));
	auto const signalled = child->terminate().has_value();
	auto const ended = child->wait_until(after(patience));
	return quiet && std::holds_alternative<process::Quiet>(*quiet) && running && !running->has_value() && signalled && ended &&
	       *ended == std::optional<process::Exit>{process::Signalled{.signal = process::termination_signal}} &&
	       child->terminate().has_value();
}

[[nodiscard]] auto an_overlong_line_is_refused() -> bool {
	auto child = shell(std::format("printf '%{}s' x", process::max_line_bytes + 1));
	if (!child) {
		return false;
	}
	auto const read = child->read_line_until(after(patience));
	return !read && read.error().cause == process::Cause::LineTooLong && exited_with(*child, 0);
}

[[nodiscard]] auto a_line_at_the_bound_is_read() -> bool {
	auto child = shell(std::format("printf '%{}s\\n' x", process::max_line_bytes));
	if (!child) {
		return false;
	}
	auto const read = lines(*child);
	return read && read->size() == 1 && read->front().size() == process::max_line_bytes && exited_with(*child, 0);
}

[[nodiscard]] auto a_missing_program_is_refused() -> bool {
	auto const child = spawn("/nonexistent/geezuz-process-test", {});
	return !child && child.error().stage == process::Stage::Spawn && child.error().cause == process::Cause::NotFound;
}

/** Dropping a running child kills it: the drop returns long before the child would have ended on its own. */
[[nodiscard]] auto dropping_a_running_child_kills_it() -> bool {
	auto const started = Clock::now();
	{
		auto const child = spawn("/bin/sleep", {"30"});
		if (!child) {
			return false;
		}
	}
	return Clock::now() - started < std::chrono::seconds{10};
}

/** The line this test's own binary says when run as `speak`, before it waits to be killed. */
constexpr auto spoken = std::string_view{"said before exit"};

/**
 * This process's output reaches a pipe as it is written: the child says a line and then waits, and the line arrives
 * long before the child would end (a buffered line would arrive only at its exit).
 */
[[nodiscard]] auto output_is_flushed(std::filesystem::path const& self) -> bool {
	auto child = process::Child::spawn(self, std::vector<std::string>{"speak"});
	if (!child) {
		return false;
	}
	auto const read = child->read_line_until(after(std::chrono::milliseconds{5'000}));
	return read && *read == process::Read{process::Line{.text = std::string{spoken}}};
}

}

auto main(int argc, char** argv) -> int {
	using engine::testing::Case;
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	if (arguments.size() == 2 && std::string_view{arguments[1]} == "speak") {
		process::output_line(spoken);
		std::this_thread::sleep_until(Clock::now() + std::chrono::seconds{30});
		return 0;
	}
	auto const cases = std::array{
	    Case{.name = "arguments reach the child, and its output comes back as a line", .holds = arguments_reach_the_child()},
	    Case{.name = "lines arrive in order, empty and unterminated ones included", .holds = lines_arrive_in_order()},
	    Case{.name = "the exit status is reported, the same on every wait", .holds = exit_status_is_reported()},
	    Case{.name = "deadlines end reads and waits; terminate ends the child by SIGTERM",
		     .holds = deadlines_end_waits_and_terminate_signals()},
	    Case{.name = "a line past the bound is refused", .holds = an_overlong_line_is_refused()},
	    Case{.name = "a line exactly at the bound is read", .holds = a_line_at_the_bound_is_read()},
	    Case{.name = "a missing program is refused as not found", .holds = a_missing_program_is_refused()},
	    Case{.name = "dropping a running child kills it", .holds = dropping_a_running_child_kills_it()},
	    Case{.name = "this process's output reaches a pipe as it is written",
		     .holds = output_is_flushed(std::filesystem::path{arguments[0]})},
	};
	return engine::testing::report(cases);
}
