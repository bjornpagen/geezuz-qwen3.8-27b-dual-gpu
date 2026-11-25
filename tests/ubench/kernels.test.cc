// SPDX-License-Identifier: AGPL-3.0-only
//
// The host's kernel declarations against the images they name: every declared kernel loads from its image through
// the driver double (whose jit reads the entries and parameter sizes the ptx declares), and every budget the host
// launches with is the budget record the kernel's own translation unit emitted.
import std;
import engine.device;
import engine.driver_double;
import engine.model;
import engine.prims;
import engine.testing;
import engine.ubench_device;

namespace {

namespace device = engine::device;
namespace fake = engine::driver_double;
namespace kernels = engine::ubench_device;
namespace ub = engine::device::ubench;

using engine::model::Card;

template<class... Ks>
[[nodiscard]] auto loads(device::Context const& context) -> bool {
	return context.load<Ks...>().has_value();
}

template<template<class> class Kernel, class Sweep>
[[nodiscard]] auto sweep_loads(device::Context const& context) -> bool {
	return std::apply(
	    [&](auto... shapes) {
		    return loads<Kernel<decltype(shapes)>...>(context);
	    },
	    Sweep{});
}

template<template<class> class Kernel, class Rows>
[[nodiscard]] auto rows_load(device::Context const& context) -> bool {
	return std::apply(
	    [&](auto... rows) {
		    return (sweep_loads<Kernel, decltype(rows)>(context) && ...);
	    },
	    Rows{});
}

/** Every kernel the drivers launch is in its image, with the one parameter struct its declaration builds. */
[[nodiscard]] auto every_declared_kernel_loads() -> bool {
	auto const driver = fake::Fake{fake::Machine{}};
	auto const opened = device::Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const& context = opened->context(Card::Zero);
	using ub::Wait;
	using ub::compute::Mix;
	using ub::link::Background;
	using ub::link::Flavor;
	using engine::device::prims::selfcheck::Function;
	auto const stream = sweep_loads<kernels::ReadKernel, kernels::ReadSweep>(context) &&
	                    loads<kernels::ReadKernel<kernels::RowMajorRead>, kernels::SustainKernel, kernels::L2PrefetchKernel, kernels::L2ProbeKernel,
	                          kernels::ReuseKernel<ub::stream::Priorities::Split>, kernels::ReuseKernel<ub::stream::Priorities::Uniform>>(context) &&
	                    rows_load<kernels::StallKernel, kernels::StallSweep>(context);
	auto const sync = loads<kernels::ClockKernel, kernels::MbarrierKernel<Wait::Backoff>, kernels::MbarrierKernel<Wait::Spin>, kernels::NamedBarrierKernel,
	                        kernels::CopyCompletionKernel, kernels::SleepGrainKernel, kernels::HopKernel<Wait::Backoff>, kernels::HopKernel<Wait::Spin>,
	                        kernels::StageHandoffKernel>(context);
	auto const compute =
	    loads<kernels::TensorKernel<Mix::Mma>, kernels::TensorKernel<Mix::MmaDecode>, kernels::TensorKernel<Mix::Decode>, kernels::TensorKernel<Mix::Widen>,
	          kernels::DecodeFoldedKernel, kernels::TranscendentalKernel<Function::Exp>, kernels::TranscendentalKernel<Function::Log>,
	          kernels::TranscendentalKernel<Function::Rsqrt>, kernels::TranscendentalKernel<Function::Sigmoid>,
	          kernels::TranscendentalKernel<Function::Silu>, kernels::TranscendentalKernel<Function::Softplus>,
	          kernels::LdmatrixKernel<ub::compute::Swizzle::Xor>, kernels::LdmatrixKernel<ub::compute::Swizzle::None>,
	          kernels::PipelineKernel<ub::compute::Pipelining::Double>, kernels::PipelineKernel<ub::compute::Pipelining::Sequential>,
	          kernels::ReductionKernel<ub::compute::Reduction::FloatButterfly>, kernels::ReductionKernel<ub::compute::Reduction::IntegerRedux>>(context);
	auto const link = loads<kernels::FlagKernel<kernels::FlagEnd::Ping, Flavor::ReleaseAcquire, Wait::Backoff>,
	                        kernels::FlagKernel<kernels::FlagEnd::Ping, Flavor::ReleaseAcquire, Wait::Spin>,
	                        kernels::FlagKernel<kernels::FlagEnd::Ping, Flavor::Relaxed, Wait::Spin>, kernels::FlagKernel<kernels::FlagEnd::Ping, Flavor::Fenced, Wait::Spin>,
	                        kernels::FlagKernel<kernels::FlagEnd::Pong, Flavor::ReleaseAcquire, Wait::Backoff>,
	                        kernels::FlagKernel<kernels::FlagEnd::Pong, Flavor::ReleaseAcquire, Wait::Spin>,
	                        kernels::FlagKernel<kernels::FlagEnd::Pong, Flavor::Relaxed, Wait::Spin>, kernels::FlagKernel<kernels::FlagEnd::Pong, Flavor::Fenced, Wait::Spin>,
	                        kernels::ReceiveKernel<4>, kernels::ReceiveKernel<16>, kernels::ReceiveKernel<64>, kernels::ReleaseKernel<Background::Idle>,
	                        kernels::ReleaseKernel<Background::PeerStores>, kernels::ReleaseKernel<Background::HostWrites>, kernels::MailboxKernel>(context) &&
	                  rows_load<kernels::PushKernel, kernels::PushSweep>(context) && rows_load<kernels::PullKernel, kernels::PushSweep>(context) &&
	                  std::apply(
	                      [&](auto... fetches) {
		                      return loads<decltype(fetches)...>(context);
	                      },
	                      kernels::FetchSweep{});
	auto const checks = loads<kernels::CheckDecodeKernel, kernels::CheckFragmentsKernel, kernels::CheckReduceKernel, kernels::DigestKernel,
	                          kernels::ProbeMmaKernel, kernels::StageRingCheckKernel, kernels::OrderCheckKernel>(context);
	auto const gemv = std::apply(
	                      [&](auto... sweep) {
		                      return loads<decltype(sweep)...>(context);
	                      },
	                      kernels::GemvSweep{}) &&
	                  loads<kernels::GemmPrefillKernel<kernels::PrefillMatrix::Down>, kernels::GemmPrefillKernel<kernels::PrefillMatrix::GateUp>>(context);
	return stream && sync && compute && link && checks && gemv && driver.violations().empty();
}

template<class K>
[[nodiscard]] constexpr auto recorded_as_declared() -> bool {
	auto const recorded = kernels::recorded_budget(K::image, K::entry);
	return recorded.threads == K::budget.threads && recorded.registers_per_thread == K::budget.registers_per_thread &&
	       recorded.shared_bytes == K::budget.shared_bytes && recorded.blocks_per_sm == K::budget.blocks_per_sm;
}

template<template<class> class Kernel, class Sweep>
[[nodiscard]] constexpr auto sweep_recorded() -> bool {
	return std::apply(
	    [](auto... shapes) {
		    return (recorded_as_declared<Kernel<decltype(shapes)>>() && ...);
	    },
	    Sweep{});
}

// The budget record parser against the shapes the device compiled the records from: every field of every kernel of
// the read sweep and a sample of every image, read back from the ptx text while compiling.
static_assert(sweep_recorded<kernels::ReadKernel, kernels::ReadSweep>());
static_assert(recorded_as_declared<kernels::ReadKernel<kernels::RowMajorRead>>() && recorded_as_declared<kernels::SustainKernel>());
static_assert(recorded_as_declared<kernels::HopKernel<ub::Wait::Spin>>() && recorded_as_declared<kernels::StageHandoffKernel>());
static_assert(recorded_as_declared<kernels::DecodeFoldedKernel>() && recorded_as_declared<kernels::TensorKernel<ub::compute::Mix::Mma>>());
static_assert(recorded_as_declared<kernels::ReleaseKernel<ub::link::Background::HostWrites>>() && recorded_as_declared<kernels::ReceiveKernel<64>>());
static_assert(recorded_as_declared<kernels::DigestKernel>());

// The prims component's self-check budgets, which the host takes from its shared header, are the ones its kernels
// record: the header and the kernels cannot have drifted apart.
static_assert(recorded_as_declared<kernels::CheckDecodeKernel>() && recorded_as_declared<kernels::CheckFragmentsKernel>());
static_assert(recorded_as_declared<kernels::CheckReduceKernel>() && recorded_as_declared<kernels::ProbeMmaKernel>());
static_assert(recorded_as_declared<kernels::StageRingCheckKernel>() && recorded_as_declared<kernels::OrderCheckKernel>());

/** The same law at run time over every declared stall and exchange kernel. */
[[nodiscard]] auto every_budget_is_its_record() -> bool {
	auto const rows = [](auto sweep, auto kernel_of) {
		return std::apply(
		    [&](auto... row) {
			    return (std::apply(
			                [&](auto... shapes) {
				                return (kernel_of(shapes) && ...);
			                },
			                decltype(row){}) &&
			            ...);
		    },
		    sweep);
	};
	auto const stall = rows(kernels::StallSweep{}, [](auto shape) {
		return recorded_as_declared<kernels::StallKernel<decltype(shape)>>();
	});
	auto const push = rows(kernels::PushSweep{}, [](auto shape) {
		return recorded_as_declared<kernels::PushKernel<decltype(shape)>>() && recorded_as_declared<kernels::PullKernel<decltype(shape)>>();
	});
	return stall && push;
}

/** A budget record is read for exactly the entry named, never for one whose name it merely begins. */
[[nodiscard]] auto records_match_whole_names() -> bool {
	constexpr auto contiguous = kernels::recorded_budget(kernels::stream_image, "ubench_read_p1_d64");
	constexpr auto row_major = kernels::recorded_budget(kernels::stream_image, "ubench_read_p1_d64_row_major");
	return contiguous.threads == ub::stream::Canonical::budget.threads && row_major.threads == kernels::RowMajorRead::budget.threads &&
	       contiguous.shared_bytes == row_major.shared_bytes;
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{.name = "every declared kernel loads from its image with its one parameter struct", .holds = every_declared_kernel_loads()},
	    engine::testing::Case{.name = "every stall, push and pull kernel launches with the budget its image records", .holds = every_budget_is_its_record()},
	    engine::testing::Case{.name = "a budget record is found by its whole entry name", .holds = records_match_whole_names()},
	};
	return engine::testing::report(cases);
}
