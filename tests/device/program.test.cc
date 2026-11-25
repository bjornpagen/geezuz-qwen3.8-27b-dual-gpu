// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.core;
import engine.device;
import engine.driver_double;
import engine.ptx;
import engine.testing;

namespace {

namespace device = engine::device;
namespace fake = engine::driver_double;

using device::Devices;
using device::Rank;
using fake::Fake;
using fake::Kind;
using fake::Machine;

constexpr auto probe_text = std::to_array(R"(//
.version 9.3
.target sm_86
.address_size 64

.visible .entry probe(
	.param .align 8 .b8 probe_param_0[24]
)
{
	ret;
}

.visible .entry wide(
	.param .align 8 .b8 wide_param_0[16]
)
{
	ret;
}

.visible .entry pair(
	.param .u64 pair_param_0,
	.param .u64 pair_param_1
)
{
	ret;
}
)");

constexpr auto probe_image = engine::ptx::image(probe_text);

struct ProbeParams {
	std::uint64_t ring;
	std::uint64_t out;
	std::uint32_t count;
	std::uint32_t flags;
};

struct WideParams {
	std::uint64_t first;
	std::uint64_t second;
};

/** A persistent-style kernel: co-resident, one block per SM, more shared memory than the default 48 KiB. */
struct Probe {
	static constexpr engine::ptx::Image image = probe_image;
	static constexpr std::string_view entry = "probe";
	static constexpr device::Dim grid = {.x = 82, .y = 1, .z = 1};
	static constexpr device::Dim block = {.x = 256, .y = 1, .z = 1};
	static constexpr std::uint32_t shared_bytes = 64 * 1024;
	static constexpr device::Residency residency = device::Residency::CoResident;
	using Params = ProbeParams;
};

struct Wide {
	static constexpr engine::ptx::Image image = probe_image;
	static constexpr std::string_view entry = "wide";
	static constexpr device::Dim grid = {.x = 4, .y = 2, .z = 1};
	static constexpr device::Dim block = {.x = 128, .y = 1, .z = 1};
	static constexpr std::uint32_t shared_bytes = 0;
	static constexpr device::Residency residency = device::Residency::Ordinary;
	using Params = WideParams;
};

struct Absent {
	static constexpr engine::ptx::Image image = probe_image;
	static constexpr std::string_view entry = "absent";
	static constexpr device::Dim grid = {.x = 1, .y = 1, .z = 1};
	static constexpr device::Dim block = {.x = 32, .y = 1, .z = 1};
	static constexpr std::uint32_t shared_bytes = 0;
	static constexpr device::Residency residency = device::Residency::Ordinary;
	using Params = WideParams;
};

/** Declares `probe` with a parameter struct of the wrong size. */
struct Narrow {
	static constexpr engine::ptx::Image image = probe_image;
	static constexpr std::string_view entry = "probe";
	static constexpr device::Dim grid = {.x = 1, .y = 1, .z = 1};
	static constexpr device::Dim block = {.x = 32, .y = 1, .z = 1};
	static constexpr std::uint32_t shared_bytes = 0;
	static constexpr device::Residency residency = device::Residency::Ordinary;
	using Params = WideParams;
};

/** Declares `pair`, which takes two parameters, as taking one struct. */
struct Pair {
	static constexpr engine::ptx::Image image = probe_image;
	static constexpr std::string_view entry = "pair";
	static constexpr device::Dim grid = {.x = 1, .y = 1, .z = 1};
	static constexpr device::Dim block = {.x = 32, .y = 1, .z = 1};
	static constexpr std::uint32_t shared_bytes = 0;
	static constexpr device::Residency residency = device::Residency::Ordinary;
	using Params = std::uint64_t;
};

/** More co-resident blocks than 82 SMs hold at 256 threads each. */
struct Crowded {
	static constexpr engine::ptx::Image image = probe_image;
	static constexpr std::string_view entry = "probe";
	static constexpr device::Dim grid = {.x = 82 * 7, .y = 1, .z = 1};
	static constexpr device::Dim block = {.x = 256, .y = 1, .z = 1};
	static constexpr std::uint32_t shared_bytes = 0;
	static constexpr device::Residency residency = device::Residency::CoResident;
	using Params = ProbeParams;
};

static_assert(device::Kernel<Probe> && device::Kernel<Wide>);

template<class Params>
[[nodiscard]] auto bytes_of(Params const& params) -> std::vector<std::byte> {
	auto const raw = std::bit_cast<std::array<std::byte, sizeof(Params)>>(params);
	return {raw.begin(), raw.end()};
}

template<class K>
[[nodiscard]] auto recorded_as(fake::Launch const& launch, typename K::Params const& params) -> bool {
	return launch.ordinal == 1 && launch.entry == K::entry && launch.grid.count() == K::grid.count() &&
	       launch.block.count() == K::block.count() && launch.shared_bytes == K::shared_bytes && launch.residency == K::residency &&
	       launch.parameters == bytes_of(params);
}

/** Every launch reaches the driver with the kernel's declared geometry and exactly the parameter bytes given. */
[[nodiscard]] auto launches_carry_the_declaration_and_the_exact_parameters(std::uint64_t seed) -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const& context = opened->context(Rank::One);
	auto const program = context.load<Probe, Wide>();
	auto const stream = context.create_stream();
	if (!program || !stream) {
		return false;
	}
	auto random = engine::testing::generator(seed);
	auto word = std::uniform_int_distribution<std::uint64_t>{};
	auto holds = true;
	for (auto round = std::size_t{0}; round < 200; ++round) {
		if (round % 3 == 0) {
			auto const params = WideParams{.first = word(random), .second = word(random)};
			holds =
			    holds && context.launch<Wide>(*program, *stream, params).has_value() && recorded_as<Wide>(driver.launches().back(), params);
		} else {
			auto const params = ProbeParams{
			    .ring = word(random),
			    .out = word(random),
			    .count = static_cast<std::uint32_t>(word(random)),
			    .flags = static_cast<std::uint32_t>(word(random)),
			};
			holds = holds && context.launch<Probe>(*program, *stream, params).has_value() &&
			        recorded_as<Probe>(driver.launches().back(), params);
		}
		holds = holds && driver.launches().size() == round + 1;
	}
	return holds && driver.violations().empty();
}

[[nodiscard]] auto a_rejected_image_carries_the_jit_log() -> bool {
	auto const log = std::string{"ptxas fatal   : Unresolved extern function 'missing'"};
	auto const driver = Fake{Machine{.jit = fake::JitRejects{.status = device::Status::InvalidPtx, .log = log}}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const program = opened->context(Rank::Zero).load<Probe>();
	return !program &&
	       program.error() == device::DeviceError{.reason = device::ModuleRejected{.status = device::Status::InvalidPtx, .log = log}} &&
	       driver.live(Kind::Module) == 0;
}

[[nodiscard]] auto a_long_jit_log_is_kept_as_a_prefix() -> bool {
	auto const log = std::string(100'000, 'e');
	auto const driver = Fake{Machine{.jit = fake::JitRejects{.status = device::Status::InvalidPtx, .log = log}}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const program = opened->context(Rank::Zero).load<Probe>();
	if (program || !std::holds_alternative<device::ModuleRejected>(program.error().reason)) {
		return false;
	}
	auto const& kept = std::get<device::ModuleRejected>(program.error().reason).log;
	return !kept.empty() && log.starts_with(kept);
}

[[nodiscard]] auto a_missing_kernel_is_named_and_its_module_unloaded() -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const program = opened->context(Rank::Zero).load<Probe, Absent>();
	return !program && program.error() == device::DeviceError{.reason = device::KernelMissing{.entry = "absent"}} &&
	       driver.live(Kind::Module) == 0 && driver.violations().empty();
}

[[nodiscard]] auto a_signature_drift_is_refused_at_load() -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const narrow = opened->context(Rank::Zero).load<Narrow>();
	auto const pair = opened->context(Rank::Zero).load<Pair>();
	return !narrow &&
	       narrow.error() == device::DeviceError{.reason =
	                                                 device::KernelSignature{
	                                                     .entry = "probe",
	                                                     .expected = sizeof(WideParams),
	                                                     .first = 24,
	                                                     .second = std::nullopt,
	                                                 }} &&
	       !pair &&
	       pair.error() == device::DeviceError{.reason =
	                                               device::KernelSignature{
	                                                   .entry = "pair",
	                                                   .expected = 8,
	                                                   .first = 8,
	                                                   .second = 8,
	                                               }} &&
	       driver.live(Kind::Module) == 0;
}

[[nodiscard]] auto a_grid_that_cannot_be_co_resident_is_refused() -> bool {
	auto const driver = Fake{Machine{}};
	auto const opened = Devices::open(driver.driver());
	if (!opened) {
		return false;
	}
	auto const& context = opened->context(Rank::Zero);
	auto const program = context.load<Crowded>();
	auto const stream = context.create_stream();
	if (!program || !stream) {
		return false;
	}
	auto const launched = context.launch<Crowded>(*program, *stream, ProbeParams{});
	return !launched &&
	       launched.error() == device::DeviceError{.reason =
	                                                   device::Failure{
	                                                       .symbol = "cuLaunchCooperativeKernel",
	                                                       .status = device::Status::CooperativeLaunchTooLarge,
	                                                   }} &&
	       driver.launches().empty();
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "launches carry the kernel's geometry, residency and exactly the parameter bytes (seed 21)",
	        .holds = launches_carry_the_declaration_and_the_exact_parameters(21),
	    },
	    engine::testing::Case{
	        .name = "an image the jit rejects fails with the jit's status and log, leaving no module",
	        .holds = a_rejected_image_carries_the_jit_log(),
	    },
	    engine::testing::Case{
	        .name = "a jit log longer than the buffer is kept as a prefix of itself",
	        .holds = a_long_jit_log_is_kept_as_a_prefix(),
	    },
	    engine::testing::Case{
	        .name = "a kernel the image lacks is named, and the module is unloaded",
	        .holds = a_missing_kernel_is_named_and_its_module_unloaded(),
	    },
	    engine::testing::Case{
	        .name = "a kernel whose parameters are not its declared struct is refused at load",
	        .holds = a_signature_drift_is_refused_at_load(),
	    },
	    engine::testing::Case{
	        .name = "a co-resident grid larger than the cards hold fails the launch with the driver's status",
	        .holds = a_grid_that_cannot_be_co_resident_is_refused(),
	    },
	};
	return engine::testing::report(cases);
}
