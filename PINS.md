<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# PINS.md

The toolchain this engine is built with, and every deliberate deviation from AGENTS.md or from upstream conventions.
Each `PIN(name)` site in the tree points at its entry here; the reasoning lives here, once.

Ritual on every toolchain bump: read this file top to bottom, re-test every retire condition, delete what upstream
fixed, and update the facts below in the same commit as the bump.

## Toolchain in use

| part | in use | where the requirement lives |
|---|---|---|
| compiler | LLVM clang 23.1.5 (MacPorts `clang-23` on the development mac: `/opt/local/libexec/llvm-23/bin`, also `clang++-mp-23` in `/opt/local/bin`) | configure gate: clang ≥ 23.1, not Apple clang |
| standard library | libc++ 23.1.5, with `std.cppm` (`share/libc++/v1`) and `libc++.modules.json` (`lib/libc++/`), linked dynamically with an rpath to its directory | configure gate: `libc++.modules.json` beside the compiler, `import std` for C++26 |
| linker | lld (`ld64.lld` on darwin) via `CMAKE_LINKER_TYPE LLD` | top-level `CMakeLists.txt` |
| lint | clang-tidy 23.1.5 from the compiler's own `bin/` | top-level `CMakeLists.txt` (`lint`) |
| build | CMake 4.4.4 (MacPorts `cmake-devel`), Ninja 1.13.2 | configure gate: CMake 4.4.x, Ninja ≥ 1.13 |
| device target | `.target sm_86`, ptx isa 9.3 (`+ptx93`) | `cmake/engine.cmake` (`ENGINE_PTX_ISA`) |
| ptxas gate | the system CUDA toolkit's `ptxas` on linux; absent on darwin | `cmake/engine.cmake` |
| minimum driver to jit ptx 9.3 | the R610 branch (cuda 13.3 pairs with R610: documented; ptx 9.3 shipping with cuda 13.3: inferred; see docs/HARDWARE.md) | the linux box's driver install |
| senders | NVIDIA stdexec v0.11.0 (latest tag, 2026-11-11), commit `fc5937a544db908edbde37c9aa67b8d308e6a788` | top-level `CMakeLists.txt` (`FetchContent_Declare`) |

The presets search `/opt/local/libexec/llvm-23/bin` (`CMAKE_PROGRAM_PATH`) before `PATH` for `clang++`, so the
development mac finds MacPorts clang without a shell change; the production box finds its system clang on `PATH`.
Moving to a newer LLVM major is one edit there plus this table.

## The macOS device and load strategy

There is no NVIDIA GPU and no CUDA toolkit on the development mac, and none is needed to build:

- clang compiles every device translation unit in device-only mode straight to ptx (`-x cuda --cuda-gpu-arch=sm_86
  --cuda-device-only -nocudainc -nocudalib -S`), with no SDK headers; the same ptx comes out on both machines;
- the ptx is embedded in the host binary with `#embed` by a `foreign/` module unit and exported as a typed
  `engine::ptx::Image`, validated at compile time (`consteval`);
- at run time the engine `dlopen`s `libcuda.so.1` and jit-loads the image with `cuModuleLoadData`; on macOS the
  `dlopen` fails and engine start returns `DeviceError::DriverUnavailable`;
- the `ptxas -v --warn-on-spills --Werror` resource gate runs on the same ptx only where a CUDA toolkit exists (linux,
  where the configure gate requires it). On darwin it is absent by platform: device resource budgets are checked by
  `static_assert` there and by `ptxas` first on the linux box.

## clang-ptx92-builtin-gating

- symptom: at ptx isa 9.3 clang's front end rejects every ptx-gated nvvm builtin. In llvmorg-23.1.5 and 23.1.6,
  `clang/include/clang/Basic/BuiltinsNVPTX.td` builds each builtin's required-feature string from
  `PTX_InstantiateLoop<[92, 91, 90, 88, ...]>`, an OR-list whose newest entry is ptx92, while the NVPTX backend's
  `PTXVersions` already includes 93. So `__nvvm_shfl_sync_*`, vote, `redux`, `match`, the bf16 conversions
  (`__nvvm_f2bf16_rn`), `mma`, `ldmatrix` and the `cp.async` builtins are all unavailable under `+ptx93`; only
  ungated builtins (special-register reads) remain
- decision (owner): stay on ptx 9.3, the newest isa LLVM emits for sm_86. Every ptx-gated operation is inline ptx in
  a named primitive in `device/prims/` that states and `static_assert`s its isa and sm level, as CUTLASS does
- sites: `cmake/engine.cmake` (`ENGINE_PTX_ISA 9.3`); every primitive in `device/prims/`
- retire: when the pinned clang's builtin list includes ptx93 (LLVM main adds 93 and 94; fixed for LLVM 24),
  re-evaluate primitive by primitive whether the builtin produces the same instruction; keep inline ptx where it is
  the only way to state a cache policy or ordering
- upstream: fixed on LLVM main, unreleased at the time of writing

## cmake-import-std-uuid

- symptom: `import std` is experimental in CMake 4.4, enabled by `CMAKE_EXPERIMENTAL_CXX_IMPORT_STD` set to a value
  that changes per CMake feature series; a stale value silently disables the feature
- sites: top-level `CMakeLists.txt`: the CMake 4.4.x gate and
  `set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")` (read from CMake 4.4.4's own
  experimental-feature table, the `CxxImportStd` entry)
- retire: when CMake ships `import std` as a stable feature; on any CMake series bump, read the new series' table,
  update the value and the gate together
- upstream: none; CMake's deliberate experimental-feature mechanism. The configure step prints CMake's
  "experimental" developer warning; it is expected

## macports-libcxx-layout

- symptom: MacPorts installs libc++ in `lib/libc++/` and libunwind in `lib/libunwind/`, outside the darwin clang
  driver's search: `clang++ -print-file-name=libc++.modules.json` does not find the metadata, CMake's `import std`
  discovery fails, and `libc++.1.dylib` loads `@rpath/libunwind.1.dylib` that no default rpath reaches
- sites: top-level `CMakeLists.txt`: before `project()`, the metadata is looked up beside the compiler
  (`<bin>/../lib/libc++/libc++.modules.json`) when the driver cannot find it, and `CMAKE_CXX_STDLIB_MODULES_JSON` is
  set from it; every binary gets an rpath to libc++'s directory and, where it exists, to the sibling `libunwind/`
- retire: when the driver finds the metadata itself (it does on a standard linux LLVM layout) and libc++ carries a
  usable rpath to libunwind; re-check on every LLVM bump
- upstream: MacPorts packaging layout

## libcxx-std-module-name

- symptom: CMake 4.4 compiles libc++'s `std.cppm` with its module directory as a plain `-I` path, not a system one,
  so `export module std;` trips `-Wreserved-module-identifier` in the synthesized std target, which no project
  setting can reach (it does not exist until generate time)
- sites: top-level `CMakeLists.txt`: `-Wno-reserved-module-identifier` in the global `CMAKE_CXX_FLAGS`. The
  diagnostic only fires on a module that names itself `std` or another reserved name; every engine module is
  `engine.<name>`, so nothing else is hidden
- retire: when CMake passes the libc++ module directory as a system include, or exposes the synthesized target
- upstream: not filed

## clang-embed-extension

- symptom: clang 23 still classifies `#embed` in C++26 as a C23 extension and warns under `-Wpedantic`
  (`-Wc23-extensions`), though `#embed` is standard C++26
- sites: `cmake/engine.cmake` (`engine_embed_ptx`): `-Wno-c23-extensions` on exactly the foreign units that embed
  ptx, nowhere else
- retire: when clang treats `#embed` as standard in C++26 mode
- upstream: not filed

## clang-format-embed

- symptom: clang-format 23 formats `#embed <name.ptx>` as `#embed < name.ptx>`, which names a different file
- sites: every ptx embed uses the quoted form, `#embed "name.ptx"`, which clang resolves through `--embed-dir` after
  the including file's directory (`foreign/ptx/abi.cc`, AGENTS.md §17.1)
- retire: when clang-format leaves an angled `#embed` header-name intact
- upstream: not filed

## clang-contracts

- symptom: clang 23 does not implement C++26 contracts (`pre`, `post`, `contract_assert`; `-fcontracts` is not an
  option)
- sites: `unsafe/fail/fail.cc` (`engine::fail::invariant`, re-exported as `engine::core::invariant`), and its
  defaulted `std::source_location::current()` parameter, the one default argument in the engine; it lives in
  quarantine because the dialect profile bans default arguments
- workaround: one always-on invariant function that names the site on standard error and aborts; in a constant
  expression a violation does not compile
- retire: when the pinned clang implements P2900, replace `invariant` by contracts in one change and delete
  `engine.fail`'s invariant
- upstream: clang's C++ status page lists P2900 as not implemented

## libcxx-missing-cxx26

- symptom: libc++ 23.1.5 lacks `std::inplace_vector`, `std::indirect`, `std::function_ref`, and `std::mdspan`'s
  default construction of static extents; `std::optional<T&>` is present
- substitutes (AGENTS.md): a bounded sequence is a `std::vector` reserved to its named maximum and checked before
  each insertion; heap indirection is `std::unique_ptr`; a callable parameter is a constrained template parameter
- retire: adopt each as libc++ ships it, in one change per feature, deleting the substitute
- upstream: libc++'s C++26 status page

## stdexec-module

- fact: under clang 23 + libc++ stdexec imports cleanly inside the module graph. stdexec v0.11.0 ships its own
  `stdexec` module (`modules/stdexec.cppm`, built with `STDEXEC_BUILD_MODULES=1`); `foreign/exec/` builds it as
  `engine_stdexec_vendor` and `engine.exec` imports it and re-exports the vocabulary with `export using`. Senders
  compose across module boundaries (`tests/exec/` composes them in a dialect test). The cpp-starter narrow-abi
  quarantine (a GCC ICE workaround) is gone
- what did not work: including the stdexec headers in a module's global module fragment. Clang discards the GMF
  declarations the purview does not reach, including the hidden-friend operators stdexec's metaprogramming finds by
  ADL, so templates instantiated in an importer fail (`invalid operands to binary expression` in `__mset_insert`)
- not exported: stdexec's `exec::` extras (`static_thread_pool`) are not part of its module; the engine's
  schedulers are its own (one per card, on driver events)
- stdexec declares an `exception_ptr` error channel on some adaptors even under `-fno-exceptions`; `engine::exec::wait`
  receives it as an invariant violation, because nothing can complete on it
- retire: when libc++ ships `std::execution`, rebind `engine.exec`'s using-declarations to `std::execution`, delete
  the FetchContent pin and the vendor target, and keep every caller unchanged

## stdexec-analyzer-stack-escape

- symptom: clang's static analyzer reports `core.StackAddressEscape` inside stdexec's `connect`/`__tuple` for every
  translation unit that composes senders: stdexec operation states hold their receivers and children by design, which
  the analyzer reads as stack addresses escaping
- sites: `.clang-tidy` (the dialect profile): `-clang-analyzer-core.StackAddressEscape`. Dialect code has no raw
  pointers, and dangling references are diagnosed by the compiler's lifetime warnings; the check stays on in the
  quarantine profile
- retire: re-enable on every clang-tidy or stdexec bump and keep it if `lint` is clean
- upstream: not filed

## tidy-profile-choices

- `modernize-use-trailing-return-type.TransformLambdas: none`: functions are written with trailing return types;
  lambdas deduce theirs
- `performance-move-const-arg.CheckTriviallyCopyableMove: false`: moving a sender is the sender protocol, whatever
  its triviality
- `bugprone-unchecked-optional-access.IgnoreValueCalls: true`: under `-fno-exceptions` libc++'s `value()` on an
  empty optional aborts, so it is a checked access (used in `consteval` validation, where it does not compile)
- `readability-convert-member-functions-to-static` is off: `retry()` is a member by the `Classified` concept even
  when one kind is always permanent, and calling a static through an instance is itself flagged
- the quarantine profile turns off `portability-avoid-pragma-once`: clang is the only compiler, and device headers use
  `#pragma once`
- `lint` passes `--allow-no-checks`: the std module CMake synthesizes lies outside the tree, has no profile, and is
  not ours to lint; every source in the tree has a profile through its zone's `.clang-tidy`
