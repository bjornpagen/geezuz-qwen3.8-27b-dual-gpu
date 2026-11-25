<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# AGENTS.md

This file is the law for every change to this repository, by humans and coding agents. PLAN.md part 1 is the spec;
when the two conflict, stop and report the conflict; never pick silently. PINS.md records the toolchain in use and
every deliberate deviation from this file.

The engine is one program for one model on one machine: Qwen3.8-27B-FP8 as w8a16 on two NVLinked RTX 3090s (sm_86),
tensor parallel 2, `S = 8` slots. It is written in a deliberately small C++26 language: values and algebraic data
types, structural concepts, templates, `constexpr`/`consteval`, RAII capabilities, `std::expected`, sender/receiver
execution, named modules, and explicit quarantine boundaries. Where C++ offers two mechanisms for one concept, this
file chooses one; the other is forbidden.

---

## 1. Principles

These decide every case this file does not spell out.

- **Representation first.** Before adding a branch, flag, guard, `optional`, sentinel, "unknown" case or `if`, ask
  what representation would make the state impossible, and change the data, types and invariants until the case is
  not special or not expressible. Add the branch only when the problem has the case (essential), not the
  representation (accidental), and say why in a comment above it. Parse at the boundary, once: a check whose result
  is thrown away is a validator; return a type that carries what was learned and delete the downstream re-checks.
  A string that names a thing is a missing type. A switch on a tag repeated in two places is a polymorphism not yet
  named. Do not force two genuinely different cases into one representation; that hides the branch in a flag.
- **A choice is a type or an enum case, never a `bool`.** A parameter, field or configuration that selects
  behaviour or state is an `enum class`, a `std::variant` or a type. A predicate (a question with a yes/no answer)
  may return `bool`.
- **No static fact paid for at runtime.** Every shape, offset, tile schedule, head split, buffer size and pool
  geometry is a compile-time constant checked with `static_assert`. The only runtime facts are the tokens, the
  lengths, which slots are live, and the machine's identity. What the build, the model description or a type
  already knows is never recomputed, looked up, tested or parsed back at run time.
- **One fact, held once.** A fact held twice (two constants, a type and a comment, a test and the code under test)
  drifts. Derive the second from the first or delete it. Model dimensions live only in `device/model.h`, re-exported
  as `engine.model`; the ptx isa lives only in `cmake/engine.cmake`.
- **Hardcoded.** One model, one weight format, two 3090s, tensor parallel 2, `S = 8`, sm_86. No runtime flags,
  environment variables, options, compatibility modes or fallbacks choosing between implementations. The build
  presets and the baked weight image are the only configuration.
- **Hard cutover, hard culling.** A change replaces the old form outright: no shim, alias, dual path, compatibility
  code or "remove later". It deletes the dead code, its tests, docs and comments in the same change.
- **Delete before you optimize.** Every part names the rule that requires it and the fact only it supplies; a part
  whose fact something else holds is deleted.
- **A blowup is a bug,** fixed at its cause, never a cap or governor that keeps a runaway alive.
- **No stubs, no fake completeness.** No placeholder kernel, no `return {}` standing in for logic, no "not
  implemented" path. A fake that exists only for tests is a legitimate test double and is named as one. What
  cannot be finished properly is reported, never papered over.
- **Comments explain why,** never what, and cite no documents and no rule numbers (§26.3 lists the forms).
- **Report honestly.** Type-checked is not built, built is not run, run on macOS is not run on the GPU. Say what was
  measured, on what, how many runs.
- **Done means built and checked here:** the change builds with zero warnings in its presets and its tests pass.

A handoff names, per obligation: the state made unrepresentable, the check moved to a boundary, the branch deleted,
or why the case is essential.

---

## 2. One concept, one primitive

| Concept | Blessed mechanism | Forbidden alternatives |
|---|---|---|
| Generic polymorphism | concepts + templates | inheritance, virtual interfaces, CRTP, tag dispatch |
| Closed polymorphism | `std::variant` + visitation | class hierarchies, manual tagged unions |
| Compile-time elaboration | `consteval`, `constexpr`, `static_assert` | preprocessor metaprogramming, generated boilerplate |
| Synchronous failure | `engine::core::Result<T, E>` (`std::expected` with a classified error) | exceptions, error-code + out-param APIs |
| Impossible programmer state | `engine::core::invariant` | `assert`, ad-hoc `abort`, error plumbing for bugs |
| Optionality | `std::optional<T>` | null sentinels, magic values |
| Async effects | sender/receiver through `engine.exec` | `std::async`, futures/promises, coroutines, detached callbacks |
| Parallel composition | sender combinators such as `when_all` | manually coordinated threads |
| Kernel readiness | typed event values + one owner-driven state machine | callbacks, opaque self pointers, addresses in kernel user data |
| Long-lived reference into owner storage | generational handle (`engine.handle`), a distinct handle struct per table | stored pointers or references, generation-free indexes, phantom `Strong<Tag, T>` wrappers |
| Resource lifetime | RAII | manual cleanup paths |
| Dynamic ownership | `std::unique_ptr<T>` when a value cannot suffice | raw owning pointers, `shared_ptr`, `weak_ptr` |
| Required borrow | `T&` / `T const&` | raw pointers |
| Optional borrow | `std::optional<T&>` | nullable raw pointers, `std::optional<std::reference_wrapper<T>>` |
| Sequence borrow | `std::span<T>` | pointer + length |
| Text borrow | `std::string_view` | pointer + length, null-terminated borrowed APIs |
| Finite atoms | `enum class` | integer constants, strings-as-enums, `bool` choices |
| Product data | public `struct` | getter/setter object shells |
| Stateful capability/resource | small RAII `class` | "manager" object graphs |
| Project dependencies | named modules | project headers, header units, textual inclusion |
| Formatting/output | `std::format` / `std::print` | iostreams, printf-family |
| Configuration | build graph (presets, platform-selected sources) | preprocessor conditionals, runtime flags, env vars |
| Semantic compatibility | structural concepts and compile-time laws | nominal tag wrappers, marker base classes |
| Integer conversion | `engine::core::narrow` (program-guaranteed), `engine::core::try_narrow` (input-derived) | unexplained `static_cast`, implicit narrowing |

If the blessed mechanism cannot express a required property, stop and identify the missing primitive before
introducing a second way.

---

## 3. Toolchain

Clang is the only compiler: LLVM clang (latest stable, at least 23.1) with libc++, `import std`, dynamic linking,
lld. No GNU compiler or runtime, no Apple clang, no MSVC. The same clang compiles host C++ and device code. The
toolchain is one tuple: clang-tidy, clang-scan-deps and libc++ come from the compiler's own installation, found by
the configure gate, never from `PATH` separately.

The configure gate (top-level `CMakeLists.txt`) accepts exactly two machines and decides everything about the
toolchain; translation units never negotiate features:

- **darwin arm64** — development: builds everything, runs the cpu tests, emits ptx; no GPU, no CUDA toolkit;
- **linux x86_64** — production: the same, plus the `ptxas` gates and the GPU.

It also requires: Ninja (1.13 or later), CMake 4.4 (the `import std` gate value is series-specific; PINS.md),
clang 23.1 or later with libc++ and its `libc++.modules.json`, and `ENGINE_BUILD` set by a preset.

Project compiler properties (module-ABI flags, global so the synthesized std module matches every importer):

```text
-std=c++26 -stdlib=libc++ -fno-exceptions -fno-rtti
-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE    host library assertions, always on
```

Warnings are errors, everywhere, through the `engine_language_profile` interface target and the device rule (one
list, `ENGINE_WARNINGS`). Do not add compatibility branches for other standards, compilers or standard libraries.
The C++26 features the pinned libc++ lacks are listed in PINS.md with the blessed substitute; do not emulate them.

**Contracts.** Clang does not implement C++26 contracts (`pre`, `post`, `contract_assert`). Impossible programmer
state is `engine::core::invariant(condition)`: always on, in every build; it names the violated site on standard
error and aborts; in a constant expression a violation does not compile. When the pinned clang implements
contracts, `invariant` is replaced by them in one change (PINS.md `clang-contracts`).

---

## 4. Build system: CMake describes, Ninja executes

CMake generating Ninja is the only build system. No Makefiles, scripts that compile, hand-written Ninja, or second
build graph. The interface is the presets; Ninja is an implementation detail.

### 4.1 Presets

| preset | what it is |
|---|---|
| `dev` | `-O1 -g`, warnings as errors, trap-mode UBSan on host, device invariant checks on |
| `release` | `-O3`, thin LTO, device checks off |
| `asan-ubsan` | host AddressSanitizer + UndefinedBehaviorSanitizer for the cpu test suite, device checks on |
| `profile` | `release` plus the per-phase profiler (device profiler on): a build-graph choice, never a runtime flag |
| `lint` | the `dev` graph with clang-tidy on every translation unit, host and device, dialect and quarantine profiles |

Each preset sets one cache variable, `ENGINE_BUILD`; the top-level `CMakeLists.txt` maps it to flags in one place.
`CMAKE_BUILD_TYPE` is unused and refused. Build presets exist for all five, test presets for all but `lint`.

The coordinator builds in `build/<preset>`. **Every lane builds in its own binary directory**, which the command line
overrides (`-B` takes precedence over the preset's `binaryDir`):

```sh
cmake --preset dev -B build/lanes/<lane>/dev
cmake --build build/lanes/<lane>/dev --target <your targets>
ctest --test-dir build/lanes/<lane>/dev -R '^<name>'
```

Same for `asan-ubsan`, `lint` and `release` (`build/lanes/<lane>/<preset>`). `cmake --build --preset <p>` and
`ctest --preset <p>` always mean `build/<p>`.

### 4.2 Components

A component is one named module in one directory with one CMake target:

| | spelling |
|---|---|
| directory | `<zone>/<name>/` (zones: `src/`, `foreign/`, `unsafe/`, `apps/`, `bench/`; tests in `tests/<name>/`) |
| module | `engine.<name>` |
| namespace | `engine::<name>` |
| target | `engine_<name>` (component names are unique across zones) |
| tests | `tests/<name>/`, ctest names `<name>` and `<name>.<case>` |

Components are **discovered**: the top-level `CMakeLists.txt` adds every `<zone>/*/CMakeLists.txt`. A lane adds a
component by creating files only inside its own directories; no shared file changes. A new directory is picked up at
the next build (the glob is `CONFIGURE_DEPENDS`). A component declares its module dependencies in its own
`CMakeLists.txt`:

```cmake
# src/json/CMakeLists.txt
add_library(engine_json)
target_sources(engine_json PUBLIC FILE_SET CXX_MODULES FILES json.cc value.cc parse.cc)
target_link_libraries(engine_json PUBLIC engine_language_profile engine_core)
```

```cmake
# tests/json/CMakeLists.txt
engine_test(json SOURCES json.test.cc LINK engine_json)
```

Every component links `engine_language_profile`. Target names resolve at generate time, so the order of discovery
carries no meaning. Every file of the module is listed explicitly in its `FILE_SET CXX_MODULES`; never glob sources.
CMake owns module scanning and BMI ordering; never hand-write module maps.

### 4.3 The shared rules (`cmake/engine.cmake`)

CMake stays boring: no project `macro()`, no `execute_process()` as logic, no source generation. The shared rules
exist only because a plain target declaration cannot express them once:

- `engine_test(<name> SOURCES ... LINK ...)`: one test executable, ctest name `<name>`, with a deadline.
- `engine_death_test(<name> SOURCES ... LINK ...)`: passes only if the program aborts (an invariant violation).
- `engine_compile_fail(<name> SOURCE <file> LINK ... MATCH <regex>)`: passes only if building the file fails with
  the pinned diagnostic; ctest name `compile_fail.<name>`.
- `engine_device_ptx(<name> SOURCE <tu.cu>)`: one device translation unit to `<build>/ptx/<name>.ptx` (§17.1).
- `engine_embed_ptx(<target> SOURCE <unit.cc> IMAGES <name>...)`: lets a foreign module unit `#embed` images.

### 4.4 Platform selection

A platform difference is selected in the build graph, one source file per platform, never by a preprocessor test or a
runtime check. The configure gate sets `ENGINE_PLATFORM` to `darwin` or `linux`; the component that has a platform
seam chooses its file:

```cmake
# unsafe/net/CMakeLists.txt
if(ENGINE_PLATFORM STREQUAL "darwin")
	set(wait_backend wait.darwin.cc)   # kqueue
else()
	set(wait_backend wait.linux.cc)    # epoll
endif()
target_sources(engine_net PRIVATE ${wait_backend})
```

Both files implement the same narrow internal surface; dialect code and the loop above them contain no platform
test. Adding a platform means adding one such file per seam and widening the gate.

### 4.5 Dependencies

Prefer, in order: project-owned source; a module-native dependency pinned to an immutable revision; a pinned
dependency adapted behind `foreign/`; rewriting a dependency that would force exceptions, shared ownership, macro
APIs or headers into dialect code. Acquired dependencies are pinned to a commit (the tag is a comment), consumed as
targets, and never leak their preprocessor configuration upward.

---

## 5. Source zones

| zone | kind | holds |
|---|---|---|
| `src/` | dialect | the engine's components |
| `apps/` | dialect | `geezuz-server`, `geezuz`, `geezuz-bake`, `geezuz-ref` |
| `tests/` | dialect | unit, property, death and compile-fail tests; `tests/testing/` is the harness |
| `bench/` | dialect | benchmark clients, `bench/ubench/` device microbenchmark drivers |
| `device/` | quarantine | cuda device code and the model description header |
| `foreign/` | quarantine | the cuda driver api, the stdexec boundary, embedded ptx, the model-description re-export |
| `unsafe/` | quarantine | the reactor, mmap, pinned and hugepage memory, process failure |
| `data/` | data | draft vocabulary, tokenizer and chat-format fixtures |

### 5.1 Dialect code

Every C++ translation unit in a dialect zone obeys every rule in this file. No preprocessor, no headers, no unsafe
primitives, no lint suppression.

### 5.2 Quarantine

Only unavoidable machine and ABI adaptation lives in `device/`, `foreign/` and `unsafe/`: OS and libc headers,
vendor C APIs, compiler intrinsics, syscalls, inline ptx, atomics and locks that implement a higher-level primitive,
pointer arithmetic an ABI or memory primitive requires, `reinterpret_cast` a boundary requires. These zones may use
headers and the preprocessor where the external interface requires them. They export a narrow, typed, owning surface
upward; no dialect unit includes a quarantine header or depends on preprocessor state.

Quarantine relaxes representation rules, not ownership rules: every resource and allocation has exactly one RAII
owner; a raw pointer is a transient ABI argument, never an owner, never stored as asynchronous state, never placed in
kernel user data. Kernel and driver APIs return facts into caller-owned storage.

Do not move ordinary code into quarantine to escape a rule.

---

## 6. Modules and preprocessing

```cpp
// src/json/json.cc — the primary interface
export module engine.json;

export import :value;
export import :parse;
```

```cpp
// src/json/parse.cc — one concern per partition
export module engine.json:parse;

import std;
import engine.core;
import :value;
```

- A component is one importable module `engine.<name>`. A single-concern component is one file; a component with
  several concerns has partitions `engine.<name>:<concern>` in `<concern>.cc`, and its primary interface does
  nothing but `export import` them. Outsiders cannot import a partition, so the module name is the whole public
  surface.
- One partition per concern, roughly 50–300 lines. Do not split a partition into interface and implementation units.
- Module-internal helpers have module linkage (no `export`, no anonymous namespace): an exported inline or template
  entity may not expose a translation-unit-local one.
- The only C++ source extension is `.cc`. Interface-ness is declared by the build, never by spelling. Device
  translation units are `.cu` and device headers `.h`, in `device/` only; foreign headers a boundary needs are `.h`
  in their quarantine directory. `.cpp`, `.cppm`, `.hpp`, `.hh`, `.hxx`, `.inc`, `.inl`, `.ipp`, `.tpp` do not exist.
- Header units (`import <vector>;`) are forbidden. Textual inclusion is forbidden in dialect code.
- Every preprocessing directive is forbidden in dialect code: `#include`, `#define`, `#undef`, `#if` and friends,
  `#pragma`, `#error`, `#warning`, `#line`, `#embed`. Toolchain macros exist; project code never branches on them.

---

## 7. Structural typing

Express capabilities as minimal concepts; conformance is implicit:

```cpp
template<class E>
concept Classified = std::regular<E> && requires(E const& error) {
	{ error.retry() } -> std::same_as<Retry>;
};
```

Prefer several small concepts to a pseudo-interface. Do not create nominal distinctions only to reject equal
representations (`Strong<Tag, T>`), marker bases, marker structs, empty tags, registration traits, inheritance or
CRTP as interface declaration, or exact-type allowlists as semantic dispatch (`if constexpr (std::same_as<T, Foo> ||
...)`). Write the structural concept that explains why the types are accepted. `std::same_as` is permitted only when
exact representation is the requirement (ABI checks, expression-result requirements).

A generational handle is not a phantom tag: `SlotHandle` and `KvBlockHandle` are distinct structs because they name
different owner tables, and mixing them is a representation error the compiler rejects.

---

## 8. Data modeling

- Plain data is a public aggregate with designated initialization. No private fields behind trivial accessors.
- Closed alternatives are `std::variant`. Never inheritance hierarchies, C unions, discriminator + payload structs, or
  boolean matrices encoding states.
- `std::optional<T>` only for genuine absence; never zero, `-1`, empty string or null as absence.
- Finite vocabularies are `enum class` with an explicit small underlying type where layout matters.
- Every dimension is a compile-time constant from the model description; a type that holds a shape takes it as a
  template parameter, never a runtime field.

## 9. `struct` and `class`

`struct` is transparent product data. `class` is reserved for a real invariant, resource or capability whose
constructor, destructor or access control makes invalid operations impossible or owns a resource (file
descriptors, mapped images, device allocations, a slot table). Inheritance is forbidden, even for classes.

## 10. No runtime OO, no RTTI

Forbidden outside an unavoidable foreign ABI adapter: `virtual`, `override`, `dynamic_cast`, `typeid`, inheritance,
CRTP, abstract bases, visitor hierarchies. `-fno-rtti` removes RTTI from the language. Use concepts for open
compile-time polymorphism and `std::variant` for closed runtime alternatives.

---

## 11. Exceptions do not exist

`-fno-exceptions` removes them. `throw`, `try`, `catch`, `std::exception_ptr` and friends do not appear in project
code, except where a vendor interface forces the spelling (the impossible exception channel stdexec declares, in
`foreign/exec/`).

### Error algebra

```text
engine::core::Result<T, E>   recoverable synchronous failure (std::expected with a Classified error)
std::optional<T>             genuine absence
std::variant<...>            alternatives
sender error channel         asynchronous failure
sender stopped channel       cancellation
engine::core::invariant      impossible programmer state
static_assert                compile-time law
termination                  unrecoverable process failure
```

One mechanism per failure class. Never error code + output parameter. Never turn a recoverable error into an
invariant, or a programmer bug into error plumbing.

---

## 12. Ownership and borrowing

Prefer values. `std::unique_ptr<T>` only when dynamic stable ownership is required. Dialect code contains no raw
pointer declarations or raw pointer return types, and no `new`, `delete`, `malloc` family, `shared_ptr`,
`weak_ptr`, `enable_shared_from_this`.

Borrows are exactly `T&`, `T const&`, `std::optional<T&>`, `std::span<T>`, `std::string_view`.

- Synchronous function parameters may borrow.
- Public parsers and decoders return owning values; `parse(std::string{...})` never returns data referring to it.
- Owning structs contain no borrows. An ephemeral `FooView` lives only inside one synchronous call chain.
- A returned view refers only to immutable program-lifetime storage (a literal, an embedded image).
- Values crossing an async or sender boundary are owned; a returned sender never captures locals or parameters by
  reference; messages between concurrent components are recursively free of references, spans and views.
- When ownership and borrowing are both viable, copy at the lifetime boundary.

This makes raw-owner double free, ownership use-after-free, reference-count cycles, parser results dangling into
temporary input, and kernel readiness dereferencing a dead object structurally absent from dialect designs. It is
not general lifetime analysis: keep every pointer cast, arithmetic or stored borrow in a tiny reviewed quarantine
boundary with a `/* SAFETY: ... */` line, and run the sanitizer and lint presets.

## 13. Shared ownership is forbidden

If ownership is ambiguous, redesign the graph: value ownership, one unique owner, an arena with one reset boundary,
a serialized owner, or handles into an owning store. Reference counting is not an answer to an unclear lifetime. (A
refcounted kv block is a counted *resource* in an owning pool addressed by handles, not shared ownership of memory.)

## 14. Mutation

Mutation is permitted when it is local and unobservable: input value → local mutation → output value. Shared
observable mutation is an effect and is isolated behind one owner. Non-const globals, function-local statics and
`thread_local` state are forbidden. Pass dependencies, state, clocks, randomness and schedulers explicitly.

---

## 15. Concurrency and async

Kernel-level concurrency (the http reactor) is one owner-driven state machine. Application-level composition is
sender/receiver: `engine.exec` re-exports the vocabulary (`just`, `then`, `upon_error`, `upon_stopped`, `let_value`,
`let_error`, `let_stopped`, `when_all`, `starts_on`, `continues_on`, `on`, `schedule`, `into_variant`,
`stopped_as_optional`, `just_error`, `just_stopped`, `run_loop`, the sender/receiver/scheduler concepts and
customization points) under the names it keeps when libc++ ships `std::execution`, plus `engine::exec::wait<E>`.

- `foreign/exec/exec.cc` is the only file that spells `stdexec::`. Everything else uses `engine::exec::`.
- `sync_wait` is not exported: without exceptions it cannot return a typed error. `wait<E>(sender)` returns
  `optional<expected<tuple<values...>, E>>`: nullopt is stopped, unexpected carries the typed error.
- The decode cycle is one persistent kernel, not a graph of senders: senders submit cycles and collect results.
- Forbidden in dialect code: `co_await`, `co_yield`, `co_return`, `std::thread`, `std::jthread`, `std::async`,
  `std::future`, `std::promise`, `std::packaged_task`, detached work, callback concurrency.
- `std::mutex`, `std::condition_variable`, `std::atomic` and explicit memory orders live only in `unsafe/`
  primitives. No lock survives a suspension.

### Reactor law

The reactor follows the skalibs `iopause` shape: prepare caller-owned interest records, wait, receive readiness
facts, advance an explicit state machine.

- One thread owns a reactor and every resource registered in it.
- Kernel user data holds an integer token, never an address: a generational handle. Stale generations are ignored;
  a looked-up slot reference is never stored; generation exhaustion retires the slot.
- Slots own their descriptors, buffers, deadlines and closed state variant; slot storage is bounded and allocated at
  setup, never resized while tokens exist; cancellation invalidates the generation before reuse.
- Readiness callbacks, `void* self`, intrusive waiter nodes and operation-state pointers are forbidden, in `unsafe/`
  too. A returned event batch is inspected for stop events before ordinary work.
- Handlers run inline on the owner loop and are bounded and nonblocking; blocking work is a sender on another
  scheduler, never a stall of the loop. The wait backend is selected per platform in the build graph (§4.4): kqueue
  on darwin, epoll on linux.

### Time

Waiting is spelled with absolute `std::chrono::steady_clock::time_point` deadlines, never relative timeouts; only
`*_until` waits exist. A configured timeout policy is converted to a deadline once, when the operation begins.

---

## 16. Functional core

Prefer `A -> B`, `A -> Result<B, E>`, `A -> sender<B>`. Prefer composition, ranges and algorithms over
index-and-flag loops where they read clearer. Recursion is allowed. No hidden I/O, clocks, randomness, global caches
or implicit schedulers in otherwise pure functions.

## 17. Device code

### 17.1 The ptx rule

Device code compiles with clang in device-only mode straight to ptx; there is no CUDA SDK at build time:

```text
clang++ -x cuda --cuda-gpu-arch=sm_86 --cuda-device-only -nocudainc -nocudalib -std=c++26 -fno-exceptions
        -fno-rtti -O3 -Xclang -target-feature -Xclang +ptx93 -S
```

`engine_device_ptx(<name> SOURCE <tu.cu>)` produces `<build>/ptx/<name>.ptx`, one loadable image per translation
unit, tracked through its headers. On linux the same ptx then passes `ptxas -arch=sm_86 -v --warn-on-spills --Werror`
in the build graph; on darwin that gate does not exist. In the `lint` build the translation unit also passes
clang-tidy with the device flags.

The ptx isa is one constant, `ENGINE_PTX_ISA` in `cmake/engine.cmake`: 9.3, the newest LLVM emits for
`.target sm_86`. It enters device code once, in `device/target.h`, as `engine::device::ptx_isa` (major * 10 +
minor); the build personality enters the same way as `engine::device::checks` and `engine::device::profiler`.
Every other device header reads those constants, never a define.

At ptx 9.3, clang's front end rejects every ptx-gated nvvm builtin (shuffles, votes, `redux`, `match`, bf16
conversions, `mma`, `ldmatrix`, `cp.async`, ...) because its builtin table ends at ptx 9.2 (PINS.md). Therefore
every ptx-gated operation is inline ptx inside a named primitive in `device/prims/`; only ungated builtins (special
register reads) are called directly.

A device lane adds an image:

```cmake
# device/decode/CMakeLists.txt
engine_device_ptx(decode SOURCE decode.cu)
```

and embeds it from a foreign component it owns, which exports it upward as a typed image:

```cpp
// foreign/decode_image/decode_image.cc
export module engine.decode_image;

import std;
import engine.ptx;

namespace engine::decode_image {

constexpr char text[] = {
#embed "decode.ptx" suffix(, '\0')
};

export inline constexpr ptx::Image image = ptx::image(text);

static_assert(image.header == ptx::abi.header, "every image shares the engine's isa and target");

}
```

```cmake
# foreign/decode_image/CMakeLists.txt
add_library(engine_decode_image)
target_sources(engine_decode_image PUBLIC FILE_SET CXX_MODULES FILES decode_image.cc)
target_link_libraries(engine_decode_image PUBLIC engine_language_profile engine_ptx)
engine_embed_ptx(engine_decode_image SOURCE decode_image.cc IMAGES decode)
```

The quoted `#embed "name.ptx"` form is required (PINS.md `clang-format-embed`). `ptx::image` is `consteval`: a
missing terminator or an unreadable header does not compile. The image text ends in its one NUL because
`cuModuleLoadData` takes a NUL-terminated string. `foreign/` declares the driver-api subset it uses and `dlopen`s
`libcuda.so.1`; where that fails (macOS) engine start returns `DeviceError::DriverUnavailable`.

### 17.2 Device code rules

- Device code is quarantine: it cannot import modules, so it uses headers (`.h`) under `device/`, included relative
  to `device/`. Allowed: templates, concepts, `constexpr`, `consteval`. Forbidden: exceptions, RTTI, virtual,
  dynamic allocation, `printf`, CUDA SDK headers. bf16 and e4m3 are our own types.
- **No runtime shape arguments.** Every dimension is a template parameter or a `constexpr` from `device/model.h`;
  kernel parameters carry only runtime facts.
- Inline ptx only inside small named primitives in `device/prims/`, each with a why-comment, each stating its ptx
  isa and sm level and checking them at compile time (`static_assert(ptx_isa >= 70)`, `static_assert(sm >= 80)`).
- Every kernel's resource budget is a static fact: threads, registers, shared memory (at most 99 KB per block) and
  blocks per SM are `static_assert`ed; on linux the build fails if `ptxas -v` reports spills.
- Device invariant traps exist only where `engine::device::checks == Checks::On` (`dev`, `asan-ubsan`, `lint`).
- One narrow typed launch function per kernel, exported upward as a module from `foreign/`.

### 17.3 Device style: the newest idiom sm_86 has

sm_86 is the only binary target. Device code is written from the start in the newest ptx isa LLVM emits for it,
with every async, synchronization and cache-control feature Ampere has, and Hopper-era kernel architecture emulated
where the silicon lacks it. No legacy idioms. Each idiom below is a named primitive in `device/prims/` with a
microbenchmark in `bench/ubench/`.

Required:

- **warp specialization**: producer warps own the weight stream, consumer warps own the math; roles are fixed per
  kernel by the compile-time schedule;
- **async copy**: `cp.async.cg` 16-byte transfers, completion signalled with `cp.async.mbarrier.arrive.noinc` into
  mbarriers; `cp.async.commit_group` / `wait_group` only where an mbarrier does not fit;
- **mbarriers inside a block**: `mbarrier.init` / `arrive` and phase-parity waits in multi-stage rings. sm_86 has no
  `try_wait` and no `expect_tx`, so a wait is a `test_wait.parity` spin with `nanosleep` backoff and `cp.async`
  completion is counted in arrivals. No `__syncthreads` in steady state; named barriers (`bar.sync id, n`) only for
  fixed sub-groups;
- **l2 control**: `createpolicy.fractional` with evict_first on the weight stream and evict_last on activations, kv
  tiles and flags; `prefetch.global.L2` for run-ahead; `discard.global.L2` on dead scratch;
- **fragments**: `ldmatrix` (and `.trans`) from xor-swizzled shared memory with zero bank conflicts;
  `mma.sync.m16n8k16` bf16 × bf16 → fp32; register double-buffering across k-steps;
- **reductions**: `redux.sync` for integers, butterfly `shfl.sync` for floats in a fixed order (determinism);
- **cross-card**: peer stores over NVLink; `st.release.sys` / `ld.acquire.sys` flags with `nanosleep` backoff, or
  `red.release.sys` counters; one fence per item, never per element.

Emulated: tma (producers issue the baked image's precomputed 16-byte transaction stream: a pointer bump plus
`cp.async`); clusters (cross-SM handoff through l2-resident flags and buffers with evict_last); fp8 cvt (an exact
e4m3 → bf16 bit sequence straight into mma a-fragments, exhaustively tested against all 256 codes); wgmma (consumer
warpgroups of 4 warps sharing a stage, the next stage's `ldmatrix` overlapped).

Banned: dp4a and any int8 or fp16-accumulate math path; cooperative-groups `grid.sync` and atomic-counter grid
barriers in the dataflow core (allowed only in the step-1 baseline, deleted at step 2); per-phase kernel launches;
weight-stream loads without a cache policy; unswizzled shared tiles with bank conflicts; texture and surface paths;
`printf`; dynamic shared allocation decided at run time.

### 17.4 The lossless contract

e4m3 weights convert to bf16 exactly; the block scale is applied in fp32; every matmul accumulates in fp32; the
residual stream and deltanet state are fp32; activations and the kv cache are bf16; verification and sampling are
exact. The only allowed difference from the reference is floating-point summation order. Never proposed: sub-8-bit
weights, int8/fp8 activations, fp8/int8/4-bit kv, bf16 deltanet state, fp16-accumulate matmuls, compressed
cross-card partial sums.

---

## 18. Concepts, templates and metaprogramming

Concepts are the public language of generic requirements: named when reusable, a local `requires` for a one-off.
Forbidden as dispatch: `enable_if`, SFINAE tricks, `void_t` detection, tag dispatch, marker traits, trait
registration, CRTP, inheritance. Type traits remain for representation facts. Specialization is not a substitute
for a concept.

## 19. No exact-type semantic dispatch

`if constexpr (std::same_as<T, Foo>) ... else if constexpr (std::same_as<T, Bar>)` is forbidden for behaviour.
Define the structural property or use a closed variant.

## 20. No manual dynamic type containers

Forbidden in dialect code: `std::any`, `std::type_info`, `std::type_index`, `void*`, manual type tags, runtime
registration by type identity.

## 21. Callables

Compile-time callable polymorphism is templates and concepts. `std::function` and `std::bind` are forbidden; use
lambdas. libc++ lacks `std::function_ref` (PINS.md); a callable parameter is a constrained template parameter. Do
not store callbacks to simulate an object graph.

## 22. Casting

Forbidden in dialect code: C-style casts, `reinterpret_cast`, `const_cast`, `dynamic_cast`, downcasts. Integer
conversions go through `engine::core::narrow` or `try_narrow`; `static_cast` is for explicit, checked language
conversions with no clearer API (enum to underlying, a value already proven in range). `std::bit_cast` for
representation reinterpretation. Pointer/integer conversion belongs in quarantine.

## 23. Numeric conversions

Implicit narrowing and silent signedness changes are compile errors (`-Wconversion -Wsign-conversion`). Do not
silence them with an unexplained cast: `narrow` states "the program guarantees it fits" (an invariant),
`try_narrow` states "it came from outside" (a `Result`). Counts, offsets, lengths and physical quantities keep their
representation explicit; do not invent nominal wrappers to tell equal representations apart.

## 24. C remnants

Forbidden in dialect code: C arrays, varargs, C unions, `setjmp`/`longjmp`, printf-family formatting, malloc-family
allocation, erased-callback C algorithms, pointer + length APIs, null-terminated borrowed APIs. Use `std::array`,
`std::span`, `std::variant`, templates, `std::format`/`std::print`, RAII containers, ranges, `std::string_view`.

## 25. I/O and formatting

`std::print`/`std::println` for output, `std::format` for strings; never iostreams or printf. Standard error is
reached through quarantine (`engine.fail`), because `stderr` is a macro. Binary I/O uses typed byte/span APIs and
explicit codecs. Protocol framing has one writer: the http response writer derives `Content-Length`, fixes
connection semantics and rejects caller-supplied framing headers; handlers return owning response values.

---

## 26. API shape

Public APIs are explicit and narrow. **Default arguments are forbidden**, with one exception: `engine::fail::invariant`
takes `std::source_location::current()` as a defaulted last parameter, the only way to capture the caller's site; it
lives in quarantine and dialect code reaches it as `engine::core::invariant`. Optional configuration is an explicit
aggregate. Avoid long positional parameter lists. Return values, never output parameters; name the fields of a
returned product.

### 26.1 Return values are contracts

Every non-void function in dialect code is `[[nodiscard]]`; error types are `[[nodiscard]]` structs. If a result is
safely ignorable the function returns `void`. Exempt by language convention: assignment and increment operators,
constructors, `main`, immediately-invoked lambdas. Discarding has two spellings: `std::ignore = advisory();` (discard
now) and `auto _ = scoped_capability();` (hold to scope end). `(void)expr` does not exist. A `Result` is never
discarded by either spelling; check it first. `[[nodiscard]]` + `-Werror` enforce this in every preset
(`tests/compile_fail/discarded_expected.cc` pins it).

### 26.2 Errors are decision surfaces

An error type exists so the caller can branch, and the universal decision of a long-running server is retryability.
Every error type answers it with `[[nodiscard]] constexpr auto retry() const -> engine::core::Retry`
(`Retry::Permanent` or `Retry::Transient`). `Result<T, E>` requires `Classified<E>`, so an unclassified error cannot
be the failure of a result (`tests/compile_fail/unclassified_error.cc`). Transient: could the same operation, retried
later with nothing changed by the caller, plausibly succeed (contention, peer disconnect, resource pressure)?
Everything else, and anything unsure, is permanent. Classification is a `switch` over the error kind with no
`default`, so a new kind fails the build until classified; a test pins the policy. Messages are rendering, never
input: two failures a caller must tell apart are two kinds.

Failure is transparent: on the error path every caller-visible output is exactly as the caller left it. Prefer to
make partial output unrepresentable (return one complete owning value or an error).

### 26.3 Comments

The code is the document: a capability is a concept, an invariant is `invariant` or `static_assert`, a valid-values
story is a type, behaviour is a test, a name that needs explaining is renamed. Prose states only what code cannot:
units and coordinate conventions, error meanings and the decisions they drive, lifetime rules the types cannot see,
ABI and protocol facts, why a case is essential, why a design is the way it is.

Comments explain why, never what, and cite no documents or rule numbers. The forms that exist in C++ source:

1. the license line, first in every file: `// SPDX-License-Identifier: AGPL-3.0-only`;
2. a documentation block `/** ... */` immediately above a declaration, saying what the signature cannot, then
   stopping;
3. `/* PIN(name): one line */` at a pinned-workaround site, backed by a PINS.md entry;
4. `/* SAFETY: one line */` in quarantine only, at a site whose soundness the types cannot see;
5. a `//` why-comment above code whose reason the code cannot show (an essential branch, a device-side fact, a
   quarantine header's purpose).

No narration, banners, `} // namespace` closers, TODOs or commented-out code. CMake files carry the license line as
`# SPDX-License-Identifier: AGPL-3.0-only` and comment only why.

---

## 27. Boolean parameters and state flags

A boolean that selects qualitatively different behaviour is an enum, variant or options aggregate. State machines
are closed variants, never several independent booleans.

## 28. Resource classes

A resource `class` is small, non-copyable when ownership is unique, movable when movement is valid (moves
`noexcept`), destructible without failure, explicit about ownership, free of inheritance. Raw native handles do not
escape quarantine.

## 29. Allocation

Application code never uses `new` or `delete`. Prefer, in order: values and automatic storage; fixed-size containers
(`std::array`, fixed-capacity structures); standard containers when capacity is genuinely dynamic; `std::unique_ptr`
for an independent dynamic lifetime; arenas for a demonstrated bulk lifetime. libc++ lacks `std::inplace_vector` and
`std::indirect` (PINS.md): a bounded sequence is a `std::vector` reserved to its named maximum and checked against it
before every insertion. Every heap allocation enters one RAII owner at the allocation expression, in quarantine too.

### Steady state

Setup may allocate freely and may fail. The serving loop reuses capacity acquired at setup; short-lived owning values
may allocate within named bounds. **Unbounded growth in the serving path is a bug.** Every wire-facing quantity has a
named maximum (`max_request_bytes`, `max_header_count`, ...), and buffers and tables are sized from it. Device memory
is planned entirely at build time from the model description: the pool geometry, the rings and the workspace are
constants.

## 30. Performance discipline

Abstractions are expected to vanish in optimized code when that is their design. For critical paths: benchmark,
inspect the optimized assembly or ptx, count allocations, test layout assumptions. Do not replace a clear value
abstraction with manual low-level code without evidence; isolate low-level code behind a safe module boundary.
Speculative decode varies run to run: compare speed at temperature 0, within one boot, per accepted token, best of n.

## 31. Testing

Tests obey the dialect and use the module-native harness `engine.testing` (`tests/testing/`): a test is a list of
`Case{.name, .holds}` and `report` returns the exit status. No macro framework.

- **Tests check behaviour**: outputs, invariants, round trips, exactness against an oracle. Never incidental text,
  ordering the contract leaves open, numbering, or a log line.
- **An oracle first.** A specification turned into a checker, a second implementation we own (a plain model of the
  same contract, the cpu reference), or a universal oracle (the sanitizers, the compile-time checks). A test without
  an oracle that can fail is deleted.
- **Search, not examples.** State the property for all inputs and generate them: exhaustively for small domains
  (every int16, all 256 e4m3 codes), randomly with fixed seeds named in the case (`engine::testing::generator`)
  otherwise. Encode the harness's invariants in the generator, never the system's intended behaviour, so the buggy
  sequence stays generable.
- A brittle test (checks more than the contract) or a dead one (cannot fail) is deleted, not restated.
- `engine_death_test` pins that an invariant violation aborts; `engine_compile_fail` pins that a misuse does not
  compile, with the diagnostic it must fail with.
- Every test has a deadline: a hang is a failure. A skipped test is never reported as passed.

Required runs before handing off: the lane's targets in `dev` with zero warnings, its tests in `dev` and
`asan-ubsan`, and `lint` over its targets.

## 32. Suppressions and escape hatches

Forbidden in dialect code: `NOLINT`, `NOLINTNEXTLINE`, `NOLINTBEGIN`, `NOLINTEND`, diagnostic pragmas,
warning-disable pragmas. If a rule cannot apply: the design changes, the code belongs in quarantine, or the rule
itself is changed centrally (and a vendor interaction is recorded as a PIN). A diagnostic disabled for a whole
target or profile is a PIN with its retire condition.

## 33. Enforcement

The rules are enforced by a ladder, strongest first:

1. **Compiler flags** make violations unrepresentable: `-fno-exceptions`, `-fno-rtti`, `-Werror` with the
   conversion, shadowing and `[[nodiscard]]` diagnostics.
2. **Types and compile-time checks**: `Classified` on `Result`, distinct handle structs, `consteval` image
   validation, `static_assert` on every budget and geometry.
3. **The build graph**: explicit module lists, header units fail scanning, the configure gate refuses wrong
   machines and toolchains, one platform source per seam.
4. **clang-tidy** over the whole graph (`lint`): the dialect profile (`.clang-tidy`) bans what an AST check can see
   (C arrays, casts, varargs, owning memory, default arguments, inheritance, non-const globals, missing trailing
   return types, unchecked optionals and results) and finds defects; the quarantine profile
   (`cmake/quarantine.clang-tidy`, linked as `device/.clang-tidy`, `foreign/.clang-tidy`, `unsafe/.clang-tidy`)
   looks for boundary defects without flagging the pointers and casts those zones exist for. A finding fails the
   build.
5. **Review** for what only source text shows: preprocessor use, comments, extensions. There is no grep checker,
   by decision.

Every rule aspires upward: when a concept, contract, type, name, build rule or check can carry a review convention,
it must, and the prose is deleted the same day.

## 34. Dependency policy

Prefer dependencies with a module-native interface, no exceptions or shared ownership in the dialect-facing API, no
macro configuration and no raw pointer ownership. Legacy and C dependencies go behind `foreign/`. Never weaken the
dialect to make a dependency convenient; adapt or replace it. Today: NVIDIA stdexec (pinned in the top-level
`CMakeLists.txt`, imported as its own `stdexec` module behind `engine.exec`), and at run time `libcuda.so.1` through
`dlopen`.

## 35. Generated code

Code generation is allowed only when the language and the build cannot derive the information with `consteval`
and templates. Generated sources live under `generated/` (or the build directory), are never hand-edited, and
expose a normal module boundary. Tile schedules and buffer geometries are `consteval` functions over the model
description, not generated files.

---

## 36. Review checklist

Before accepting a change, ask in order:

1. What state did this make unrepresentable? Could a type, enum or variant replace the branch, flag or guard?
2. Is every choice a type or enum case, and no `bool` selects behaviour?
3. Is any static fact paid for at run time? Is every shape a compile-time constant from the model description?
4. Is every fact held once?
5. Can this be a value, a `struct`, a structural concept?
6. Can `consteval` or `static_assert` reject this earlier?
7. Can `Result` encode this failure, and does its error classify retryability exhaustively?
8. Can a sender encode this effect, and does it own everything it captures?
9. Is ownership unique, and is every borrow call-scoped or program-lifetime?
10. Is every low-level operation confined to quarantine, with a `SAFETY` line where soundness is invisible?
11. Does it introduce a second mechanism for something already solved, or keep an old form beside the new one?
12. Is the serving path's footprint bounded by named maxima?
13. Is every wait an absolute deadline?
14. Does kernel readiness return integer-keyed facts to one owner, with no callback or address?
15. Device: no runtime shapes, budgets asserted, inline ptx only in named primitives with isa and sm checked, cache
    policy on every weight-stream load, nothing from the banned list?
16. Do the tests check the contract against an oracle, search rather than enumerate, and fail when the code is
    wrong?
17. Is every remaining comment one of the allowed forms, saying why?
18. Was it built and run here, in the presets this file requires, and is the handoff honest about what was not?
