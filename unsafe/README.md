# unsafe/

Quarantine zone for machine interfaces the dialect cannot express: the reactor (one wait backend per platform,
selected in the build graph), mmap, pinned and hugepage memory, process failure. It may include OS headers and use
transient ABI pointers; it does not relax ownership: every descriptor and allocation has one RAII owner, no pointer
is stored asynchronously, no kernel user data holds an address, readiness returns as typed facts to caller-owned
state. AGENTS.md §5.2 and §15.

- `fail/` — `engine.fail`: the always-on `invariant` (reached from dialect code as `engine::core::invariant`).
- `file/` — `engine.file`: read-only whole-file mappings (the checkpoint, the baked image, test inputs).
- `process/` — `engine.process`: child processes the caller owns (spawn, output lines, SIGTERM, reaping); flushed stdout.
- `file_write/` — `engine.file_write`: positional writes into a file that appears under its name only when complete.
- `net/` — `engine.net`: the owner-driven reactor, kqueue on darwin and epoll on linux.
- `hugepage/` — `engine.hugepage`: hugepage-backed (linux) host memory for pinned regions.
- `sha256_block/` — `engine.sha256_block`: SHA-256 compression on the machine's SHA instructions.
- `thread_pool/` — `engine.thread_pool`: the fork-join pool the cpu reference uses (the zone's mutex and condition
  variable).
