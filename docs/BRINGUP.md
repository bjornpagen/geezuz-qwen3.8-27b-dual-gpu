<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# BRINGUP: the linux box, from bare metal to phase c

the exact, ordered procedure that takes the box (two rtx 3090 fe under xg7 blocks, a 2-slot nvlink bridge, PCIE_6 and
PCIE_4 of a gigabyte mz72-hb0, dual epyc 7v13) from bare metal to the correctness runs of phase c, and the overclock
acceptance of phase d. every setting comes with how to verify it took and how it persists. nothing persists by itself:
power limits, clock locks and offsets are lost on every reboot or driver reload, so they are applied by systemd units
in `data/platform/`.

**status of this document.** nothing here has been run on the box yet. each step is labelled:

- **[checked here]**: the tool's behaviour was built and tested on the development mac, against the CUDA driver and
  NVML test double (`foreign/driver_double`), not on the cards;
- **[documented]**, **[measured by others]**: what docs/HARDWARE.md cites, with its sources;
- **[inferred]**: reasoned, not observed; the step that would prove it is named.

the microbenchmarks are run, not restated: `geezuz-ubench <name>` prints, for each benchmark, the review's expectation,
its evidence, the decision it settles, and an `operator:` line with what must be set by hand before it runs. that line
is the source of truth for the settings this document names by benchmark; read it before each run.

every geezuz program on the box runs under `numactl --cpunodebind=0 --membind=0` (step 6). the examples below write
`run` for that prefix:

```sh
alias run='numactl --cpunodebind=0 --membind=0'
```

---

## 1. hardware

1. both cards in **PCIE_6 and PCIE_4**: both hang off cpu0's root complex (board datasheets, both revisions)
   [documented]. no other gpu in the box (a display on one card took an nvlink link in a reported case [measured by
   others]).
2. the **2-slot bridge**, seated until it clicks; an unseated bridge or card leaves every link inactive [measured by
   others]. the owner confirmed it clears the water blocks.
3. a **fan on each card's backplate**, blowing across the 12 back-side gddr6x chips: the 110 °c junction throttle is
   the binding limit of the memory, and the xg7's passive backplate without airflow is the hot spot [measured by others;
   the margin it buys is inferred, measured in step 14].
4. psu headroom of 460-500 w per card for 1-10 ms transients at a 400 w limit [measured by others].

**verify**: steps 7 and 8 (links read NV4; both cards on numa node 0; pcie gen4 x16).

## 2. bios (mz72-hb0)

| setting | value | why |
|---|---|---|
| IOMMU (AMD-Vi) | enabled | the kernel puts it in passthrough (step 3); nvlink peer traffic never touches it [inferred] |
| Above 4G Decoding | enabled | 64-bit bars of 24 GB cards [inferred] |
| NUMA nodes per socket | NPS1 | node 0 is then all of cpu0, which owns both slots' root ports [inferred]; under NPS4 use the node step 6 reads |
| PCIe slot 6 and 4 link | Gen4 / auto | x16 to the host for the embedding rows and the ring [documented] |

no community p2p patch and no resizable-bar requirement: the engine uses nvlink peer access, which the stock driver
grants over the bridge, never pcie bar1 peer access [measured by others].

**persist**: bios settings persist themselves. **verify**: step 6 (numa node), `sudo lspci -vv -s <bus id> | grep LnkSta`
reads `Speed 16GT/s, Width x16` for both cards (bus ids from `nvidia-smi --query-gpu=pci.bus_id --format=csv`).

## 3. kernel command line

add to `GRUB_CMDLINE_LINUX` in `/etc/default/grub`:

```text
iommu=pt
```

then `sudo update-grub` (or `grub2-mkconfig -o /boot/grub2/grub.cfg`) and reboot. `iommu=off` is the fallback if
peer access misbehaves under passthrough [inferred; the cuda guide's iommu rule is about pcie p2p, documented].

`iomem=relaxed` is **not** on the command line now. the owner allows it only for the overclock sweep and the soaks,
whose junction temperature only the `gddr6` tool can read; steps 13 and 17 add it and take it off again.

**persist**: grub. **verify**: `cat /proc/cmdline` contains `iommu=pt` and not `iomem=relaxed`;
`sudo dmesg | grep -i 'iommu.*passthrough'` reports the passthrough default domain [inferred wording].

## 4. driver, toolkit and the engine's build

1. install the **R610 branch or newer** driver: the engine's ptx isa 9.3 is jit-compiled by the driver, and R610 is
   the first branch whose jit reads it (PINS.md). install the **open kernel module** first (the default for ampere on
   current branches). whether the 3090's nvlink works on the open module is not established [inferred]; step 8
   decides it and falls back to the proprietary module of the same version.
2. install the matching cuda toolkit (13.3 pairs with R610): the build needs its `ptxas` for the static gate, which
   on linux runs `ptxas -v` on every image with spills as errors. no other part of the toolkit is used; the engine
   loads `libcuda.so.1` itself.
3. build the engine with the latest llvm and libc++ (`cmake --preset release`, then `dev` for the tests). the linux
   build is the first to run the `ptxas` gate on every image: a spill or a register count over a kernel's declared
   budget fails the build.

**persist**: the package manager. **verify**:
- `nvidia-smi` prints a driver version of 610 or later; `cat /proc/driver/nvidia/version` says which module (`Open
  Kernel Module` or not);
- `ctest --preset dev` passes, and the release build's `ptx gate` steps all succeed;
- every geezuz program refuses an older driver itself, as "driver too old" [checked here: engine.device's start];
- `run geezuz-soak clocks` gets past opening the cards (it then refuses the clock until step 15; that is expected).

## 5. persistence mode

```sh
sudo systemctl enable --now nvidia-persistenced
sudo reboot
```

links stay `<inactive>` until persistence mode is on, and one report needed a reboot after enabling it [measured by
others]. the daemon is also what the platform unit orders itself after (step 11).

**persist**: the daemon's unit. **verify**: `nvidia-smi -q | grep 'Persistence Mode'` reads `Enabled` for both cards.

## 6. numa node 0 for everything

both slots hang off cpu0 [documented], so every pinned host buffer (the embedding table, the rings) and the resident
loop's host thread belong on numa node 0; a wrong binding costs a little, not a lot [measured by others; U14 measures
it on this box].

**verify**:

```sh
for bus in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader); do
	cat /sys/bus/pci/devices/$(echo "$bus" | tr 'A-Z' 'a-z' | sed 's/^0000\(....\)/\1/')/numa_node
done
numactl -H
```

both cards read `0`. under NPS4 they read the node that owns the root ports; use that node wherever this document says
0 (the hugepage path in `data/platform/geezuz-hugepages.service` too).

**the host thread**: run every geezuz program under `numactl --cpunodebind=0 --membind=0` (the `run` alias). that binds
all its threads to node 0's cores and its memory, hugepages included, to node 0. pinning the resident loop's one host
thread to a single cpu0 core is the server's to do when it starts the loop; the server (apps lane) is not in the tree
yet, so this document can only require the node binding [inferred: the per-core pin is not implemented].

**persist**: the prefix is part of every command and of the server's unit when it exists.

## 7. hugepages on node 0

the engine's pinned host memory (`engine.hugepage`) is anonymous 2 MiB hugetlb pages, each allocation rounded up to
whole pages and faulted in at setup. what it needs:

| allocation | bytes | pages | where the number comes from |
|---|---|---|---|
| the embedding table (the host image's payload) | 2,542,796,800 | 1,213 | `model::embedding_host_bytes` = 248,320 × 5,120 × 2 (device/model.h); engine.image asserts the host image holds exactly it |
| the rings, masks, row-token and control staging, the soak's state cells and tallies, ubench's mapped tables | one to a few pages each | ~20-40 | each `MappedArray` rounds up to a whole page (engine.hugepage); the decode lane's ring sizes are not in the tree yet [inferred] |

reserve **1,280 pages (2.5 GiB) on node 0**: the table plus 67 pages of headroom.

```sh
sudo install -m 644 data/platform/geezuz-hugepages.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable geezuz-hugepages.service
sudo reboot
```

the unit reserves early in boot, before memory fragments, and fails if the reservation fell short.

**persist**: the unit. **verify**:

```sh
cat /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages    # 1280
cat /sys/devices/system/node/node0/hugepages/hugepages-2048kB/free_hugepages  # 1280 while nothing runs
grep Huge /proc/meminfo
```

a geezuz program that cannot get its pages says "no hugepages for pinned host memory" [checked here: the soak's report
of the error]. hugetlb allocations under `--membind=0` come from node 0's pool [documented: the kernel's hugetlb
mempolicy; inferred for this box].

## 8. nvlink: NV4, and which kernel module

```sh
nvidia-smi topo -m          # GPU0 <-> GPU1 reads NV4
nvidia-smi nvlink -s        # each card: 4 links, each 14.062 GB/s
run geezuz-ubench link_health
```

`link_health` (U13) reads each link's version and speed and the peer path's rank and native atomics; its `operator:`
line says what to confirm against the two commands. every geezuz program also refuses, at start, a machine whose eight
links do not all join the peer or whose cards cannot map each other's memory [checked here: engine.device's open].

**decision**: if the links read inactive on the open module (with persistence on and after a reboot), install the
proprietary module of the same driver version, reboot, and repeat this step. record which module holds in PINS.md
(the coordinator's ledger).

**persist**: the module choice is the package installed. **verify**: the three commands above, after every reboot.

## 9. power limit: 400 w

the fe vbios maximum [documented]. power the cap removes comes out of the core clock, and the row budget needs it.
apply it by hand now; step 11 makes it persistent with the clock lock:

```sh
sudo nvidia-smi -i 0 -pl 400
sudo nvidia-smi -i 1 -pl 400
```

**verify**: `nvidia-smi -q -d POWER` reads `Current Power Limit : 400.00 W` on both cards; the soak's readings print
"of a 400.0 W limit" for both cards [checked here: the report renders the driver's enforced limit].

## 10. the microbenchmarks, in order

`run geezuz-ubench` with no names runs every benchmark in the catalogue's order; name benchmarks to run some. exit
status 0 when every selected benchmark ran and none failed its gate. run them in this order, because each group's
results are preconditions of the next:

1. **the self-checks**: `check_fragments check_reduce check_transcendentals check_stage_ring check_order decode_exact
   probe_mma`. each holds or fails; a failure means a primitive the kernels are built from is wrong on the hardware,
   and nothing measured after it means anything. stop and report.
2. **`link_health`** (U13): done in step 8.
3. **`memory_clock`** (U3): its `operator:` line lists three attempts to get p0's memory clock under cuda, to be tried
   one at a time, in order, rerunning the benchmark after each. expected: p2 at 9501 mhz [measured by others]. keep the
   first attempt that reads 9751 under load with no throughput regression in `read_stream`, and persist it with the
   matching file in `data/platform/` (table below); if none holds, keep p2. it decides the budget's base (912 or 936
   gb/s) and the clock the overclock counts from.

   | attempt that held | persist with |
   |---|---|
   | the first (sm and memory lock in one call) | `memory-lock.conf.in`, filled like the unit in step 11, as a drop-in |
   | the second (cuda clocks override) | `cuda-clocks.conf` as a drop-in |
   | the third (the nvidia-modeset parameter) | `nvidia-modeset.conf` in `/etc/modprobe.d/`, then rebuild the initramfs |
   | none | nothing: the cards stay in p2 |

4. **`read_stream`** (U1): **the phase-b gate**. its best point must reach 93% of the measured clock's peak; a failure
   stops all kernel tuning until the cause is found. it also picks the producer warp count.
5. **`sustained_clock`** (U18): a 10-minute soak of both cards at 400 w. run it unlocked first: the sustained sm clock
   it reports is the ceiling. then lock one step below it (`sudo nvidia-smi -i N -lgc F,F`, its `operator:` line) and
   rerun: accept the lock **only if the `sw_power_cap` counter stays flat** over the soak. if it climbs, the lock is
   fake (throttling turns it back into a floating clock): drop F one step and rerun. F goes into step 11.
6. **everything else, with the lock in force**: the remaining benchmarks, in the runner's order. several need the sm
   clock locked (their `operator:` lines say so); `host_fetch` (U14) runs twice, once with `--membind=1`, as its line
   says. each prints the decision it settles; record every result in the ubench lane's tracker.

`offset_step` (U19) is the overclock sweep, step 13; it is not part of a plain run.

## 11. the platform unit: power and clocks on every boot

with F from step 10.5:

```sh
sed 's/@SM_LOCK_MHZ@/F/g' data/platform/geezuz-platform.service.in | sudo tee /etc/systemd/system/geezuz-platform.service
sudo systemctl daemon-reload
sudo systemctl enable --now geezuz-platform.service
```

and, if step 10.3 kept the first attempt, its drop-in (which restates the unit's lines with the memory lock):

```sh
sudo mkdir -p /etc/systemd/system/geezuz-platform.service.d
sed 's/@SM_LOCK_MHZ@/F/g' data/platform/memory-lock.conf.in | sudo tee /etc/systemd/system/geezuz-platform.service.d/memory.conf
sudo systemctl daemon-reload && sudo systemctl restart geezuz-platform.service
```

(the second attempt's drop-in is `cuda-clocks.conf`, copied as is.) the unit runs after `nvidia-persistenced`, which
keeps the driver loaded so the settings stay [documented].

**persist**: the unit. **verify**, after a reboot:

```sh
systemctl status geezuz-platform.service                      # active (exited), no failed ExecStart
nvidia-smi -q -d POWER | grep 'Current Power Limit'            # 400.00 W, both cards
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv -lms 100   # under load: F, the memory clock, P2 (or P0)
run geezuz-soak clocks                                         # the memory clock under load, against the accepted one
```

## 12. bake the images

the soak hashes the baked card images where they lie on the cards, so they are baked before it. the bake is cpu
work: on the box, or on the mac and copied (every program verifies each image's payload digest when it opens it).

```sh
run geezuz-bake ~/models/Qwen3.8-27B-FP8 /var/lib/geezuz/images
```

it verifies every checkpoint file against its published SHA-256 before reading it, writes `card0.image`, `card1.image`
and `host.image`, and prints each payload's SHA-256. **verify**: it exits 0 and prints three payload digests; the
soak and the engine refuse an image whose payload does not hash to the digest in its header [checked here: the soak
refuses a corrupted payload, the other card's image and a missing file].

## 13. the memory-overclock sweep (U19)

this characterises the overclock; it accepts nothing. the clock the engine runs at is accepted only by a soak (steps
14 and 17).

1. add `iomem=relaxed` to the kernel command line (step 3's procedure) and reboot: the `gddr6` tool reads the memory
   junction temperature through /dev/mem, and nvml cannot read it on the 3090 [measured by others]. the tool's exact
   interface was not read (its repository was unreachable when the hardware notes were written) [inferred].
2. log the junction temperature with timestamps for the whole sweep, in its own terminal (for instance through
   moreutils' `ts`):

   ```sh
   sudo gddr6 | ts '%F %T' > /var/log/geezuz/junction-sweep.log
   ```

3. bring the cards to operating temperature first: run the soak's load for ten minutes
   (`run geezuz-soak integrity /var/lib/geezuz/images 10`; only its heat matters here, not its verdict), so every step
   measures hot memory.
4. one offset step per run, as `offset_step`'s `operator:` line says (+250 in linux units, +125 mhz, through nvml or
   nvidia-settings; an offset breaks a `-lmc` lock, so with step 10.3's first attempt in force, remove that drop-in
   for the sweep). after each step:

   ```sh
   nvidia-smi --query-gpu=clocks.mem,pstate --format=csv      # the clock the step really set
   run geezuz-ubench offset_step
   ```

   and note the step's highest junction temperature from the log.
5. **decision**: the edr knee is the step where efficiency against the clock's peak falls; the candidate is the knee
   minus two steps, at a junction temperature at or below 95 °c at worst-case ambient [inferred, from hw-dram's
   procedure]. record the candidate; it is soaked in step 17, after the engine reproduces its goldens.
6. set the offset back to 0 (and restore step 10.3's drop-in) before step 14.

`iomem=relaxed` stays on through step 14 and comes off after it.

## 14. the acceptance soak at the stock clock (phase b)

the engine refuses to start until a soak has accepted the clock its cards run at (step 15), so the stock clock is
soaked here, with the junction temperature logged as in step 13.2.

1. check the clock the cards run at under load:

   ```sh
   run geezuz-soak clocks
   ```

   it loads both cards with the soak's pattern for two seconds, then reads each card's memory clock while the pattern
   runs, and prints it (expected: `9501 MHz in P2`, or `9751 MHz in P0` if step 10.3 held). before step 15 it refuses
   with "no memory clock is accepted on this box yet": exit status 1, as designed [checked here].
2. soak for several hours at temperature [the duration is the owner's call; hw-dram asks for several hours]:

   ```sh
   run geezuz-soak integrity /var/lib/geezuz/images 480
   ```

   it prints each card's pass plan first (a 13,701,992,960-byte image is 205 slices of 64 MiB per pass, about a
   second and a half each, so about five minutes per pass [inferred: one SHA-256 chain on one SM]; the pattern region
   is 11,529,144,832 bytes per card, everything the engine writes while it serves). then, back to back on both cards
   at once:
   - **the image lane**: the card's whole weight image hashed where it lies, pass after pass, each pass's SHA-256 held
     against the payload digest the baker stamped in the image;
   - **the pattern lane**: a pattern written over the region and read back, round after round; every word gets its
     own value in a round, and an odd round stores the inverse of the round before, so every bit of every cell holds
     both values;
   - **the readings**: two seconds in, each card's memory clock and its `sw_power_cap`, `sw_thermal_slowdown` and
     `hw_thermal_slowdown` counters (and `hw_power_brake_slowdown` where nvml keeps it) are taken as the baseline;
     after every image step, any change to the clock or growth of a counter is a finding.

   it stops at the **first finding**, drains the in-flight work, and prints the finding with its wall time (UTC, to
   set beside the junction log), the time into the soak, and both cards' memory clock and state, sm clock, power and
   limit, clock-event reasons and slowdown counters; exit status 1. a hung launch, a driver failure or an unkept
   counter also exit 1. a soak that ends before each card finishes one whole pass vouches for nothing (exit 1). a clean
   soak prints the clock it held each card at, the passes and rounds each card got through, and exits 0 [checked here:
   the loop's control, every finding and its report against test doubles; not run on the cards].
3. **accept only on**: exit status 0; the junction log's maximum at or below 95 °c; no thermal or power clock events
   (the soak fails on those itself).
4. take `iomem=relaxed` off the kernel command line and reboot (step 3). **verify**: `cat /proc/cmdline`.

what the soak does not do yet: the golden-hash leg of the determinism harness (geezuz-ref's goldens against the
engine's logits) needs the engine, which is not in the tree. the stock clock is accepted on the on-device checks alone;
the overclock's acceptance (step 17) adds the goldens.

## 15. record the accepted clock: the engine's start refuses any other

the accepted memory clock is a fact of the box, held once, in the build: `engine::platform::accepted_memory_clock` in
`src/platform/clock.cc`. it is absent until this step, and while it is absent the engine refuses to start. with the
clock the clean soak printed (say 9501 mhz in P2):

```cpp
/* PIN(accepted-memory-clock): the memory clock the acceptance soak accepted on this box */
export inline constexpr std::optional<AcceptedClock> accepted_memory_clock = AcceptedClock{.megahertz = 9'501, .state = device::PerformanceState::P2};
```

and an `accepted-memory-clock` entry in PINS.md (the coordinator's): the date, the driver version and kernel module, the
clock and state, the soak's duration, passes and rounds per card, the junction log's maximum, and the reason. rebuild.

the engine's start runs `engine::platform::check_memory_clocks` right after the cards open and before it loads a
byte of its images: it loads both cards for two seconds, reads each card's memory clock under load, and refuses with a
named error (exit nonzero) if either differs from the accepted clock in megahertz or performance state [checked here:
against the driver double, every vendor call's failure reported].

**verify**: `run geezuz-soak clocks` exits 0 and prints "accepted"; after a reboot too (the unit re-applies the
settings that produce the clock).

## 16. phase c: correctness on the cards

the tools exist; the engine side of these runs does not yet (the decode and server lanes are not in the tree).

1. the noise floor the hardware's own accumulation causes:
   `run geezuz-ref noise-floor ~/models/Qwen3.8-27B-FP8 text <file>`.
2. record the goldens from the cpu reference, each with its reason:
   `run geezuz-ref golden record ~/models/Qwen3.8-27B-FP8 <golden> <prompts> "<reason>"`.
3. with the engine (when it exists): the baked images load (a mismatched image is refused at open); its teacher-forced
   dump against the reference's, `geezuz-ref rulers <reference.dump> <engine.dump>`, within the noise floor; its greedy
   run, `geezuz-ref decode-ruler ~/models/Qwen3.8-27B-FP8 <engine-run.dump>`, with no divergence; its golden hashes,
   `geezuz-ref golden compare <golden> <candidate>`, every entry reproducing, for every slot mix.
4. the server end to end with 8 parallel streams, tools and thinking.

## 17. phase d: accepting the overclock

after step 2 runs under the harness:

1. apply the candidate offset from step 13 persistently: a drop-in for `geezuz-platform.service` that applies it the
   way the sweep did (the nvml call or nvidia-settings line `offset_step`'s `operator:` line names), and without a
   `-lmc` lock [inferred: the exact headless command is whatever step 13 used; no drop-in is shipped for it].
2. `iomem=relaxed` on again (step 13.1), the junction log running, then the acceptance soak of step 14 at the new clock,
   for several hours, plus the golden hashes of step 16.3 reproducing bit for bit at that clock (the soak's golden leg,
   once the engine exists).
3. on a clean soak: step 15 again with the new clock, and a new PINS.md entry; `iomem=relaxed` off again.
4. repeat the soak after every driver change and when the season changes the ambient temperature [inferred]. a
   low-rate weight-hash scrub in production is part of the plan and not in the tree.

## settings at a glance

| setting | value | persists through | verify |
|---|---|---|---|
| slots | PCIE_6, PCIE_4 (cpu0) | hardware | `numa_node` reads 0 for both |
| iommu | `iommu=pt` | grub | `/proc/cmdline` |
| `iomem=relaxed` | only during steps 13-14 and 17 | grub (removed after) | `/proc/cmdline` |
| driver | R610+, open module unless step 8 says otherwise | packages | `nvidia-smi`, `/proc/driver/nvidia/version` |
| persistence | on | `nvidia-persistenced` | `nvidia-smi -q` |
| hugepages | 1,280 × 2 MiB on node 0 | `geezuz-hugepages.service` | node0 `nr_hugepages` |
| numa binding | node 0, cpus and memory | `run` prefix | `numactl --show` inside the prefix |
| nvlink | NV4, 4 × 14.062 GB/s | bridge + persistence | `topo -m`, `nvlink -s`, `link_health` |
| power limit | 400 w | `geezuz-platform.service` | `nvidia-smi -q -d POWER` |
| sm clock lock | F from U18 | `geezuz-platform.service` | `clocks.sm` under load; `sw_power_cap` flat |
| memory clock | U3's outcome (step 10.3), later the soaked overclock | drop-in or modprobe.d | `geezuz-soak clocks` exits 0 |
| accepted memory clock | from the clean soak | the build (`src/platform/clock.cc`), PINS.md | `geezuz-soak clocks` exits 0 |
| backplate fans | on | hardware | the junction log during the soaks |
