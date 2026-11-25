// SPDX-License-Identifier: AGPL-3.0-only
//
// A verdict cannot accept more drafts than the cycle ran: the ledger stops the program rather than replay rows the
// ring never held.
import engine.deltanet;

auto main() -> int {
	namespace dn = engine::deltanet;
	auto const running = dn::run(dn::settled({.tokens = 4}), 2);
	auto const settled = dn::settle(running, running.rows);
	return static_cast<int>(settled.pending);
}
