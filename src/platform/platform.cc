// SPDX-License-Identifier: AGPL-3.0-only
//
// The box's platform facts and their checks: the memory clock the cards were accepted at and the engine's start-time
// refusal of any other, and the acceptance soak that hashes each card's weight image where it lies and pattern-tests
// the memory the engine writes, at temperature, for hours.
export module engine.platform;

export import :clock;
export import :telemetry;
export import :soak;
export import :bench;
export import :start;
export import :report;
