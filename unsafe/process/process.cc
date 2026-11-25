// SPDX-License-Identifier: AGPL-3.0-only
//
// A child process the caller owns: spawned from a path and arguments, its standard output read as bounded lines with
// a deadline, asked to stop with SIGTERM, and reaped with a deadline. Dropping it kills and reaps whatever is left.
export module engine.process;

export import :error;
export import :child;
