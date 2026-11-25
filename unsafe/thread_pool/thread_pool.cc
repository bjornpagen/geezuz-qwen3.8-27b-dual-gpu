// SPDX-License-Identifier: AGPL-3.0-only
//
// A fork-join pool for the cpu reference: a fixed set of worker threads that share out the indices of one synchronous
// call. The reference's parallelism is exactly that shape (every output row of a matmul is independent and writes its
// own place), and the sender vocabulary has no bulk algorithm; a sender scheduler would also have to queue operation
// states by address. So the pool is a synchronous primitive: the work is borrowed for the call alone, and the call
// returns only after every index has run and every worker has let go of it.
export module engine.thread_pool;

import std;
import engine.core;

namespace engine::thread_pool {

/** One call's work, erased so the workers can run it: the borrowed callable and its index range. */
struct Job {
	void (*run)(void const* work, std::size_t index);
	void const* work;
	std::size_t count;
};

export class Pool {
public:
	/** A pool of `workers` threads beside the caller's; the caller works too, so zero workers is a serial pool. */
	explicit Pool(std::size_t workers) {
		threads_.reserve(workers);
		for (auto index = std::size_t{0}; index < workers; ++index) {
			threads_.emplace_back([this](std::stop_token const& stop) {
				serve(stop);
			});
		}
	}

	Pool(Pool const&) = delete;
	auto operator=(Pool const&) -> Pool& = delete;
	Pool(Pool&&) = delete;
	auto operator=(Pool&&) -> Pool& = delete;

	~Pool() {
		{
			auto const lock = std::scoped_lock{mutex_};
			for (auto& thread : threads_) {
				thread.request_stop();
			}
		}
		posted_.notify_all();
	}

	/** Threads that run work, the caller's included. */
	[[nodiscard]] auto threads() const -> std::size_t {
		return threads_.size() + 1;
	}

	/**
	 * Runs `work(index)` for every index below `count`, each exactly once, spread over the workers and the calling
	 * thread, and returns when all have run. Indices are claimed in ascending order but run concurrently, so `work`
	 * must make each index's effect independent of the others'.
	 */
	template<class Work>
	    requires std::invocable<Work const&, std::size_t>
	auto share(std::size_t count, Work const& work) -> void {
		if (count == 0) {
			return;
		}
		auto const run = [](void const* erased, std::size_t index) {
			/* SAFETY: `erased` is the address of `work` below, alive until this call returns. */
			std::invoke(*static_cast<Work const*>(erased), index);
		};
		{
			auto const lock = std::scoped_lock{mutex_};
			core::invariant(!job_.has_value());
			job_ = Job{.run = run, .work = &work, .count = count};
			next_ = 0;
			finished_ = 0;
			++generation_;
		}
		posted_.notify_all();
		work_on(Job{.run = run, .work = &work, .count = count});
		auto lock = std::unique_lock{mutex_};
		done_.wait(lock, [this, count] {
			return finished_ == count && busy_ == 0;
		});
		job_.reset();
	}

private:
	/** Claims and runs indices of `job` until none is left. */
	auto work_on(Job const job) -> void {
		while (true) {
			auto index = std::size_t{0};
			{
				auto const lock = std::scoped_lock{mutex_};
				if (next_ >= job.count) {
					return;
				}
				index = next_;
				++next_;
			}
			job.run(job.work, index);
			auto const lock = std::scoped_lock{mutex_};
			++finished_;
			if (finished_ == job.count) {
				done_.notify_all();
			}
		}
	}

	auto serve(std::stop_token const& stop) -> void {
		auto seen = std::uint64_t{0};
		while (true) {
			auto job = Job{};
			{
				auto lock = std::unique_lock{mutex_};
				posted_.wait(lock, [&] {
					return stop.stop_requested() || (job_.has_value() && generation_ != seen);
				});
				if (stop.stop_requested()) {
					return;
				}
				seen = generation_;
				job = job_.value();
				++busy_;
			}
			work_on(job);
			auto const lock = std::scoped_lock{mutex_};
			--busy_;
			done_.notify_all();
		}
	}

	std::mutex mutex_;
	std::condition_variable posted_;
	std::condition_variable done_;
	std::optional<Job> job_;
	std::uint64_t generation_ = 0;
	std::size_t next_ = 0;
	std::size_t finished_ = 0;
	std::size_t busy_ = 0;
	// Last, so the threads stop and join before the state they read is destroyed.
	std::vector<std::jthread> threads_;
};

}
