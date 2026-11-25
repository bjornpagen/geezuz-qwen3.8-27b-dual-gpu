// SPDX-License-Identifier: AGPL-3.0-only
export module engine.bake:write;

import std;
import engine.core;
import engine.file_write;
import engine.image;
import engine.model;
import engine.schedule;
import engine.sha256;
import :bound;
import :error;
import :plan;
import :produce;
import :source;
import :verify;

namespace engine::bake {

/** The file name of the image for `target` in the bake's output directory. */
export [[nodiscard]] constexpr auto image_name(image::Target target) -> std::string_view {
	switch (target) {
	case image::Target::CardZero:
		return "card0.image";
	case image::Target::CardOne:
		return "card1.image";
	case image::Target::Host:
		return "host.image";
	}
	std::unreachable();
}

/** One written image: where, how long, and the digest its header records for its payload. */
export struct ImageReport {
	image::Target target;
	std::filesystem::path path;
	std::uint64_t bytes;
	sha256::Digest payload;
};

/** What a bake verified and wrote. */
export struct Report {
	std::size_t files;
	std::uint64_t checkpoint_bytes;
	image::Checkpoint checkpoint;
	Bound bound;
	std::array<ImageReport, 3> images;
};

/** A complete image, not yet under its name. */
struct Ready {
	file_write::Output output;
	ImageReport report;

	/** Puts the image in place under its name. */
	[[nodiscard]] auto commit() && -> core::Result<void, BakeError> {
		if (auto committed = std::move(output).commit(); !committed) {
			return refuse(WriteFailed{.image = std::string{image_name(report.target)}, .error = committed.error()});
		}
		return {};
	}
};

// Writes go out in runs this long: few enough calls, little enough memory.
inline constexpr std::size_t write_run_bytes = std::size_t{8} * 1024 * 1024;

/**
 * One image being written: its extents arrive in catalog order, in pieces, gathered into runs; each run is hashed by
 * the image writer and written at the offset it gives. Finishing it gives a complete image, not yet under its name.
 */
template<image::Catalog C>
class Sink {
public:
	[[nodiscard]] static auto create(std::filesystem::path path, image::Checkpoint const& checkpoint) -> core::Result<Sink, BakeError> {
		auto output = file_write::Output::create(path, image::file_bytes<C>);
		if (!output) {
			return refuse(WriteFailed{.image = std::string{image_name(C::target)}, .error = output.error()});
		}
		return Sink{std::move(path), *std::move(output), checkpoint};
	}

	/** Appends the next bytes of extent `index`, which must be the extent being written or the next one. */
	[[nodiscard]] auto put(std::size_t index, std::span<std::byte const> bytes) -> core::Result<void, BakeError> {
		if (index != extent_) {
			if (auto flushed = flush(); !flushed) {
				return flushed;
			}
			extent_ = index;
		}
		while (!bytes.empty()) {
			auto const room = write_run_bytes - run_.size();
			auto const piece = bytes.first(std::min(room, bytes.size()));
			run_.insert(run_.end(), piece.begin(), piece.end());
			bytes = bytes.subspan(piece.size());
			if (run_.size() == write_run_bytes) {
				if (auto flushed = flush(); !flushed) {
					return flushed;
				}
			}
		}
		return {};
	}

	/** Writes the header once every extent is complete: the image is then whole, waiting to be put in place. */
	[[nodiscard]] auto finish() && -> core::Result<Ready, BakeError> {
		if (auto flushed = flush(); !flushed) {
			return std::unexpected(flushed.error());
		}
		auto finished = std::move(writer_).finish();
		if (auto written = output_.write(0, finished.header); !written) {
			return refuse(WriteFailed{.image = std::string{image_name(C::target)}, .error = written.error()});
		}
		return Ready{
		    .output = std::move(output_),
		    .report = {.target = C::target, .path = path_, .bytes = image::file_bytes<C>, .payload = finished.payload},
		};
	}

private:
	Sink(std::filesystem::path path, file_write::Output output, image::Checkpoint const& checkpoint)
	    : path_{std::move(path)}, output_{std::move(output)}, writer_{checkpoint} {
		run_.reserve(write_run_bytes);
	}

	[[nodiscard]] auto flush() -> core::Result<void, BakeError> {
		if (run_.empty()) {
			return {};
		}
		auto const offset = writer_.write(extent_, run_);
		if (auto written = output_.write(offset, run_); !written) {
			return refuse(WriteFailed{.image = std::string{image_name(C::target)}, .error = written.error()});
		}
		run_.clear();
		return {};
	}

	std::filesystem::path path_;
	file_write::Output output_;
	image::Writer<C> writer_;
	std::vector<std::byte> run_;
	std::size_t extent_ = 0;
};

/** Card `card`'s image: every placement's bytes in order. */
template<Plan P, model::Card card>
[[nodiscard]] auto write_card(Source const& source, std::filesystem::path const& directory)
    -> core::Result<Ready, BakeError> {
	using C = CardCatalog<P, card>;
	auto sink = Sink<C>::create(directory / image_name(C::target), source.verified().checkpoint);
	if (!sink) {
		return std::unexpected(sink.error());
	}
	auto const placements = P::placements(card);
	for (auto index = std::size_t{0}; index < placements.size(); ++index) {
		auto produced = produce<P>(source, placements[index], card, [&](std::span<std::byte const> bytes) { return sink->put(index, bytes); });
		if (!produced) {
			return std::unexpected(produced.error());
		}
	}
	return std::move(*sink).finish();
}

/** The host image: every host tensor whole, as the checkpoint stores it. */
template<Plan P>
[[nodiscard]] auto write_host(Source const& source, std::filesystem::path const& directory)
    -> core::Result<Ready, BakeError> {
	using C = typename P::Host;
	auto sink = Sink<C>::create(directory / image_name(C::target), source.verified().checkpoint);
	if (!sink) {
		return std::unexpected(sink.error());
	}
	auto const tensors = std::span<std::size_t const>{P::host_tensors};
	for (auto index = std::size_t{0}; index < tensors.size(); ++index) {
		auto const& tensor = std::span{P::inventory}[tensors[index]];
		auto const name = checkpoint_name(tensor, schedule::Instance{.stream = model::Stream::Head, .layer = 0});
		if (auto put = sink->put(index, source.bytes(name)); !put) {
			return std::unexpected(put.error());
		}
	}
	return std::move(*sink).finish();
}

/**
 * Bakes a verified checkpoint into the card and host images in `output_directory`: checks every tensor against the
 * model description, proves the attention bound, then writes the three images and puts them in place together. A
 * refusal at any step leaves no new image under a final name.
 */
export template<Consistent P>
[[nodiscard]] auto bake_verified(Verified verified, std::filesystem::path const& output_directory) -> core::Result<Report, BakeError> {
	auto const files = verified.files.size();
	auto const checkpoint_bytes = verified.bytes;
	auto source = Source::open<P>(std::move(verified));
	if (!source) {
		return std::unexpected(source.error());
	}
	auto bound = attention_bound<P>(*source);
	if (!bound) {
		return std::unexpected(bound.error());
	}
	auto zero = write_card<P, model::Card::Zero>(*source, output_directory);
	if (!zero) {
		return std::unexpected(zero.error());
	}
	auto one = write_card<P, model::Card::One>(*source, output_directory);
	if (!one) {
		return std::unexpected(one.error());
	}
	auto host = write_host<P>(*source, output_directory);
	if (!host) {
		return std::unexpected(host.error());
	}
	if (auto committed = std::move(*zero).commit(); !committed) {
		return std::unexpected(committed.error());
	}
	if (auto committed = std::move(*one).commit(); !committed) {
		return std::unexpected(committed.error());
	}
	if (auto committed = std::move(*host).commit(); !committed) {
		return std::unexpected(committed.error());
	}
	return Report{
	    .files = files,
	    .checkpoint_bytes = checkpoint_bytes,
	    .checkpoint = source->verified().checkpoint,
	    .bound = *std::move(bound),
	    .images = {zero->report, one->report, host->report},
	};
}

/** Verifies every file `P` reads in `checkpoint_directory` against its published SHA-256, then bakes it. */
export template<class P>
    requires Consistent<P> && Published<P>
[[nodiscard]] auto bake(std::filesystem::path const& checkpoint_directory, std::filesystem::path const& output_directory)
    -> core::Result<Report, BakeError> {
	auto verified = verify(checkpoint_directory, P::files, P::commit);
	if (!verified) {
		return std::unexpected(verified.error());
	}
	return bake_verified<P>(*std::move(verified), output_directory);
}

}
