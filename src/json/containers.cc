// SPDX-License-Identifier: AGPL-3.0-only
export module engine.json:containers;

import std;
import engine.core;
import engine.simdjson;
import :bounds;
import :error;
import :text;
import :value;

namespace engine::json {

/**
 * The members of one object, read in document order under bounds `B`. Iterating checks every name: its length,
 * the member count, and that no name repeats. `find` and `member` look a name up from the object's start, compare
 * names unescaped, and start iteration over; they check no names, so a reader that needs every name checked
 * iterates.
 */
export template<Bounds B>
    requires Enforceable<B>
class Object {
public:
	/** The next member, or nothing at the end; the previous member's value is skipped where it was not read. */
	[[nodiscard]] auto next() -> core::Result<std::optional<Member<B>>, ReadError> {
		auto const field = cursor_.next();
		if (!field) {
			return refuse(field.error());
		}
		if (!*field) {
			return std::nullopt;
		}
		if (++count_ > B.members) {
			return refuse(ParseFault::TooManyMembers);
		}
		auto& [name, value] = **field;
		if (name.size() > B.string_bytes) {
			return refuse(ParseFault::StringTooLong);
		}
		if (!names().insert(name).second) {
			return refuse(ParseFault::DuplicateKey);
		}
		return Member<B>{.name = name, .value = Value<B>{std::move(value), reader_, depth_}};
	}

	/** The value of the member named `name`, if there is one. */
	[[nodiscard]] auto find(std::string_view name) -> core::Result<std::optional<Value<B>>, ReadError> {
		names().clear();
		count_ = 0;
		auto found = cursor_.find(name);
		if (!found) {
			return refuse(found.error());
		}
		if (!*found) {
			return std::nullopt;
		}
		return Value<B>{std::move(**found), reader_, depth_};
	}

	/** The value of the member named `name`, which the reader requires. */
	[[nodiscard]] auto member(std::string_view name) -> core::Result<Value<B>, ReadError> {
		auto found = find(name);
		if (!found) {
			return std::unexpected(found.error());
		}
		if (!*found) {
			return std::unexpected(ReadError{.cause = MissingMember{.name = std::string{name}}});
		}
		return std::move(**found);
	}

private:
	friend class Value<B>;

	Object(simdjson::ObjectCursor cursor, Reader<B>& reader, std::size_t depth) : cursor_{std::move(cursor)}, reader_{reader}, depth_{depth} {
		names().clear();
	}

	[[nodiscard]] auto names() -> std::unordered_set<std::string_view>& {
		return reader_.names_[depth_];
	}

	simdjson::ObjectCursor cursor_;
	Reader<B>& reader_;
	/** The containers around this object's members, this one included. */
	std::size_t depth_;
	std::size_t count_ = 0;
};

/** The elements of one array, read in document order under bounds `B`; as `Object`. */
export template<Bounds B>
    requires Enforceable<B>
class Array {
public:
	/** The number of elements, counted ahead without reading them; only before the first `next`. */
	[[nodiscard]] auto size() -> core::Result<std::size_t, ReadError> {
		auto const count = cursor_.count();
		if (!count) {
			return refuse(count.error());
		}
		if (*count > B.members) {
			return refuse(ParseFault::TooManyMembers);
		}
		return *count;
	}

	/** The next element, or nothing at the end. */
	[[nodiscard]] auto next() -> core::Result<std::optional<Value<B>>, ReadError> {
		auto element = cursor_.next();
		if (!element) {
			return refuse(element.error());
		}
		if (!*element) {
			return std::nullopt;
		}
		if (++count_ > B.members) {
			return refuse(ParseFault::TooManyMembers);
		}
		return Value<B>{std::move(**element), reader_, depth_};
	}

private:
	friend class Value<B>;

	Array(simdjson::ArrayCursor cursor, Reader<B>& reader, std::size_t depth) : cursor_{std::move(cursor)}, reader_{reader}, depth_{depth} {}

	simdjson::ArrayCursor cursor_;
	Reader<B>& reader_;
	std::size_t depth_;
	std::size_t count_ = 0;
};

}
