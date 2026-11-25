# SPDX-License-Identifier: AGPL-3.0-only
add_library(engine_net)

# The platform seam: one wait backend per machine, both defining the partition engine.net:wait with the same
# surface. Nothing above it tests the platform.
if(ENGINE_PLATFORM STREQUAL "darwin")
	set(wait_backend wait.darwin.cc)
else()
	set(wait_backend wait.linux.cc)
endif()

target_sources(engine_net PUBLIC FILE_SET CXX_MODULES FILES
	net.cc
	error.cc
	fd.cc
	fact.cc
	${wait_backend}
	socket.cc
	poller.cc
	mailbox.cc
)
target_link_libraries(engine_net PUBLIC engine_language_profile engine_core)
