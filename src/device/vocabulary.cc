// SPDX-License-Identifier: AGPL-3.0-only
export module engine.device:vocabulary;

import engine.driver;

/**
 * The driver boundary's typed surface, under the names the engine uses: owners, kernels, errors, readings. The raw
 * tables and native handles stay behind in `engine.driver`.
 */
export namespace engine::device {

using DeviceAddress = driver::CUdeviceptr;

using driver::Context;
using driver::DeviceArray;
using driver::Dim;
using driver::Event;
using driver::EventState;
using driver::Kernel;
using driver::MappedArray;
using driver::Program;
using driver::Residency;
using driver::Stream;
using driver::Wire;

using driver::ring_line;
using driver::RingLayout;
using driver::RingSlot;

using driver::MemoryClock;
using driver::PciAddress;
using driver::PerformanceState;

using driver::CardCount;
using driver::CardFact;
using driver::DeviceError;
using driver::DriverTooOld;
using driver::DriverUnavailable;
using driver::Failure;
using driver::KernelMissing;
using driver::KernelSignature;
using driver::Library;
using driver::LinkFault;
using driver::LoadStage;
using driver::ManagementFailure;
using driver::ManagementStatus;
using driver::ModuleRejected;
using driver::NotNvlink;
using driver::PeerUnreachable;
using driver::Status;
using driver::WrongCard;

}
