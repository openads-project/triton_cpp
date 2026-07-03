// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(TRITON_CPP_ENABLE_CUDA_SHM)
#include <cuda_runtime_api.h>
#endif

namespace triton_cpp {

#if defined(TRITON_CPP_ENABLE_CUDA_SHM)

/**
 * @brief Throw a descriptive exception when a CUDA Runtime API call fails.
 * @param status Status returned by the CUDA Runtime API.
 * @param operation Human-readable name of the failed operation.
 * @throws std::runtime_error if @p status is not cudaSuccess.
 */
inline void throw_on_cuda_error(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
}

/**
 * @brief Check whether this process can allocate CUDA IPC shared memory.
 * @param reason Optional destination for a diagnostic when support is unavailable.
 * @return true when at least one usable CUDA device is visible, otherwise false.
 */
inline bool LocalCudaSharedMemorySupported(std::string* reason = nullptr) {
  int device_count = 0;
  const auto status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    if (reason != nullptr) {
      *reason = cudaGetErrorString(status);
    }
    cudaGetLastError();
    return false;
  }
  if (device_count <= 0) {
    if (reason != nullptr) {
      *reason = "no CUDA-capable device is visible to the client";
    }
    return false;
  }
  return true;
}

/**
 * @brief Own a CUDA device allocation exportable through a CUDA IPC handle.
 *
 * The allocation is created on the current CUDA device and released when the
 * region is destroyed. Regions are deliberately neither copyable nor movable.
 */
class CudaSharedMemoryRegion {
 public:
  /**
   * @brief Allocate a CUDA shared-memory region on the current device.
   * @param name Name used when registering the region with Triton.
   * @param size Allocation size in bytes.
   * @throws std::runtime_error if allocation or IPC handle creation fails.
   */
  CudaSharedMemoryRegion(const std::string& name, std::int64_t size) : name_{name}, size_{size} {
    throw_on_cuda_error(cudaGetDevice(&device_id_), "cudaGetDevice");
    throw_on_cuda_error(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), static_cast<std::size_t>(size_)), "cudaMalloc");
    throw_on_cuda_error(cudaIpcGetMemHandle(&ipc_handle_, device_ptr_), "cudaIpcGetMemHandle");
  }

  /** @brief Release the owned CUDA device allocation. */
  ~CudaSharedMemoryRegion() {
    if (device_ptr_ != nullptr) {
      cudaFree(device_ptr_);
    }
  }

  CudaSharedMemoryRegion(const CudaSharedMemoryRegion&) = delete;
  CudaSharedMemoryRegion& operator=(const CudaSharedMemoryRegion&) = delete;
  CudaSharedMemoryRegion(CudaSharedMemoryRegion&&) = delete;
  CudaSharedMemoryRegion& operator=(CudaSharedMemoryRegion&&) = delete;

  /** @return Device address of the allocation. */
  uint8_t* getDeviceAddress() const { return device_ptr_; }
  /** @return CUDA IPC handle for importing the allocation. */
  const cudaIpcMemHandle_t& getIpcHandle() const { return ipc_handle_; }
  /** @return CUDA device ordinal that owns the allocation. */
  std::size_t getDeviceId() const { return static_cast<std::size_t>(device_id_); }
  /** @return Allocation size in bytes. */
  std::int64_t getSize() const { return size_; }
  /** @return Name used to register this region with Triton. */
  const std::string& getName() const { return name_; }

 private:
  std::string name_;
  std::int64_t size_ = 0;
  int device_id_ = 0;
  uint8_t* device_ptr_ = nullptr;
  cudaIpcMemHandle_t ipc_handle_{};
};

#endif

}  // namespace triton_cpp
