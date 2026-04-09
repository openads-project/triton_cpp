#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#if defined(TRITON_CPP_ENABLE_CUDA_SHM)
#include <cuda_runtime_api.h>
#endif

namespace triton_cpp {

#if defined(TRITON_CPP_ENABLE_CUDA_SHM)

inline void throw_on_cuda_error(cudaError_t status, const char* operation)
{
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
}

inline bool LocalCudaSharedMemorySupported(std::string* reason = nullptr)
{
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

class CudaSharedMemoryRegion {
 public:
  CudaSharedMemoryRegion(const std::string& name, std::int64_t size) : name_{name}, size_{size}
  {
    throw_on_cuda_error(cudaGetDevice(&device_id_), "cudaGetDevice");
    throw_on_cuda_error(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), static_cast<std::size_t>(size_)),
                        "cudaMalloc");
    throw_on_cuda_error(cudaIpcGetMemHandle(&ipc_handle_, device_ptr_), "cudaIpcGetMemHandle");
  }

  ~CudaSharedMemoryRegion()
  {
    if (device_ptr_ != nullptr) {
      cudaFree(device_ptr_);
    }
  }

  CudaSharedMemoryRegion(const CudaSharedMemoryRegion&) = delete;
  CudaSharedMemoryRegion& operator=(const CudaSharedMemoryRegion&) = delete;
  CudaSharedMemoryRegion(CudaSharedMemoryRegion&&) = delete;
  CudaSharedMemoryRegion& operator=(CudaSharedMemoryRegion&&) = delete;

  uint8_t* getDeviceAddress() const { return device_ptr_; }
  const cudaIpcMemHandle_t& getIpcHandle() const { return ipc_handle_; }
  std::size_t getDeviceId() const { return static_cast<std::size_t>(device_id_); }
  std::int64_t getSize() const { return size_; }
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
