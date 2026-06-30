// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "triton_cpp/shm_utils.hpp"
#include "triton_cpp/utils.hpp"

namespace triton_cpp {

class SharedMemoryRegion {
  std::string shm_key_;
  std::uint8_t* shm_addr_;
  int shm_fd_;
  std::int64_t shm_size_;

 public:
  SharedMemoryRegion(const std::string& key, std::int64_t size) : shm_key_{key}, shm_size_{size} {
    fail_on_error(triton::client::CreateSharedMemoryRegion(shm_key_, shm_size_, &shm_fd_), "CreateSharedMemoryRegion");
    fail_on_error(triton::client::MapSharedMemory(shm_fd_, 0, shm_size_, (void**)&shm_addr_), "MapSharedMemory");
    fail_on_error(triton::client::CloseSharedMemory(shm_fd_), "CloseSharedMemory");
  }

  ~SharedMemoryRegion() {
    triton::client::UnmapSharedMemory(shm_addr_, shm_size_);
    triton::client::UnlinkSharedMemoryRegion(shm_key_);
  }

  uint8_t* getAddress() const { return shm_addr_; }
  std::string getKey() const { return shm_key_; }
};

}  // namespace triton_cpp
