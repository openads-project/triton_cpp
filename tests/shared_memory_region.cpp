// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "triton_cpp/shm.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string uniqueKey() { return "/triton_cpp_test_" + std::to_string(::getpid()); }

void regionIsSizedAndVisibleThroughAnIndependentMapping() {
  // A second descriptor and mapping verify POSIX visibility independently of
  // the address returned by SharedMemoryRegion itself.
  const std::string key = uniqueKey();
  constexpr std::size_t size = 4096;
  constexpr std::array<std::uint8_t, 5> payload{1, 2, 3, 4, 5};
  ::shm_unlink(key.c_str());

  triton_cpp::SharedMemoryRegion region{key, size};
  require(region.getKey() == key, "Shared-memory key changed");
  std::memcpy(region.getAddress(), payload.data(), payload.size());

  const int fd = ::shm_open(key.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
  require(fd != -1, "Could not open shared memory through a second descriptor");

  struct stat status {};
  require(::fstat(fd, &status) == 0, "Could not inspect shared-memory size");
  require(status.st_size == static_cast<off_t>(size), "Shared-memory size changed");

  void* second_mapping = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  require(second_mapping != MAP_FAILED, "Could not create second mapping");
  require(std::memcmp(second_mapping, payload.data(), payload.size()) == 0, "Mappings do not share data");
  require(::munmap(second_mapping, size) == 0, "Could not unmap second mapping");
  require(::close(fd) == 0, "Could not close second descriptor");
}

void destructionUnlinksTheRegionName() {
  const std::string key = uniqueKey();
  { triton_cpp::SharedMemoryRegion region{key, 4096}; }

  const int after_destruction = ::shm_open(key.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
  require(after_destruction == -1, "Destructor did not unlink shared memory");
  if (after_destruction != -1) {
    ::close(after_destruction);
  }
}

void failedConstructionDoesNotLeaveANamedRegion() {
  // mmap rejects a zero-length mapping after shm_open and ftruncate succeed,
  // exercising cleanup of a partially constructed owner.
  const std::string key = uniqueKey();
  bool zero_size_failed = false;
  try {
    triton_cpp::SharedMemoryRegion invalid{key, 0};
  } catch (const std::runtime_error&) {
    zero_size_failed = true;
  }
  require(zero_size_failed, "Zero-sized mapping did not fail");

  const int after_failure = ::shm_open(key.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
  require(after_failure == -1, "Failed construction did not unlink shared memory");
  if (after_failure != -1) {
    ::close(after_failure);
  }
}

}  // namespace

int main() {
  regionIsSizedAndVisibleThroughAnIndependentMapping();
  destructionUnlinksTheRegionName();
  failedConstructionDoesNotLeaveANamedRegion();
}
