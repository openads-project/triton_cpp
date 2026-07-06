// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace triton_cpp {

namespace detail {

/** Owns one mapped POSIX shared-memory object. */
class PosixSharedMemory {
 public:
  PosixSharedMemory(std::string key, std::int64_t size) : key_{std::move(key)}, size_{size} {
    fd_ = ::shm_open(key_.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd_ == -1) {
      throw std::runtime_error("CreateSharedMemoryRegion: unable to get shared memory descriptor for shared-memory key '" + key_ +
                               "'");
    }

    if (::ftruncate(fd_, size_) == -1) {
      closeDescriptor();
      unlinkRegion();
      throw std::runtime_error("CreateSharedMemoryRegion: unable to initialize shared-memory key '" + key_ +
                               "' to requested size: " + std::to_string(static_cast<std::size_t>(size_)) + " bytes");
    }

    address_ = ::mmap(nullptr, static_cast<std::size_t>(size_), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (address_ == MAP_FAILED) {
      address_ = nullptr;
      const int failed_fd = fd_;
      closeDescriptor();
      unlinkRegion();
      throw std::runtime_error("MapSharedMemory: unable to process address space or shared-memory descriptor: " +
                               std::to_string(failed_fd));
    }

    const int mapped_fd = fd_;
    if (::close(fd_) == -1) {
      fd_ = -1;
      unmapRegion();
      unlinkRegion();
      throw std::runtime_error("CloseSharedMemory: unable to close shared-memory descriptor: " + std::to_string(mapped_fd));
    }
    fd_ = -1;
  }

  PosixSharedMemory(const PosixSharedMemory&) = delete;
  PosixSharedMemory& operator=(const PosixSharedMemory&) = delete;

  PosixSharedMemory(PosixSharedMemory&& other) noexcept
      : key_{std::move(other.key_)},
        address_{std::exchange(other.address_, nullptr)},
        fd_{std::exchange(other.fd_, -1)},
        size_{other.size_},
        owns_name_{std::exchange(other.owns_name_, false)} {}

  PosixSharedMemory& operator=(PosixSharedMemory&&) = delete;

  ~PosixSharedMemory() noexcept {
    unmapRegion();
    closeDescriptor();
    unlinkRegion();
  }

  std::uint8_t* address() const noexcept { return static_cast<std::uint8_t*>(address_); }
  const std::string& key() const noexcept { return key_; }

 private:
  void unmapRegion() noexcept {
    if (address_ != nullptr) {
      ::munmap(address_, static_cast<std::size_t>(size_));
      address_ = nullptr;
    }
  }

  void closeDescriptor() noexcept {
    if (fd_ != -1) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  void unlinkRegion() noexcept {
    if (owns_name_) {
      ::shm_unlink(key_.c_str());
      owns_name_ = false;
    }
  }

  std::string key_;
  void* address_{nullptr};
  int fd_{-1};
  std::int64_t size_;
  bool owns_name_{true};
};

}  // namespace detail

/**
 * @brief Own a POSIX shared-memory mapping used by the Triton client.
 *
 * Construction creates, sizes, and maps the region. Destruction unmaps and
 * unlinks it.
 */
class SharedMemoryRegion {
 public:
  /**
   * @brief Create and map a system shared-memory region.
   * @param key POSIX shared-memory key, including its leading slash.
   * @param size Region size in bytes.
   * @throws std::runtime_error if the region cannot be created, mapped, or closed.
   */
  SharedMemoryRegion(const std::string& key, std::int64_t size) : memory_{key, size} {}

  SharedMemoryRegion(const SharedMemoryRegion&) = delete;
  SharedMemoryRegion& operator=(const SharedMemoryRegion&) = delete;
  SharedMemoryRegion(SharedMemoryRegion&&) noexcept = default;
  SharedMemoryRegion& operator=(SharedMemoryRegion&&) = delete;
  ~SharedMemoryRegion() = default;

  /** @return Host address of the mapped region. */
  uint8_t* getAddress() const { return memory_.address(); }
  /** @return POSIX shared-memory key used for the region. */
  std::string getKey() const { return memory_.key(); }

 private:
  detail::PosixSharedMemory memory_;
};

}  // namespace triton_cpp
