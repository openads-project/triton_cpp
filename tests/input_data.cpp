// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "triton_cpp/types.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void defaultInputIsEmpty() {
  triton_cpp::InputData empty{};
  require(empty.data_raw == nullptr, "Empty input has a host pointer");
  require(empty.device_data_raw == nullptr, "Empty input has a device pointer");
  require(empty.data_raw_size == 0, "Empty input has a nonzero size");
}

void copiedOwnedInputRebindsItsDataPointer() {
  // Regression: the implicit copy once left data_raw pointing into the source
  // vector, which became dangling when the source was destroyed.
  triton_cpp::InputData original{nullptr, std::vector<std::uint8_t>{1, 2, 3, 4}};
  triton_cpp::InputData copy{original};
  require(copy.data == original.data, "Copy changed owned bytes");
  require(copy.data_raw == copy.data.data(), "Copy points into the original owned buffer");
  require(copy.data_raw != original.data_raw, "Copy aliases the original owned buffer");
}

void movedOwnedInputRetainsItsStorageAssociation() {
  triton_cpp::InputData original{nullptr, std::vector<std::uint8_t>{1, 2, 3, 4}};
  triton_cpp::InputData moved{std::move(original)};
  require(moved.data_raw == moved.data.data(), "Move does not point into the moved owned buffer");
  require(moved.data_raw_size == moved.data.size(), "Move changed the owned buffer size");
}

void assignmentPreservesOwnedStorageAssociation() {
  triton_cpp::InputData copy{nullptr, std::vector<std::uint8_t>{1, 2, 3, 4}};
  triton_cpp::InputData copy_assigned;
  copy_assigned = copy;
  require(copy_assigned.data_raw == copy_assigned.data.data(), "Copy assignment did not rebind owned storage");
  require(copy_assigned.data_raw != copy.data_raw, "Copy assignment aliases the source owned buffer");

  triton_cpp::InputData move_assigned;
  move_assigned = std::move(copy_assigned);
  require(move_assigned.data_raw == move_assigned.data.data(), "Move assignment did not rebind owned storage");
  require(copy_assigned.data_raw == nullptr, "Moved-from input retains its host pointer");
}

void externalPointersRemainExternalWhenCopiedOrMoved() {
  std::uint8_t external_buffer[4]{};
  triton_cpp::InputData external{nullptr, external_buffer, sizeof(external_buffer)};
  triton_cpp::InputData external_copy{external};
  require(external_copy.data_raw == external_buffer, "Copy changed an external host pointer");
  require(external_copy.data_raw_size == sizeof(external_buffer), "Copy changed an external buffer size");

  triton_cpp::InputData external_move{std::move(external)};
  require(external_move.data_raw == external_buffer, "Move changed an external host pointer");
  require(external.data_raw == nullptr, "Moved-from external input retains its host pointer");
}

}  // namespace

int main() {
  defaultInputIsEmpty();
  copiedOwnedInputRebindsItsDataPointer();
  movedOwnedInputRetainsItsStorageAssociation();
  assignmentPreservesOwnedStorageAssociation();
  externalPointersRemainExternalWhenCopiedOrMoved();
}
