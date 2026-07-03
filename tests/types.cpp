// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include "triton_cpp/types.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename T>
void requireZero(inference::DataType datatype) {
  const auto zero = triton_cpp::getZero(datatype);
  require(std::holds_alternative<T>(zero), "Datatype maps to the wrong C++ type");
  require(std::get<T>(zero) == T{}, "Datatype does not map to zero");
}

void supportedDatatypesMapToTheirZeroValuedCppTypes() {
  requireZero<bool>(inference::DataType::TYPE_BOOL);
  requireZero<std::uint8_t>(inference::DataType::TYPE_UINT8);
  requireZero<std::uint16_t>(inference::DataType::TYPE_UINT16);
  requireZero<std::uint32_t>(inference::DataType::TYPE_UINT32);
  requireZero<std::uint64_t>(inference::DataType::TYPE_UINT64);
  requireZero<std::int8_t>(inference::DataType::TYPE_INT8);
  requireZero<std::int16_t>(inference::DataType::TYPE_INT16);
  requireZero<std::int32_t>(inference::DataType::TYPE_INT32);
  requireZero<std::int64_t>(inference::DataType::TYPE_INT64);
  requireZero<float>(inference::DataType::TYPE_FP32);
  requireZero<double>(inference::DataType::TYPE_FP64);
}

void unsupportedDatatypesAreRejected() {
  bool unsupported_failed = false;
  try {
    static_cast<void>(triton_cpp::getZero(inference::DataType::TYPE_FP16));
  } catch (const std::invalid_argument&) {
    unsupported_failed = true;
  }
  require(unsupported_failed, "Unsupported datatype did not fail");
}

void offsetsAreAlignedForSharedMemory() {
  require(triton_cpp::alignUp(0, 8) == 0, "Zero offset was changed");
  require(triton_cpp::alignUp(8, 8) == 8, "Aligned offset was changed");
  require(triton_cpp::alignUp(9, 8) == 16, "Offset was not aligned upward");
  require(triton_cpp::getSharedMemoryAlignment(inference::DataType::TYPE_UINT8) == 8,
          "Shared-memory alignment is below eight bytes");
}

void tensorByteSizeIncludesShapeAndElementSize() {
  const triton_cpp::InputOutputMetaData fp32{{2, 3, 4}, inference::DataType::TYPE_FP32};
  require(fp32.bytesize == 24 * static_cast<std::int64_t>(sizeof(float)), "FP32 byte size is incorrect");
}

void dynamicDimensionsContributeOneElement() {
  // Triton reports an unresolved dimension as -1. Allocation uses one element
  // until the caller supplies a concrete shape.
  const triton_cpp::InputOutputMetaData dynamic{{-1, 3}, inference::DataType::TYPE_INT64};
  require(dynamic.bytesize == 3 * static_cast<std::int64_t>(sizeof(std::int64_t)),
          "Dynamic dimension did not contribute one element");
}

void emptyShapeRepresentsOneScalar() {
  const triton_cpp::InputOutputMetaData scalar{{}, inference::DataType::TYPE_UINT16};
  require(scalar.bytesize == static_cast<std::int64_t>(sizeof(std::uint16_t)), "Scalar byte size is incorrect");
}

}  // namespace

int main() {
  supportedDatatypesMapToTheirZeroValuedCppTypes();
  unsupportedDatatypesAreRejected();
  offsetsAreAlignedForSharedMemory();
  tensorByteSizeIncludesShapeAndElementSize();
  dynamicDimensionsContributeOneElement();
  emptyShapeRepresentsOneScalar();
}
