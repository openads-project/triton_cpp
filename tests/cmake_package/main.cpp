// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#include <triton_cpp/types.hpp>

int main() {
  const auto value = triton_cpp::TritonDataType{1.0F};
  return std::holds_alternative<float>(value) ? 0 : 1;
}
