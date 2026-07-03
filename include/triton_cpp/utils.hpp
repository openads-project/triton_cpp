// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <common.h>

namespace triton_cpp {

/**
 * @brief Write a vector in bracketed, comma-separated form.
 *
 * @tparam T Streamable vector element type.
 * @param os Stream receiving the formatted vector.
 * @param data Vector to format.
 * @return Reference to @p os.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& data) {
  os << "[";
  if (data.size() > 0) {
    for (auto d = data.begin(); d != data.end() - 1; ++d) {
      os << *d << ", ";
    }
    os << data.back();
  }
  os << "]";
  return os;
};

/**
 * @brief Compute the element count represented by a tensor shape.
 *
 * Dimensions smaller than one contribute one to the product. This treats
 * Triton's dynamic dimension marker `-1` as an unspecified unit dimension.
 *
 * @tparam It Iterator over values convertible to std::int64_t.
 * @param begin Beginning of the shape range.
 * @param end End of the shape range.
 * @return Product of all positive dimensions.
 */
template <typename It>
std::int64_t accumulate_shape(It begin, It end) {
  return std::accumulate(begin, end, 1l, [](std::int64_t a, std::int64_t b) { return a * std::max(b, 1l); });
}

/**
 * @brief Convert a failed Triton client status into a C++ exception.
 * @param err Triton client status to inspect.
 * @param message Context prepended to the Triton error message.
 * @throws std::runtime_error if @p err does not represent success.
 */
inline void fail_on_error(const triton::client::Error& err, std::string message = "") {
  if (!err.IsOk()) {
    throw std::runtime_error(message + ": " + err.Message());
  }
}

/**
 * @brief Generate an alphanumeric random string.
 * @param len Number of characters to generate.
 * @return Random string of exactly @p len characters.
 */
inline std::string randstring(std::size_t len) {
  static constexpr auto chars =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  thread_local static std::mt19937 rng{std::random_device{}()};
  thread_local static std::uniform_int_distribution<std::string::size_type> dist(0, std::strlen(chars) - 1);

  std::string result(len, '\0');
  std::generate_n(begin(result), len, [&]() { return chars[dist(rng)]; });
  return result;
}

}  // namespace triton_cpp
