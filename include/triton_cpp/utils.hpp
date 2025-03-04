#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <common.h>

namespace triton_cpp {

/**
 * @brief Helper to print std::vector to std::ostream
 *  * 
 * @tparam T type of vector elements
 * @param os ostream to print to
 * @param data vector to print
 * @return std::ostream& os
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
 * @brief Algorithm to compute the product of a range of integers, with all values less than one ignored (especially -1, which commonly stands for the batch dimension)
 * 
 * @tparam It Iterator type of underlying container
 * @param begin start of range
 * @param end end of range
 * @return int64_t product of all values in range
 */
template <typename It>
std::int64_t accumulate_shape(It begin, It end) {
  return std::accumulate(begin, end, 1l, [](std::int64_t a, std::int64_t b) { return a * std::max(b, 1l); });
}

inline void fail_on_error(const triton::client::Error& err, std::string message = "") {
  if (!err.IsOk()) {
    throw std::runtime_error(message + ": " + err.Message());
  }
}

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
