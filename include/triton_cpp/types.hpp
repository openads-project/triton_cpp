// Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include <model_config.pb.h>

#include "triton_cpp/utils.hpp"

namespace triton_cpp {
/**
 * @brief Helper type template for Eigen::Vector compatible with Triton
 * 
 * @tparam T Scalar data type. If T is const, the underlying Vector will be const
 */
template <typename T>
using VectorType = typename std::conditional_t<std::is_const_v<T>,
                                               Eigen::Map<const Eigen::VectorX<typename std::remove_const_t<T>>>,
                                               Eigen::Map<Eigen::VectorX<T>>>;

/**
 * @brief Helper type template for Eigen::Matrix compatible with Triton
 * 
 * @tparam T Scalar data type. If T is const, the underlying Matrix will be const
 */
template <typename T>
using MatrixType = typename std::conditional_t<
    std::is_const_v<T>,
    Eigen::Map<const Eigen::Matrix<typename std::remove_const_t<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>,
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>;

/**
 * @brief Helper type template for Eigen::Tensor compatible with Triton
 * 
 * Note that Eigen::Tensor is officially unsupported in Eigen3, but works for
 * the mappings used here.
 * 
 * @tparam T Scalar data type. If T is const, the underlying Tensor will be const
 */
template <typename T, int rank>
using TensorType = typename std::conditional_t<
    std::is_const_v<T>,
    Eigen::TensorMap<const Eigen::Tensor<typename std::remove_const_t<T>, rank, Eigen::RowMajor, Eigen::Index>>,
    Eigen::TensorMap<Eigen::Tensor<T, rank, Eigen::RowMajor, Eigen::Index>>>;

/**
 * @brief Type alias for all possible C++ scalar types that the triton server supports
 * 
 * String and FP16 values are not represented because they require dedicated
 * storage and conversion handling.
 */
using TritonDataType =
    std::variant<bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double>;

/**
 * @brief Get a zero value in the correct C++ type, given a Triton datatype
 * 
 * @param type The triton type
 * @return TritonDataType Zero as C++ value in the correct datatype
 * @throws std::invalid_argument, if the datatype is not supported yet
 */
inline TritonDataType getZero(inference::DataType type) {
  switch (type) {
    case inference::DataType::TYPE_BOOL:
      return false;
    case inference::DataType::TYPE_UINT8:
      return static_cast<uint8_t>(0);
    case inference::DataType::TYPE_UINT16:
      return static_cast<uint16_t>(0);
    case inference::DataType::TYPE_UINT32:
      return static_cast<uint32_t>(0);
    case inference::DataType::TYPE_UINT64:
      return static_cast<uint64_t>(0);
    case inference::DataType::TYPE_INT8:
      return static_cast<int8_t>(0);
    case inference::DataType::TYPE_INT16:
      return static_cast<int16_t>(0);
    case inference::DataType::TYPE_INT32:
      return static_cast<int32_t>(0);
    case inference::DataType::TYPE_INT64:
      return static_cast<int64_t>(0);
    case inference::DataType::TYPE_FP32:
      return static_cast<float>(0);
    case inference::DataType::TYPE_FP64:
      return static_cast<double>(0);
    default:
      throw std::invalid_argument("Unsupported data type");
  }
}

/**
 * @brief Return the natural C++ alignment for a Triton datatype.
 * @param type Triton datatype to inspect.
 * @return Alignment requirement in bytes.
 * @throws std::invalid_argument if @p type is unsupported.
 */
inline std::size_t getAlignment(inference::DataType type) {
  return std::visit([](auto&& arg) { return alignof(std::decay_t<decltype(arg)>); }, getZero(type));
}

/**
 * @brief Return the alignment used when packing tensors into shared memory.
 * @param type Triton datatype to inspect.
 * @return Alignment in bytes, with a minimum of eight bytes.
 * @throws std::invalid_argument if @p type is unsupported.
 */
inline std::size_t getSharedMemoryAlignment(inference::DataType type) {
  constexpr std::size_t kMinSharedMemoryAlignment = 8;
  return std::max(getAlignment(type), kMinSharedMemoryAlignment);
}

/**
 * @brief Round an offset up to an alignment boundary.
 * @param offset Original byte offset.
 * @param alignment Required alignment in bytes. Values zero and one leave the offset unchanged.
 * @return Smallest aligned offset greater than or equal to @p offset.
 */
inline std::size_t alignUp(std::size_t offset, std::size_t alignment) {
  if (alignment <= 1) {
    return offset;
  }
  const std::size_t remainder = offset % alignment;
  return remainder == 0 ? offset : offset + (alignment - remainder);
}

/**
 * @brief Map model output names to Triton requested-output objects.
 */
using ModelOutput = std::map<std::string, std::shared_ptr<triton::client::InferRequestedOutput>>;

/**
 * @brief Couple a Triton input descriptor with its backing storage.
 *
 * Storage may be an owned host vector, externally owned host shared memory, or
 * CUDA device shared memory. Pointer validity follows the lifetime of the
 * corresponding backing allocation.
 */
struct InputData {
  /** Triton input descriptor. */
  std::shared_ptr<triton::client::InferInput> input;
  /** Owned storage used by standard host inputs. */
  std::vector<uint8_t> data;
  /** Host-mappable buffer address, or nullptr for device-only storage. */
  uint8_t* data_raw;
  /** CUDA device buffer address, or nullptr for host storage. */
  uint8_t* device_data_raw;
  /** Buffer size in bytes. */
  std::size_t data_raw_size;

  /** @brief Construct an empty input-storage descriptor. */
  InputData() = default;

  /**
   * @brief Construct an input backed by an owned host vector.
   * @param input Triton input descriptor.
   * @param data Host buffer used to initialize this object's owned storage.
   */
  InputData(std::shared_ptr<triton::client::InferInput> input, std::vector<uint8_t>&& data)
      : input{input}, data{data}, data_raw{this->data.data()}, device_data_raw{nullptr}, data_raw_size{this->data.size()} {}

  /**
   * @brief Construct an input backed by externally owned host memory.
   * @param input Triton input descriptor.
   * @param data_raw Host buffer address.
   * @param data_raw_size Buffer size in bytes.
   */
  InputData(std::shared_ptr<triton::client::InferInput> input, uint8_t* data_raw, std::size_t data_raw_size)
      : input{input}, data{}, data_raw{data_raw}, device_data_raw{nullptr}, data_raw_size{data_raw_size} {}

  /**
   * @brief Construct an input backed by CUDA shared memory.
   * @param input Triton input descriptor.
   * @param data_raw Optional host address; normally nullptr for CUDA-only storage.
   * @param device_data_raw CUDA device address.
   * @param data_raw_size Buffer size in bytes.
   */
  InputData(std::shared_ptr<triton::client::InferInput> input,
            uint8_t* data_raw,
            uint8_t* device_data_raw,
            std::size_t data_raw_size)
      : input{input}, data{}, data_raw{data_raw}, device_data_raw{device_data_raw}, data_raw_size{data_raw_size} {}

  /** @return true when the input exposes a host-accessible address. */
  bool isHostMappable() const { return data_raw != nullptr; }
  /** @return true when the input is backed by a CUDA device allocation. */
  bool isDeviceBacked() const { return device_data_raw != nullptr; }
};

/** @brief Shape, datatype, and byte-size metadata for one model tensor. */
struct InputOutputMetaData {
  /** Tensor dimensions reported by Triton or supplied by the caller. */
  const std::vector<int64_t> shape;
  /** Triton tensor datatype. */
  const inference::DataType datatype;
  /** Total tensor buffer size in bytes. */
  const int64_t bytesize;

 public:
  /**
   * @brief Build metadata and calculate the required tensor byte size.
   * @param shape Tensor dimensions. Dynamic dimensions contribute one to the size.
   * @param datatype Triton tensor datatype.
   * @throws std::invalid_argument if @p datatype is unsupported.
   */
  InputOutputMetaData(const std::vector<int64_t>& shape, const inference::DataType& datatype)
      : shape{shape},
        datatype{datatype},
        bytesize{accumulate_shape(shape.begin(), shape.end()) *
                 std::visit([](auto&& arg) { return sizeof(arg); }, getZero(datatype))} {}
};

}  // namespace triton_cpp
