#pragma once

#include <type_traits>

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
using VectorType = typename std::conditional_t<
    std::is_const_v<T>,
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
 * Note that Eigen::Tensor is officially unsopported in Eigen3, but seems to work well
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
 * TODO: Check, what should happen to String and float16. Make sure that bool works as intended
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
 * @brief Type alias for internal storage of ModelOutputs
 * 
 */
using ModelOutput = std::map<std::string, std::shared_ptr<triton::client::InferRequestedOutput>>;

/**
 * @brief Helper struct to couple a Triton input with its underlying data buffer
 * 
 */
struct InputData {
  std::shared_ptr<triton::client::InferInput> input;
  std::vector<uint8_t> data;
  uint8_t* data_raw;
  uint8_t* device_data_raw;
  std::size_t data_raw_size;
  InputData() = default;
  InputData(std::shared_ptr<triton::client::InferInput> input, std::vector<uint8_t>&& data)
      : input{input},
        data{data},
        data_raw{this->data.data()},
        device_data_raw{nullptr},
        data_raw_size{this->data.size()} {}
  InputData(std::shared_ptr<triton::client::InferInput> input, uint8_t* data_raw, std::size_t data_raw_size)
      : input{input}, data{}, data_raw{data_raw}, device_data_raw{nullptr}, data_raw_size{data_raw_size} {}
  InputData(std::shared_ptr<triton::client::InferInput> input, uint8_t* data_raw, uint8_t* device_data_raw,
            std::size_t data_raw_size)
      : input{input}, data{}, data_raw{data_raw}, device_data_raw{device_data_raw}, data_raw_size{data_raw_size} {}

  bool isHostMappable() const { return data_raw != nullptr; }
  bool isDeviceBacked() const { return device_data_raw != nullptr; }
};

struct InputOutputMetaData {
  const std::vector<int64_t> shape;
  const inference::DataType datatype;
  const int64_t bytesize;

 public:
  InputOutputMetaData(const std::vector<int64_t>& shape, const inference::DataType& datatype)
      : shape{shape},
        datatype{datatype},
        bytesize{accumulate_shape(shape.begin(), shape.end()) *
                 std::visit([](auto&& arg) { return sizeof(arg); }, getZero(datatype))} {}
};

}  // namespace triton_cpp
