#pragma once

#include <optional>
#include <variant>

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#if defined(TRITON_CPP_ENABLE_CUDA_SHM) && !defined(TRITON_ENABLE_GPU)
#define TRITON_ENABLE_GPU
#endif

#include <common.h>
#include <grpc_client.h>
#include <grpc_service.pb.h>
#include <ipc.h>
#include <model_config.pb.h>

#include "triton_cpp/cuda_shm.hpp"
#include "triton_cpp/shm.hpp"
#include "triton_cpp/shm_utils.hpp"
#include "triton_cpp/types.hpp"
#include "triton_cpp/utils.hpp"

namespace triton_cpp {

/**
 * @brief Main class of this library, providing an interface to the Triton server
 * 
 */
class TritonInterface {
 public:
  /**
   * @brief Construct a new Triton Interface object
   * 
   * @param model_name Name of the model to use
   * @param model_version Version of the model to use
   * @param server_url URL of the Triton server
   * @param shm Whether to use shared memory for communication with the server
   * @param variable_input_size Whether the input size is variable. If true, Triton will reallocate memory for the input tensors on each call, so false is recommended.
   */
  TritonInterface(const std::string& model_name, const std::string& model_version, const std::string& server_url,
                  bool shm, bool variable_input_size = false, bool retry_connection = false,
                  double client_timeout_s = 0.0, bool cuda_input_shm = false)
      : options_{model_name},
        shm_{shm},
        variable_input_size_{variable_input_size},
        cuda_input_shm_requested_{cuda_input_shm} {
    if ((shm || cuda_input_shm) && variable_input_size) {
      throw std::invalid_argument("Variable input size and shared memory cannot be combined");
    }
    options_.model_version_ = model_version;
    if (client_timeout_s < 0.0) {
      throw std::invalid_argument("client_timeout_s must be >= 0.0");
    }
    options_.client_timeout_ = static_cast<decltype(options_.client_timeout_)>(client_timeout_s * 1e6);
    triton::client::Error err;
    err = triton::client::InferenceServerGrpcClient::Create(&triton_client_, server_url, false);
    while (retry_connection && !err.IsOk()) {
      std::cerr << "Failed to create Triton client: " << err.Message() << ". Retrying..." << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
      err = triton::client::InferenceServerGrpcClient::Create(&triton_client_, server_url, false);
    }
    if (!err.IsOk()) {
      throw std::runtime_error("Failed to create Triton client: " + err.Message());
    }
    inference::ModelConfigResponse model_config;
    err = triton_client_->ModelConfig(&model_config, model_name, model_version);
    while (retry_connection && !err.IsOk()) {
      std::cerr << "Failed to get model config from Triton server: " << err.Message() << ". Retrying..." << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
      err = triton_client_->ModelConfig(&model_config, model_name, model_version);
    }
    if (!err.IsOk()) {
      throw std::runtime_error("Failed to get model config from Triton server: " + err.Message());
    }

    std::tie(input_metadata_, output_metadata_) = build_model_info(model_config);

    if (cuda_input_shm_requested_) {
#if defined(TRITON_CPP_ENABLE_CUDA_SHM)
      inference::ServerMetadataResponse server_metadata;
      const auto metadata_status = triton_client_->ServerMetadata(&server_metadata);
      if (!metadata_status.IsOk()) {
        throw std::runtime_error("CUDA input shared memory was requested, but Triton server metadata could not be "
                                 "queried: " +
                                 metadata_status.Message());
      } else {
        const bool server_supports_cuda_shm =
            std::find(server_metadata.extensions().begin(), server_metadata.extensions().end(), "cuda_shared_memory") !=
            server_metadata.extensions().end();
        std::string reason;
        if (!server_supports_cuda_shm) {
          throw std::runtime_error("CUDA input shared memory was requested, but the Triton server does not advertise "
                                   "cuda_shared_memory support");
        } else if (!LocalCudaSharedMemorySupported(&reason)) {
          throw std::runtime_error("CUDA input shared memory was requested, but local CUDA shared memory is not "
                                   "available: " +
                                   reason);
        } else {
          cuda_input_shm_enabled_ = true;
        }
      }
#else
      throw std::runtime_error("CUDA input shared memory was requested, but triton_cpp was built without CUDA SHM "
                               "support");
#endif
    }
  }

  // Rule of 5: Disable copy and move operations
  TritonInterface(const TritonInterface&) = delete;
  TritonInterface& operator=(const TritonInterface&) = delete;
  TritonInterface(TritonInterface&&) = delete;
  TritonInterface& operator=(TritonInterface&&) = delete;
  ~TritonInterface() {
    releaseSharedMemoryRegistrations();
  };

  /**
   * @brief Creates all input and output buffers for the model, based on the model metadata
   * 
   * @param special_output_shapes Some models don't know their output shape, i.e. it is given as -1. In this case, you must provide the correct shape here.
   * @param special_input_shapes Some models don't know their input shape, i.e. it is given as -1. In this case, you must provide the correct shape here.
   * @throws std::invalid_argument if a provided input or output name is unknown.
   * @throws std::runtime_error if shared-memory setup was requested but initialization fails.
   */
  void initInOutputs(std::optional<std::map<std::string, std::vector<int64_t>>> special_output_shapes,
                     std::optional<std::map<std::string, std::vector<int64_t>>> special_input_shapes = std::nullopt) {
    releaseSharedMemoryRegistrations();
    input_shm_.reset();
#if defined(TRITON_CPP_ENABLE_CUDA_SHM)
    input_cuda_shm_.reset();
#endif
    output_shm_.reset();
    inputs_.clear();
    outputs_.clear();
    if (special_input_shapes.has_value()) {
      for (const auto& [name, shape] : special_input_shapes.value()) {
        auto it = input_metadata_.find(name);
        if (it == input_metadata_.end()) {
          throw std::invalid_argument("Input name not found in model metadata: " + name);
        }
        const auto datatype = it->second.datatype;
        input_metadata_.erase(it);
        input_metadata_.emplace(name, InputOutputMetaData{shape, datatype});
      }
    }

    if (special_output_shapes.has_value()) {
      for (const auto& [name, shape] : special_output_shapes.value()) {
        auto it = output_metadata_.find(name);
        if (it == output_metadata_.end()) {
          throw std::invalid_argument("Output name not found in model metadata: " + name);
        }
        const auto datatype = it->second.datatype;
        output_metadata_.erase(it);
        output_metadata_.emplace(name, InputOutputMetaData{shape, datatype});
      }
    }
    if (!variable_input_size_) {
      if (cuda_input_shm_enabled_) {
        try {
          setup_cuda_shm_inputs(input_metadata_);
        } catch (const std::exception& e) {
          throw std::runtime_error("CUDA input shared memory was requested, but initialization failed: " +
                                   std::string(e.what()));
        }
      } else if (shm_) {
        setup_shm_inputs(input_metadata_);
      } else {
        setup_standard_inputs(input_metadata_);
      }
    }

    if (shm_) {
      setup_shm_outputs(output_metadata_);
    } else {
      setup_standard_outputs(output_metadata_);
    }

    // Also store the raw pointers to in/outputs in vectors, as required by the Triton client
    // We don't have to delete these anywhere, as they are managed by the respective shared pointers
    if (!variable_input_size_) {
      raw_inputs_.clear();
      for (const auto& input : inputs_) {
        raw_inputs_.push_back(input.second.input.get());
      }
    }

    raw_outputs_.clear();
    for (const auto& output : outputs_) {
      raw_outputs_.push_back(output.second.get());
    }
  }

  /**
   * @brief Perform the inference on the Triton server, with the input currently present in the buffers.
   * 
   * The result object will be stored in a shared pointer, which will be valid until the next call to this function.
   * 
   */
  void infer() {
    if (variable_input_size_) {
      raw_inputs_.clear();
      for (const auto& input : inputs_) {
        raw_inputs_.push_back(input.second.input.get());
      }
    }
    triton::client::InferResult* raw_results{nullptr};
    auto status = triton_client_->Infer(&raw_results, options_, raw_inputs_, raw_outputs_);
    if (!status.IsOk()) {
      throw(std::runtime_error("ModelInfer failed: " + status.Message()));
    }
    results_.reset(raw_results);
  };

  /**
   * @brief Get a view to the InputBuffer, interpreted as a Vector
   * 
   * @tparam T Scalar data type
   * @param name name of the input
   * @param rows size of the vector.
   * @throws std::invalid_argument if the size of the input buffer does not match the requested size
   * @return VectorType<T> the requested Eigen::Map
   */
  template <typename T>
  VectorType<T> getInputTensor(const std::string& name, int64_t rows) {
    if (variable_input_size_) {
      CreateInputTensor(name, rows);
    }
    auto& input = inputs_[name];
    if (!input.isHostMappable()) {
      throw std::invalid_argument("Input tensor '" + name + "' is backed by CUDA shared memory and is not host-mappable");
    }
    if (rows * sizeof(T) != input.data_raw_size) {
      std::stringstream ss;
      ss << name << ": rows: " << rows << " sizeof(T): " << sizeof(T) << " input.data_raw_size: " << input.data_raw_size
         << std::endl;
      throw std::invalid_argument("Invalid input tensor size: " + ss.str());
    }
    return VectorType<T>{reinterpret_cast<T*>(input.data_raw), rows};
  }

  /**
   * @brief Get a view to the InputBuffer, interpreted as a Matrix
   * 
   * @tparam T Scalar data type
   * @param name name of the input
   * @param rows  size of the matrix.
   * @param cols  size of the matrix.
   * @throws std::invalid_argument if the size of the input buffer does not match the requested size
   * @return MatrixType<T> the requested Eigen::Map
   */
  template <typename T>
  MatrixType<T> getInputTensor(const std::string& name, int64_t rows, int64_t cols) {
    if (variable_input_size_) {
      CreateInputTensor(name, rows, cols);
    }
    auto& input = inputs_[name];
    if (!input.isHostMappable()) {
      throw std::invalid_argument("Input tensor '" + name + "' is backed by CUDA shared memory and is not host-mappable");
    }
    if (rows * cols * sizeof(T) != input.data_raw_size) {
      std::stringstream ss;
      ss << name << ": rows: " << rows << " cols: " << cols << " sizeof(T): " << sizeof(T)
         << " input.data_raw_size: " << input.data_raw_size << std::endl;
      throw std::invalid_argument("Invalid input tensor size: " + ss.str());
    }
    return MatrixType<T>{reinterpret_cast<T*>(input.data_raw), rows, cols};
  }

  /**
   * @brief Get a view to the InputBuffer, interpreted as a Tensor of rank >=3
   * 
   * @tparam T Scalar data type
   * @param name name of the input
   * @param dim0  size of the tensor.
   * @param dim1  size of the tensor.
   * @param dim2  size of the tensor.
   * @param dims  size of the tensor (arbitrary count).
   * @throws std::invalid_argument if the size of the input buffer does not match the requested size
   * @return TensorType<T> the requested Eigen::TensorMap
   */
  template <typename T, typename... DimType>
  TensorType<T, sizeof...(DimType) + 3> getInputTensor(const std::string& name, int64_t dim0, int64_t dim1,
                                                       int64_t dim2, DimType... dims) {
    if (variable_input_size_) {
      CreateInputTensor(name, dim0, dim1, dim2, dims...);
    }
    auto& input = inputs_[name];
    if (!input.isHostMappable()) {
      throw std::invalid_argument("Input tensor '" + name + "' is backed by CUDA shared memory and is not host-mappable");
    }
    if (((dim0 * dim1 * dim2) * ... * dims) * sizeof(T) != input.data_raw_size) {
      std::stringstream ss;
      // Print the further dims, separated by commas
      ss << name << ": dims: " << dim0 << ", " << dim1 << ", " << dim2;
      ((ss << ", " << dims), ...);
      ss << " sizeof(T): " << sizeof(T) << " input.data_raw_size: " << input.data_raw_size << std::endl;
      throw std::invalid_argument("Invalid input tensor size: " + ss.str());
    }
    return TensorType<T, sizeof...(dims) + 3>{reinterpret_cast<T*>(input.data_raw), dim0, dim1, dim2, dims...};
  }

  /**
   * @brief Get the raw input buffer
   * 
   * @param name name of the input
   * @return std::pair<uint8_t*, std::size_t> Pointer to the host-mappable input buffer and its size in bytes.
   * @throws std::invalid_argument if variable input size is enabled and this input has not been created yet,
   *         or if the input is not host-mappable.
   */
  std::pair<uint8_t*, std::size_t> getInputTensor(const std::string& name) {
    if (variable_input_size_ && inputs_.count(name) == 0) {
      throw(std::invalid_argument("Variable input size is enabled, but no size was provided for input " + name));
    }
    auto& input = inputs_[name];
    if (!input.isHostMappable()) {
      throw std::invalid_argument("Input tensor '" + name + "' is backed by CUDA shared memory and is not host-mappable");
    }
    return {input.data_raw, input.data_raw_size};
  }

  /**
   * @brief Whether input tensors are currently backed by Triton CUDA shared memory.
   *
   * @return true if CUDA shared memory is enabled for inputs, otherwise false.
   */
  bool usesCudaInputSharedMemory() const { return cuda_input_shm_enabled_; }

  /**
   * @brief Get the device pointer for a CUDA shared-memory-backed input tensor.
   *
   * @param name Name of the input tensor.
   * @return std::pair<uint8_t*, std::size_t> Device pointer and tensor size in bytes.
   * @throws std::invalid_argument if the input is not backed by CUDA shared memory.
   */
  std::pair<uint8_t*, std::size_t> getInputTensorDevice(const std::string& name) {
    auto& input = inputs_.at(name);
    if (!input.isDeviceBacked()) {
      throw std::invalid_argument("Input tensor '" + name + "' is not backed by CUDA shared memory");
    }
    return {input.device_data_raw, input.data_raw_size};
  }

  /**
   * @brief Copy host data into a CUDA shared-memory-backed input tensor.
   *
   * @param name Name of the input tensor.
   * @param host_data Pointer to the host buffer to copy from.
   * @param bytes Number of bytes to copy. Must exactly match the input tensor size.
   * @throws std::invalid_argument if the tensor is not CUDA-backed, the byte size does not match,
   *         or triton_cpp was built without CUDA SHM support.
   * @throws std::runtime_error if the underlying CUDA copy fails.
   */
  void copyInputTensorToDevice(const std::string& name, const void* host_data, std::size_t bytes) {
#if defined(TRITON_CPP_ENABLE_CUDA_SHM)
    auto& input = inputs_.at(name);
    if (!input.isDeviceBacked()) {
      throw std::invalid_argument("Input tensor '" + name + "' is not backed by CUDA shared memory");
    }
    if (bytes != input.data_raw_size) {
      throw std::invalid_argument("Input tensor '" + name + "' byte size mismatch for host-to-device copy");
    }
    throw_on_cuda_error(cudaMemcpy(input.device_data_raw, host_data, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
#else
    // Suppress unused-parameter warnings when CUDA SHM support is not compiled in.
    (void)name;
    (void)host_data;
    (void)bytes;
    throw std::invalid_argument("triton_cpp was built without CUDA SHM support");
#endif
  }

  /**
   * @brief Get a view to the OutputBuffer, interpreted as a Vector
   * 
   * @tparam T Scalar data type
   * @param name name of the output
   * @param rows size of the vector.
   * @throws std::invalid_argument if the size of the input buffer does not match the requested size
   * @return VectorType<const T> the requested Eigen::Map
   */
  template <typename T>
  VectorType<const T> getOutputTensor(const std::string& name, int64_t rows) const {
    auto [raw_data_buf, raw_data_size] = getOutputTensor(name);
    if (raw_data_size != rows * sizeof(T)) {
      std::stringstream ss;
      ss << name << ": rows: " << rows << " sizeof(T): " << sizeof(T) << " raw_data_size: " << raw_data_size
         << std::endl;
      throw std::invalid_argument("Invalid output tensor size: " + ss.str());
    }
    return VectorType<const T>{reinterpret_cast<const T*>(raw_data_buf), rows};
  }

  /**
   * @brief Get a view to the OutputBuffer, interpreted as a Matrix
   * 
   * @tparam T Scalar data type
   * @param name name of the output
   * @param rows  size of the matrix.
   * @param cols  size of the matrix.
   * @throws std::invalid_argument if the size of the output buffer does not match the requested size
   * @return MatrixType<const T> the requested Eigen::Map
   */
  template <typename T>
  MatrixType<const T> getOutputTensor(const std::string& name, int64_t rows, int64_t cols) const {
    auto [raw_data_buf, raw_data_size] = getOutputTensor(name);
    if (raw_data_size != rows * cols * sizeof(T)) {
      std::stringstream ss;
      ss << name << ": rows: " << rows << " cols: " << cols << " sizeof(T): " << sizeof(T)
         << " raw_data_size: " << raw_data_size << std::endl;
      throw std::invalid_argument("Invalid output tensor size: " + ss.str());
    }
    return MatrixType<const T>{reinterpret_cast<const T*>(raw_data_buf), rows, cols};
  }

  /**
   * @brief Get a view to the OutputBuffer, interpreted as a Tensor of rank >=3
   * 
   * @tparam T Scalar data type
   * @param name name of the output
   * @param dim0  size of the tensor.
   * @param dim1  size of the tensor.
   * @param dim2  size of the tensor.
   * @param dims  size of the tensor (arbitrary count).
   * @throws std::invalid_argument if the size of the output buffer does not match the requested size
   * @return TensorType<const T> the requested Eigen::TensorMap
   */
  template <typename T, typename... DimType>
  TensorType<const T, sizeof...(DimType) + 3> getOutputTensor(const std::string& name, int64_t dim0, int64_t dim1,
                                                              int64_t dim2, DimType... dims) const {
    auto [raw_data_buf, raw_data_size] = getOutputTensor(name);
    if (raw_data_size != ((dim0 * dim1 * dim2) * ... * dims) * sizeof(T)) {
      std::stringstream ss;
      ss << name << ": dims: " << dim0 << ", " << dim1 << ", " << dim2;
      ((ss << ", " << dims), ...);
      ss << " sizeof(T): " << sizeof(T) << " raw_data_size: " << raw_data_size << std::endl;
      throw std::invalid_argument("Invalid output tensor size: " + ss.str());
    }
    return TensorType<const T, sizeof...(dims) + 3>{reinterpret_cast<const T*>(raw_data_buf), dim0, dim1, dim2,
                                                    dims...};
  }

  /**
   * @brief Get the raw output buffer, as C-array + its size
   * 
   * @param name name of the output
   * @return std::pair<const uint8_t*, std::size_t> 
   */
  std::pair<const uint8_t*, std::size_t> getOutputTensor(const std::string& name) const {
    const uint8_t* raw_data_buf{nullptr};
    std::size_t raw_data_size;
    if (shm_) {
      auto shm = outputs_.at(name);
      std::size_t offset;
      std::string shm_name;
      shm->SharedMemoryInfo(&shm_name, &raw_data_size, &offset);
      raw_data_buf = output_shm_->getAddress() + offset;
    } else {
      results_->RawData(name, &raw_data_buf, &raw_data_size);
    }
    return {raw_data_buf, raw_data_size};
  }

  /**
   * @brief Get a description of the model's in and outputs as human-readable text
   * 
   * @return std::string model info
   */
  std::string getModelInfo() const { return model_info_; }

  /**
   * @brief Get the number of inputs
   * 
   * @return std::size_t 
   */
  std::size_t nInputs() const { return input_metadata_.size(); }

  /**
   * @brief Get the number of outputs
   * 
   * @return std::size_t 
   */
  std::size_t nOutputs() const { return output_metadata_.size(); }

  /**
   * @brief Get the Input Shape of a tensor
   * 
   * @param name name of the input tensor
   * @return std::vector<int64_t> shape of  the input tensor
   * @throws std::invalid_argument if the input name is not found in the model metadata
   */
  std::vector<int64_t> getInputShape(const std::string& name) const {
    auto it = input_metadata_.find(name);
    if (it == input_metadata_.end()) {
      throw std::invalid_argument("Input name not found in model metadata: " + name);
    }
    return it->second.shape;
  }

  /**
   * @brief Get the Output Shape of a tensor
   * 
   * @param name name of the output tensor
   * @return std::vector<int64_t> shape of the output tensor
   * @throws std::invalid_argument if the output name is not found in the model metadata
   */
  std::vector<int64_t> getOutputShape(const std::string& name) const {
    auto it = output_metadata_.find(name);
    if (it == output_metadata_.end()) {
      throw std::invalid_argument("Output name not found in model metadata: " + name);
    }
    return it->second.shape;
  }

 private:
  void releaseSharedMemoryRegistrations() {
    if (triton_client_ == nullptr) {
      return;
    }

    if (cuda_input_shm_enabled_) {
      const auto status = triton_client_->UnregisterCudaSharedMemory(INPUT_SHM_NAME);
      if (!status.IsOk()) {
        std::cerr << "Failed to unregister Triton CUDA shared memory region '" << INPUT_SHM_NAME
                  << "': " << status.Message() << std::endl;
      }
    } else if (shm_) {
      const auto status = triton_client_->UnregisterSystemSharedMemory(INPUT_SHM_NAME);
      if (!status.IsOk()) {
        std::cerr << "Failed to unregister Triton system shared memory region '" << INPUT_SHM_NAME
                  << "': " << status.Message() << std::endl;
      }
    }

    if (shm_) {
      const auto status = triton_client_->UnregisterSystemSharedMemory(OUTPUT_SHM_NAME);
      if (!status.IsOk()) {
        std::cerr << "Failed to unregister Triton system shared memory region '" << OUTPUT_SHM_NAME
                  << "': " << status.Message() << std::endl;
      }
    }
  }

  std::pair<std::map<std::string, InputOutputMetaData>, std::map<std::string, InputOutputMetaData>> build_model_info(
      const inference::ModelConfigResponse& model_config) {
    std::pair<std::map<std::string, InputOutputMetaData>, std::map<std::string, InputOutputMetaData>> metadata;
    std::stringstream model_info_builder;
    int n_inputs = model_config.config().input_size();
    for (int i{0}; i < n_inputs; ++i) {
      auto input = model_config.config().input(i);
      std::vector<int64_t> shape{};
      if (model_config.config().max_batch_size() != 0) {
        // Triton omits the batch dimension from the reported tensor dims, but InferInput expects it.
        shape.push_back(1);
      }

      for (int j{0}; j < input.dims_size(); ++j) {
        shape.push_back(input.dims(j));
      }
      metadata.first.emplace(input.name(), InputOutputMetaData{shape, input.data_type()});
      model_info_builder << "input name: " << input.name() << '\n';
      model_info_builder << "input datatype: " << inference::DataType_Name(input.data_type()) << '\n';
      model_info_builder << "input dims: " << (input.dims_size() + (model_config.config().max_batch_size() != 0))
                         << ", shape: " << shape << "\n\n";
    }
    model_info_builder << "-------------------\n";

    int n_outputs = model_config.config().output_size();
    for (int i{0}; i < n_outputs; ++i) {
      auto output = model_config.config().output(i);
      std::vector<int64_t> shape;
      for (int j{0}; j < output.dims_size(); ++j) {
        shape.push_back(output.dims(j));
      }
      metadata.second.emplace(output.name(), InputOutputMetaData{shape, output.data_type()});
      model_info_builder << "output name: " << output.name() << '\n';
      model_info_builder << "output datatype: " << inference::DataType_Name(output.data_type()) << '\n';
      model_info_builder << "output dims: " << output.dims_size() << ", shape: " << shape << "\n\n";
    }
    model_info_builder << "-------------------\n";
    model_info_ = model_info_builder.str();
    return metadata;
  }

  template <typename... DimType>
  void CreateInputTensor(const std::string& name, DimType... dims) {
    if (input_metadata_.find(name) == input_metadata_.end()) {
      throw std::invalid_argument("No input named " + name + " is known.");
    }
    auto& meta = input_metadata_.at(name);
    InputOutputMetaData modified_meta = InputOutputMetaData{std::vector<int64_t>{dims...}, meta.datatype};
    triton::client::InferInput* input_ptr{nullptr};
    triton::client::InferInput::Create(&input_ptr, name, std::vector<int64_t>{dims...},
                                       inference::DataType_Name(modified_meta.datatype).substr(5));
    inputs_[name] = {std::shared_ptr<triton::client::InferInput>(input_ptr),
                     std::vector<uint8_t>(modified_meta.bytesize, 0)};
    inputs_[name].input->AppendRaw(inputs_[name].data.data(), inputs_[name].data.size());
  }

  void setup_standard_inputs(const std::map<std::string, InputOutputMetaData>& metadata) {
    for (const auto& [name, meta] : metadata) {
      triton::client::InferInput* input_ptr{nullptr};
      triton::client::InferInput::Create(&input_ptr, name, meta.shape,
                                         inference::DataType_Name(meta.datatype).substr(5));
      inputs_[name] = {std::shared_ptr<triton::client::InferInput>(input_ptr), std::vector<uint8_t>(meta.bytesize, 0)};
      inputs_[name].input->AppendRaw(inputs_[name].data.data(), inputs_[name].data.size());
    }
  }

  void setup_shm_inputs(const std::map<std::string, InputOutputMetaData>& metadata) {
    auto shm_size = std::accumulate(metadata.begin(), metadata.end(), 0l,
                                    [](int64_t a, const auto& b) { return a + b.second.bytesize; });
    input_shm_ = std::make_unique<SharedMemoryRegion>(INPUT_SHM_KEY, shm_size);
    uint8_t* current_input_shm = input_shm_->getAddress();
    const uint8_t* input_shm_begin = current_input_shm;

    fail_on_error(triton_client_->RegisterSystemSharedMemory(INPUT_SHM_NAME, input_shm_->getKey(), shm_size),
                  "RegisterSystemSharedMemory");
    for (const auto& [name, meta] : metadata) {
      std::size_t current_shm_size = meta.bytesize;
      triton::client::InferInput* input_ptr{nullptr};
      triton::client::InferInput::Create(&input_ptr, name, meta.shape,
                                         inference::DataType_Name(meta.datatype).substr(5));
      inputs_[name] = {std::shared_ptr<triton::client::InferInput>(input_ptr), current_input_shm, current_shm_size};
      inputs_[name].input->SetSharedMemory(INPUT_SHM_NAME, current_shm_size, current_input_shm - input_shm_begin);
      current_input_shm += current_shm_size;
    }
  }

  void setup_cuda_shm_inputs(const std::map<std::string, InputOutputMetaData>& metadata) {
#if defined(TRITON_CPP_ENABLE_CUDA_SHM)
    auto shm_size = std::accumulate(metadata.begin(), metadata.end(), 0l,
                                    [](int64_t a, const auto& b) { return a + b.second.bytesize; });
    input_cuda_shm_ = std::make_unique<CudaSharedMemoryRegion>(INPUT_SHM_NAME, shm_size);
    uint8_t* current_input_shm = input_cuda_shm_->getDeviceAddress();
    const uint8_t* input_shm_begin = current_input_shm;

    fail_on_error(triton_client_->RegisterCudaSharedMemory(INPUT_SHM_NAME, input_cuda_shm_->getIpcHandle(),
                                                           input_cuda_shm_->getDeviceId(), shm_size),
                  "RegisterCudaSharedMemory");

    for (const auto& [name, meta] : metadata) {
      std::size_t current_shm_size = meta.bytesize;
      triton::client::InferInput* input_ptr{nullptr};
      triton::client::InferInput::Create(&input_ptr, name, meta.shape,
                                         inference::DataType_Name(meta.datatype).substr(5));
      inputs_[name] = {std::shared_ptr<triton::client::InferInput>(input_ptr), nullptr, current_input_shm,
                       current_shm_size};
      inputs_[name].input->SetSharedMemory(INPUT_SHM_NAME, current_shm_size, current_input_shm - input_shm_begin);
      current_input_shm += current_shm_size;
    }
#else
    (void)metadata;
    throw std::invalid_argument("triton_cpp was built without CUDA SHM support");
#endif
  }

  void setup_standard_outputs(const std::map<std::string, InputOutputMetaData>& metadata) {
    for (const auto& [name, meta] : metadata) {
      triton::client::InferRequestedOutput* output_ptr{nullptr};
      triton::client::InferRequestedOutput::Create(&output_ptr, name);
      outputs_[name] = (std::shared_ptr<triton::client::InferRequestedOutput>(output_ptr));
    }
  }

  void setup_shm_outputs(const std::map<std::string, InputOutputMetaData>& metadata) {
    auto shm_size = std::accumulate(metadata.begin(), metadata.end(), 0l,
                                    [](int64_t a, const auto& b) { return a + b.second.bytesize; });
    output_shm_ = std::make_unique<SharedMemoryRegion>(OUTPUT_SHM_KEY, shm_size);
    uint8_t* current_output_shm = output_shm_->getAddress();
    const uint8_t* output_shm_begin = current_output_shm;

    fail_on_error(triton_client_->RegisterSystemSharedMemory(OUTPUT_SHM_NAME, output_shm_->getKey(), shm_size),
                  "RegisterSystemSharedMemory");
    for (const auto& [name, meta] : metadata) {
      std::size_t current_shm_size = meta.bytesize;
      triton::client::InferRequestedOutput* output_ptr{nullptr};
      triton::client::InferRequestedOutput::Create(&output_ptr, name);
      outputs_[name] = (std::shared_ptr<triton::client::InferRequestedOutput>(output_ptr));
      outputs_[name]->SetSharedMemory(OUTPUT_SHM_NAME, current_shm_size, current_output_shm - output_shm_begin);
      current_output_shm += current_shm_size;
    }
  }

  triton::client::InferOptions options_;  // Reused Triton inference options for model/version/timeout.
  bool shm_;  // Enables system shared memory for input and output transport.
  bool variable_input_size_;  // Defers input tensor creation until the caller provides shapes.
  bool cuda_input_shm_requested_ = false;  // Records whether CUDA input SHM was explicitly requested.
  bool cuda_input_shm_enabled_ = false;  // True once CUDA input SHM support has been validated and activated.
  std::unique_ptr<triton::client::InferenceServerGrpcClient> triton_client_;  // Underlying Triton gRPC client.
  std::map<std::string, InputOutputMetaData> input_metadata_;  // Declared input tensor shapes and datatypes.
  std::map<std::string, InputOutputMetaData> output_metadata_;  // Declared output tensor shapes and datatypes.
  std::unique_ptr<SharedMemoryRegion> input_shm_;  // Backing system SHM region for host-visible input tensors.
#if defined(TRITON_CPP_ENABLE_CUDA_SHM)
  std::unique_ptr<CudaSharedMemoryRegion> input_cuda_shm_;  // Backing CUDA SHM region for device-resident inputs.
#endif
  std::unique_ptr<SharedMemoryRegion> output_shm_;  // Backing system SHM region for output tensors.
  std::string model_info_;  // Cached human-readable summary of the Triton model interface.
  std::map<std::string, InputData> inputs_;  // Input tensor handles and their backing buffers keyed by name.
  ModelOutput outputs_;  // Requested output handles keyed by tensor name.
  std::shared_ptr<triton::client::InferResult> results_;  // Most recent Triton inference result.
  std::vector<triton::client::InferInput*> raw_inputs_;  // Raw input handles passed to Triton infer calls.
  std::vector<const triton::client::InferRequestedOutput*> raw_outputs_;  // Raw output handles passed to Triton infer calls.

  const std::string RANDOM_INSTANCE_STRING = randstring(10);  // Per-instance suffix to avoid SHM name collisions.
  const std::string INPUT_SHM_NAME = "input_data_" + RANDOM_INSTANCE_STRING;  // Triton-visible input SHM region name.
  const std::string INPUT_SHM_KEY = "/triton_cpp_input_" + RANDOM_INSTANCE_STRING;  // POSIX key for input system SHM.
  const std::string OUTPUT_SHM_NAME = "output_data_" + RANDOM_INSTANCE_STRING;  // Triton-visible output SHM region name.
  const std::string OUTPUT_SHM_KEY = "/triton_cpp_output_" + RANDOM_INSTANCE_STRING;  // POSIX key for output system SHM.
};

}  // namespace triton_cpp
