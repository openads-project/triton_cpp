#pragma once

#include <optional>
#include <variant>

#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include <common.h>
#include <grpc_client.h>
#include <grpc_service.pb.h>
#include <ipc.h>
#include <model_config.pb.h>

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
   * @param add_batch_dim Whether to add a batch dimension to the input tensors. SavedModels seem to need this, ONNX don't.
   */
  TritonInterface(const std::string& model_name, const std::string& model_version, const std::string& server_url,
                  bool shm, bool add_batch_dim = true)
      : options_{model_name}, shm_{shm} {
    options_.model_version_ = model_version;
    options_.client_timeout_ = 0;
    triton::client::InferenceServerGrpcClient::Create(&triton_client_, server_url, false);
    inference::ModelConfigResponse model_config;
    auto err = triton_client_->ModelConfig(&model_config, model_name, model_version);
    if (!err.IsOk()) {
      throw std::runtime_error("Failed to get model config from Triton server: " + err.Message());
    }

    std::tie(input_metadata_, output_metadata_) = build_model_info(model_config, add_batch_dim);
  }

  // Rule of 5: Disable copy and move operations
  TritonInterface(const TritonInterface&) = delete;
  TritonInterface& operator=(const TritonInterface&) = delete;
  TritonInterface(TritonInterface&&) = delete;
  TritonInterface& operator=(TritonInterface&&) = delete;
  ~TritonInterface() {
    if (shm_) {
      triton_client_->UnregisterSystemSharedMemory(INPUT_SHM_NAME);
      triton_client_->UnregisterSystemSharedMemory(OUTPUT_SHM_NAME);
    }
  };

  /**
   * @brief Creates all input and output buffers for the model, based on the model metadata
   * 
   * @param special_output_shapes Some models don't know their output shape, i.e. it is given as -1. In this case, you must provide the correct shape here.
   */
  void initInOutputs(std::optional<std::map<std::string, std::vector<int64_t>>> special_output_shapes) {
    inputs_.clear();
    outputs_.clear();
    if (special_output_shapes.has_value()) {
      for (const auto& [name, shape] : special_output_shapes.value()) {
        auto it = output_metadata_.find(name);
        if (it == output_metadata_.end()) {
          throw std::invalid_argument("Output name not found in model metadata: " + name);
        }
        output_metadata_.erase(it);
        output_metadata_.emplace(name, InputOutputMetaData{shape, it->second.datatype});
      }
    }
    if (shm_) {
      triton_client_->UnregisterSystemSharedMemory();
      triton_client_->UnregisterCudaSharedMemory();
      setup_shm_inputs(input_metadata_);
      setup_shm_outputs(output_metadata_);
    } else {
      setup_standard_inputs(input_metadata_);
      setup_standard_outputs(output_metadata_);
    }

    // Also store the raw pointers to in/outputs in vectors, as required by the Triton client
    // We don't have to delete these anywhere, as they are managed by the respective shared pointers
    raw_inputs_.clear();
    for (const auto& input : inputs_) {
      raw_inputs_.push_back(input.second.input.get());
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
    auto& input = inputs_[name];
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
    auto& input = inputs_[name];
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
    auto& input = inputs_[name];
    if (((dim0 * dim1 * dim2) * ... * dims) * sizeof(T) != input.data_raw_size) {
      std::stringstream ss;
      ss << name << ": dims: " << dim0 << ", " << dim1 << ", " << dim2;
      // Print the further dims, separated by commas
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
   * @return std::vector<uint8_t>& raw buffer
   */
  std::pair<uint8_t*, std::size_t> getInputTensor(const std::string& name) {
    auto& input = inputs_[name];
    return {input.data_raw, input.data_raw_size};
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

 private:
  std::pair<std::map<std::string, InputOutputMetaData>, std::map<std::string, InputOutputMetaData>> build_model_info(
      const inference::ModelConfigResponse& model_config, bool add_batch_dim) {
    std::pair<std::map<std::string, InputOutputMetaData>, std::map<std::string, InputOutputMetaData>> metadata;
    std::stringstream model_info_builder;
    int n_inputs = model_config.config().input_size();
    for (int i{0}; i < n_inputs; ++i) {
      auto input = model_config.config().input(i);
      std::vector<int64_t> shape{};
      if (add_batch_dim) {
        shape.push_back(1);  // start with batch dimension. Triton deliberately does not output this, but
                             // requires it to be configured in the InferInput
      }

      for (int j{0}; j < input.dims_size(); ++j) {
        shape.push_back(input.dims(j));
      }
      metadata.first.emplace(input.name(), InputOutputMetaData{shape, input.data_type()});
      model_info_builder << "input name: " << input.name() << '\n';
      model_info_builder << "input datatype: " << inference::DataType_Name(input.data_type()) << '\n';
      model_info_builder << "input dims: " << (input.dims_size() + add_batch_dim) << ", shape: " << shape << "\n\n";
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
    int64_t offset{0};
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
    int64_t offset{0};
    for (const auto& [name, meta] : metadata) {
      std::size_t current_shm_size = meta.bytesize;
      triton::client::InferRequestedOutput* output_ptr{nullptr};
      triton::client::InferRequestedOutput::Create(&output_ptr, name);
      outputs_[name] = (std::shared_ptr<triton::client::InferRequestedOutput>(output_ptr));
      outputs_[name]->SetSharedMemory(OUTPUT_SHM_NAME, current_shm_size, current_output_shm - output_shm_begin);
      current_output_shm += current_shm_size;
    }
  }

  triton::client::InferOptions options_;  // Options for communication with the server
  bool shm_;                              // Whether to use shared memory
  std::unique_ptr<triton::client::InferenceServerGrpcClient> triton_client_;  // Raw triton client
  std::map<std::string, InputOutputMetaData> input_metadata_;                 // Metadata for inputs
  std::map<std::string, InputOutputMetaData> output_metadata_;                // Metadata for outputs
  std::unique_ptr<SharedMemoryRegion> input_shm_;                             // Shared memory for inputs
  std::unique_ptr<SharedMemoryRegion> output_shm_;                            // Shared memory for outputs
  std::string model_info_;                                                    // Human-readable model info
  std::map<std::string, InputData> inputs_;               // Input buffers, for convenience accessible by their name
  ModelOutput outputs_;                                   // Requested outputs
  std::shared_ptr<triton::client::InferResult> results_;  // Output buffers
  std::vector<triton::client::InferInput*> raw_inputs_;   // Raw pointers to inputs
  std::vector<const triton::client::InferRequestedOutput*> raw_outputs_;  // Raw pointers to outputs

  const std::string INPUT_SHM_NAME = "input_data";
  const std::string INPUT_SHM_KEY = "/input_simple";
  const std::string OUTPUT_SHM_NAME = "output_data";
  const std::string OUTPUT_SHM_KEY = "/output_simple";
};

}  // namespace triton_cpp