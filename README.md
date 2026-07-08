# triton_cpp

<p align="center">
  <a href="https://github.com/openads-project"><img src="https://img.shields.io/badge/OpenADS-f5ff01"/></a>
  <a href="https://www.ros.org"><img src="https://img.shields.io/badge/ROS 2-jazzy-22314e"/></a>
  <a href="https://github.com/openads-project/triton_cpp/releases/latest"><img src="https://img.shields.io/github/v/release/openads-project/triton_cpp"/></a>
  <a href="https://github.com/openads-project/triton_cpp/blob/main/LICENSE"><img src="https://img.shields.io/github/license/openads-project/triton_cpp"/></a>
  <br>
  <a href="https://github.com/openads-project/triton_cpp/actions/workflows/ci.yml"><img src="https://github.com/openads-project/triton_cpp/actions/workflows/ci.yml/badge.svg"/></a>
  <a href="https://openads-project.github.io/triton_cpp"><img src="https://github.com/openads-project/triton_cpp/actions/workflows/docs.yml/badge.svg"/></a>
</p>

This repository provides a header-only C++ wrapper around the [official Triton C++ client libraries](https://github.com/triton-inference-server/client) for making interaction with a [Triton Inference Server](https://github.com/triton-inference-server/server) easier in ROS 2 and standalone CMake projects.

The wrapper does not host neural networks itself. A [Triton Inference Server](https://github.com/triton-inference-server/server) with a compatible exported model repository must be available at runtime.

<p align="center">
  <strong>🚀 <a href="#-quick-start">Quick Start</a></strong> • <strong>💻 <a href="#-development">Development</a></strong> • <strong>📝 <a href="#-documentation">Documentation</a></strong>
</p>

> [!IMPORTANT]
> This repository is part of [***OpenADS***](https://github.com/openads-project), the *Open Automated Driving Systems* project. *OpenADS* and its modules have been initiated and are currently being maintained by the [**Institute for Automotive Engineering (ika) at RWTH Aachen University**](https://www.ika.rwth-aachen.de/de/).

## 🚀 Quick Start

### Requirements

- CMake 3.18 or newer
- A C++17 compiler
- Eigen3
- NVIDIA Triton C++ client libraries
- Optional: CUDA Toolkit for CUDA input shared memory

Set `TRITON_CLIENT_DIR` to the Triton client installation prefix when it is not
installed at `/opt/tritonclient`.

### Installation

Clone and install the package:

```sh
git clone https://github.com/openads-project/triton_cpp.git
cmake -S triton_cpp -B build/triton_cpp -DROS_VERSION=0
cmake --build build/triton_cpp
cmake --install build/triton_cpp --prefix /path/to/prefix
```

Installed CMake consumers can use either the original target or the additive
namespaced target:

```cmake
find_package(triton_cpp 1.1 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE triton_cpp::triton_cpp)
# Existing `target_link_libraries(my_target PRIVATE triton_cpp)` remains supported.
```

The existing subdirectory workflow also remains supported:

```cmake
add_subdirectory(triton_cpp)
target_link_libraries(my_target PRIVATE triton_cpp)
```

For ROS 2 workspaces, add this repository to the workspace and declare:

```xml
<depend>triton_cpp</depend>
```

```cmake
find_package(triton_cpp REQUIRED)
ament_target_dependencies(${TARGET_NAME} triton_cpp)
```

### Usage Example

```c++
// Include
#include <triton_cpp/triton_interface.hpp>
#include <memory>
#include <iostream>

// Connect to the server
std::unique_ptr<triton_cpp::TritonInterface> ti =
    std::make_unique<triton_cpp::TritonInterface>("PBOD", "1", "127.0.0.1:8001", false);

// Query for model info (input & output names, shapes, and datatypes) as human-readable string
std::cout << ti->getModelInfo() << std::endl;

// Get the number of inputs and outputs programmatically
int n_inputs = ti->nInputs();
int n_outputs = ti->nOutputs();

// Create all data buffers for model input and output
// Note that, if the server doesn't know some output sizes and thus prints them as -1, you need to
// provide them as argument, like {{"reg_logits", {65536l, 7l}}}
ti->initInOutputs({});

// Get an interface to the model inputs. You need to provide the name and shape as arguments
//   and the datatype as template parameter.
// The return type will be:
//    - an Eigen::Map to an Eigen::Vector, if the shape has only one entry
//    - an Eigen::Map to an Eigen::Matrix, if the shape has two entries
//    - an Eigen::TensorMap to an Eigen::Tensor, if the shape has at least three entries
// If this does not correspond to the expected number of bytes, a std::invalid_argument will be thrown. 
// If the number of bytes matches, the raw data will be interpreted as the requested type, no
//   matter if the datatype and shape equal the expected ones. So you can easily perform
//   reshapes, get rid of unnecessary batch dimensions, ...
// You can also get the raw pointer and bytesize by providing no shape, but that is strongly discouraged
auto points_xyz_map = ti->getInputTensor<float>("points_xyz", 200000, 3);

// fill with data (See Eigen documentation)
points_xyz_map(0,0) = 3.5; //...

// Infer
ti->infer();

// Get the results. The types and shapes work exactly as for input
auto class_logits = ti->getOutputTensor<float>("class_logits", 65536, 3);
std::cout << class_logits(0,0);

// Note that the Infer call takes the inputs that are currently stored inside the data buffers, and the
// output will be valid as long as no other inference is called. This means, that multi-threading
// is currently not supported!
```

## 💻 Development

### Constructor parameters

The full constructor signature is:

```c++
TritonInterface(const std::string& model_name,
                const std::string& model_version,
                const std::string& server_url,
                bool shm,
                bool variable_input_size = false,
                bool retry_connection   = false,
                double client_timeout_s = 0.0,
                bool cuda_input_shm     = false)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `string` | — | Name of the model to load from the Triton server |
| `model_version` | `string` | — | Version of the model (e.g. `"1"`) |
| `server_url` | `string` | — | gRPC URL of the Triton server (e.g. `"127.0.0.1:8001"`) |
| `shm` | `bool` | — | Use host/system shared memory for Triton input and output transport (see [Use host shared memory](#use-host-shared-memory)) |
| `variable_input_size` | `bool` | `false` | Allow input shapes to change between inference calls (see [Variable input size](#variable-input-size)) |
| `retry_connection` | `bool` | `false` | Retry connecting to the server until it becomes available (see [Retry connection](#retry-connection)) |
| `client_timeout_s` | `double` | `0.0` | Client-side inference timeout in seconds; `0.0` disables it (see [Inference timeout](#inference-timeout)) |
| `cuda_input_shm` | `bool` | `false` | Require Triton CUDA shared memory for input tensors. Construction fails with an informative error if it is unavailable. |

Transport combinations:

- `shm=false`, `cuda_input_shm=false`: standard Triton transport for inputs and outputs
- `shm=true`, `cuda_input_shm=false`: system shared memory for inputs and outputs
- `shm=false`, `cuda_input_shm=true`: CUDA shared memory for inputs, standard Triton transport for outputs
- `shm=true`, `cuda_input_shm=true`: CUDA shared memory for inputs, system shared memory for outputs
- `variable_input_size=true` cannot be combined with either `shm=true` or `cuda_input_shm=true`

### Use host shared memory {#use-host-shared-memory}

By default, triton_cpp provides the input and output as a serialized Protobuf stream to the server.
If the server and client run on the same machine, using shared memory is much more efficient.

1. If both are running locally, you can skip to step 3. If both are running in docker containers, you need to expose the shared memory between them. For this, start the server with the arguments `--ipc=shareable --shm-size=2gb`, or in `docker-compose.yml`
    ```yml
        ipc: shareable
        shm_size: '2gb'
    ```
    (Replace 2gb with a reasonable size for your application and machine)

1. Start the client container with the argument `--ipc=container:triton-triton-server-1`, replacing `triton-triton-server-1` with the name of the container running triton, or in `docker-compose.yml`
    ```yml
        ipc: "service:[triton-server]" # replace triton-server with the service's name that provides shm
    ```

1. Now in the code, change
    ```c++
    std::unique_ptr<triton_cpp::TritonInterface> ti =
        std::make_unique<triton_cpp::TritonInterface>("PBOD", "1", "127.0.0.1:8001", false);
    ```
    to
    ```c++
    std::unique_ptr<triton_cpp::TritonInterface> ti =
        std::make_unique<triton_cpp::TritonInterface>("PBOD", "1", "127.0.0.1:8001", true);
    ```
    and you're done.

1. To test whether SHM is working as expected, go to `/dev/shm` in both containers and observe whether files containing data buffers are created while inference is running.

Shared-memory regions are registered with Triton using per-client unique names, so multiple independent
`TritonInterface` instances can use SHM safely at the same time, even when they point to the same Triton server.

When SHM is enabled, tensor offsets inside each shared region are aligned using `max(type_alignment, 8)` to avoid
misaligned accesses for mixed datatypes (for example `FP32`, `INT64`, and `BOOL`) in the same packed buffer.

### Use CUDA input shared memory

If your input tensors are already on GPU memory, `cuda_input_shm=true` avoids copying them back to host memory before sending them to Triton.

```c++
std::unique_ptr<triton_cpp::TritonInterface> ti =
    std::make_unique<triton_cpp::TritonInterface>(
        "PBOD", "1", "127.0.0.1:8001", false, false, false, 0.0, true);
```

Notes:

- this is for input tensors only; outputs still follow the normal path unless `shm=true`
- if CUDA shared memory is unsupported by the local client, Triton server, or `triton_cpp` build, construction fails with an informative error
- host-mapped `getInputTensor(...)` access is not available for CUDA-backed input buffers
- for CUDA-backed inputs, use:
  - `usesCudaInputSharedMemory()`
  - `getInputTensorDevice(...)`
  - `copyInputTensorToDevice(...)`

Example:

```c++
std::unique_ptr<triton_cpp::TritonInterface> ti =
    std::make_unique<triton_cpp::TritonInterface>(
        "PBOD", "1", "127.0.0.1:8001", false, false, false, 0.0, true);

ti->initInOutputs({});

if (!ti->usesCudaInputSharedMemory()) {
  throw std::runtime_error("CUDA input SHM was requested but is not enabled");
}

// Option 1: use CUDA code to write directly into Triton's CUDA-backed input buffer.
auto [points_device, points_bytes] = ti->getInputTensorDevice("points_xyz");
my_cuda_kernel<<<blocks, threads>>>(points_device, points_bytes);

// Option 2: copy from an existing host buffer into the CUDA-backed input.
std::vector<float> host_points(num_points * 3);
fill_points(host_points);
ti->copyInputTensorToDevice("points_xyz", host_points.data(), host_points.size() * sizeof(float));

ti->infer();
```

`getInputTensorDevice(...)` and `copyInputTensorToDevice(...)` can be combined
in the same integration, depending on whether individual inputs already reside
on the GPU or originate in host memory.

### Variable input size {#variable-input-size}

By default (`variable_input_size = false`) input buffers are allocated once during `initInOutputs()` and reused on every `infer()` call, which is the most efficient mode.

Set `variable_input_size = true` when the number of elements in an input tensor changes between calls. In this mode the Triton client reallocates the input memory on every `infer()` call, so there is a small per-call overhead.

```c++
std::unique_ptr<triton_cpp::TritonInterface> ti =
    std::make_unique<triton_cpp::TritonInterface>(
        "behavior-planning", "1", "127.0.0.1:8001", false, true);
```

> **Note:** `variable_input_size = true` cannot be combined with `shm = true` or `cuda_input_shm = true`. If you try, the `TritonInterface` constructor throws `std::invalid_argument`.

### Retry connection {#retry-connection}

By default (`retry_connection = false`) the constructor throws `std::runtime_error` immediately if it cannot reach the server or fetch the model configuration.

Set `retry_connection = true` to have the constructor keep retrying (with a 1-second delay between attempts) until the server is reachable and the model is loaded. This is useful when the client node may start before the Triton server is fully ready (e.g. inside a Docker Compose stack).

```c++
std::unique_ptr<triton_cpp::TritonInterface> ti =
    std::make_unique<triton_cpp::TritonInterface>(
        "PBOD", "1", "127.0.0.1:8001", false, false, true);
```

### Inference timeout {#inference-timeout}

`TritonInterface` supports an optional client-side inference timeout via the constructor argument
`client_timeout_s`:

```c++
std::unique_ptr<triton_cpp::TritonInterface> ti =
    std::make_unique<triton_cpp::TritonInterface>(
        "PBOD", "1", "127.0.0.1:8001", false, false, false, 2.0);
```

- Unit: seconds
- `client_timeout_s == 0.0`: timeout disabled
- `client_timeout_s > 0.0`: failed/blocked inference requests return with an error after the timeout
- `client_timeout_s < 0.0`: invalid (throws)

## 📝 Documentation

Implementation details are found in the [Source Code Documentation](https://openads-project.github.io/triton_cpp).

## ⚖️ Licensing

The source code in this repository is licensed under Apache-2.0, see [LICENSE](LICENSE). Container images provided by this repository may contain third-party software shipped with their own license terms.

## 🙏 Acknowledgements

Development and maintenance of this repository are supported by the following projects. We acknowledge the funding of the respective institutions.

| Project | Funding Institution | Grant Number |
| --- | --- | --- |
| [AIGGREGATE](https://aiggregate.eu/) | 🇪🇺 European Union | 101202457 |
| [autotech.agil](https://www.autotechagil.de/) | 🇩🇪 Federal Ministry for Research, Technology and Space (BMFTR) | 01IS22088A |
