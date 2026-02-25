## TRITON CPP

This package is a header-only wrapper around the [official Triton C++ client libraries](https://github.com/triton-inference-server/client) for making interaction with a [Triton inference server](https://github.com/triton-inference-server/server) in C++ much easier.


[[_TOC_]]

### Installation

Current workflow:

1. Use one of the [rwthika/ros2-triton](https://hub.docker.com/r/rwthika/ros2-triton/) images as base image

1. Add this repo to the .repos file
    ```yml
    repositories:
      triton_cpp:
        type: git
        url: https://gitlab.ika.rwth-aachen.de/fb-fi/ml/triton_cpp.git
        version: main
    ```

1. Add `triton_cpp` to your package.xml and CMakeLists.txt file
    ```xml
      <depend>triton_cpp</depend>
    ```
    and
    ```cmake
    list(APPEND CMAKE_PREFIX_PATH "../../upstream/triton_cpp/cmake") # Current hack needed because the official Triton Lib doesn't export a config.cmake
    find_package(triton_cpp REQUIRED)
    ament_target_dependencies(
      ${TARGET_NAME}
      triton_cpp
    )
    ```

### Quick start guide

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

// Get the number of in/outputs programatically
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

### Constructor parameters

The full constructor signature is:

```c++
TritonInterface(const std::string& model_name,
                const std::string& model_version,
                const std::string& server_url,
                bool shm,
                bool variable_input_size = false,
                bool retry_connection   = false,
                double client_timeout_s = 0.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `string` | — | Name of the model to load from the Triton server |
| `model_version` | `string` | — | Version of the model (e.g. `"1"`) |
| `server_url` | `string` | — | gRPC URL of the Triton server (e.g. `"127.0.0.1:8001"`) |
| `shm` | `bool` | — | Use shared memory for zero-copy data transfer (see [Use shared memory](#use-shared-memory)) |
| `variable_input_size` | `bool` | `false` | Allow input shapes to change between inference calls (see [Variable input size](#variable-input-size)) |
| `retry_connection` | `bool` | `false` | Retry connecting to the server until it becomes available (see [Retry connection](#retry-connection)) |
| `client_timeout_s` | `double` | `0.0` | Client-side inference timeout in seconds; `0.0` disables it (see [Inference timeout](#inference-timeout)) |

### Use shared memory

triton_cpp provides the input and output as serialized Protobuf stream to the server per default.
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

1. To test wether SHM is working as expected, go to `/dev/shm` in both containers and observe, while inference is running, if files with data buffers are created there.

### Variable input size

By default (`variable_input_size = false`) input buffers are allocated once during `initInOutputs()` and reused on every `infer()` call, which is the most efficient mode.

Set `variable_input_size = true` when the number of elements in an input tensor changes between calls (e.g. a variable-length environment model representation, as seen in [behavior-planning](https://gitlab.ika.rwth-aachen.de/fb-fi/its-modules/planning/behavior_planning)). In this mode the Triton client reallocates the input memory on every `infer()` call, so there is a small per-call overhead.

```c++
std::unique_ptr<triton_cpp::TritonInterface> ti =
    std::make_unique<triton_cpp::TritonInterface>(
        "behavior-planning", "1", "127.0.0.1:8001", false, true);
```

> **Note:** `variable_input_size = true` and `shm = true` cannot be combined and will throw `std::invalid_argument`.

### Retry connection

By default (`retry_connection = false`) the constructor throws `std::runtime_error` immediately if it cannot reach the server or fetch the model configuration.

Set `retry_connection = true` to have the constructor keep retrying (with a 1-second delay between attempts) until the server is reachable and the model is loaded. This is useful when the client node may start before the Triton server is fully ready (e.g. inside a Docker Compose stack).

```c++
std::unique_ptr<triton_cpp::TritonInterface> ti =
    std::make_unique<triton_cpp::TritonInterface>(
        "PBOD", "1", "127.0.0.1:8001", false, false, true);
```

### Inference timeout

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

### Documentation

TBD. Should be generated with doxygen
