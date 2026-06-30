# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable
        TRITON_CPP_BINARY_DIR
        TRITON_CPP_INSTALL_PREFIX
        TRITON_CPP_SOURCE_DIR)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR "${required_variable} must be defined")
  endif()
endforeach()

set(consumer_build_dir "${TRITON_CPP_BINARY_DIR}/installed-package-test")
file(REMOVE_RECURSE "${TRITON_CPP_INSTALL_PREFIX}" "${consumer_build_dir}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${TRITON_CPP_BINARY_DIR}"
          --prefix "${TRITON_CPP_INSTALL_PREFIX}"
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -S "${TRITON_CPP_SOURCE_DIR}/tests/cmake_package"
          -B "${consumer_build_dir}"
          "-DCMAKE_PREFIX_PATH=${TRITON_CPP_INSTALL_PREFIX}"
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}"
  COMMAND_ERROR_IS_FATAL ANY
)
