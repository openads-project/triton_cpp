# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(triton_cpp_package_test LANGUAGES CXX)

find_package(triton_cpp 1.1 CONFIG REQUIRED)

add_executable(triton_cpp_bare_target_test main.cpp)
target_link_libraries(triton_cpp_bare_target_test PRIVATE triton_cpp)

add_executable(triton_cpp_namespaced_target_test main.cpp)
target_link_libraries(triton_cpp_namespaced_target_test PRIVATE triton_cpp::triton_cpp)
