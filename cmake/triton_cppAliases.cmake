# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

if(TARGET triton_cpp::triton_cpp AND NOT TARGET triton_cpp)
  add_library(triton_cpp ALIAS triton_cpp::triton_cpp)
elseif(TARGET triton_cpp AND NOT TARGET triton_cpp::triton_cpp)
  add_library(triton_cpp::triton_cpp ALIAS triton_cpp)
endif()
