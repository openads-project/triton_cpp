# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

include(FindPackageHandleStandardArgs)

find_package(Eigen3 REQUIRED CONFIG)

if(DEFINED TRITON_CLIENT_DIR)
  set(_triton_client_root "${TRITON_CLIENT_DIR}")
elseif(DEFINED ENV{TRITON_CLIENT_DIR})
  set(_triton_client_root "$ENV{TRITON_CLIENT_DIR}")
else()
  set(_triton_client_root "/opt/tritonclient")
endif()

set(_triton_client_roots
  "${_triton_client_root}"
  "${_triton_client_root}/tritonserver/clients"
)

find_path(Triton_INCLUDE_DIR
  NAMES grpc_client.h
  HINTS ${_triton_client_roots}
  PATH_SUFFIXES include
)
find_library(Triton_GRPC_CLIENT_LIBRARY
  NAMES grpcclient
  HINTS ${_triton_client_roots}
  PATH_SUFFIXES lib
)
find_library(Triton_HTTP_CLIENT_LIBRARY
  NAMES httpclient
  HINTS ${_triton_client_roots}
  PATH_SUFFIXES lib
)
find_library(Triton_PROTOBUF_LIBRARY
  NAMES protobuf
  HINTS ${_triton_client_roots}
  PATH_SUFFIXES lib
)
find_library(Triton_PROTOBUF_LITE_LIBRARY
  NAMES protobuf-lite
  HINTS ${_triton_client_roots}
  PATH_SUFFIXES lib
)

find_package_handle_standard_args(Triton
  REQUIRED_VARS
    Triton_INCLUDE_DIR
    Triton_GRPC_CLIENT_LIBRARY
    Triton_HTTP_CLIENT_LIBRARY
    Triton_PROTOBUF_LIBRARY
    Triton_PROTOBUF_LITE_LIBRARY
)

if(Triton_FOUND)
  set(Triton_INCLUDE_DIRS "${Triton_INCLUDE_DIR}")
  set(Triton_LIBRARIES
    "${Triton_GRPC_CLIENT_LIBRARY}"
    "${Triton_HTTP_CLIENT_LIBRARY}"
    "${Triton_PROTOBUF_LIBRARY}"
    "${Triton_PROTOBUF_LITE_LIBRARY}"
  )
  set(Triton_TARGETS Triton::Client)

  if(NOT TARGET Triton::Client)
    add_library(Triton::Client INTERFACE IMPORTED)
    set_target_properties(Triton::Client PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Triton_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${Triton_LIBRARIES};Eigen3::Eigen"
    )
  endif()
endif()

mark_as_advanced(
  Triton_INCLUDE_DIR
  Triton_GRPC_CLIENT_LIBRARY
  Triton_HTTP_CLIENT_LIBRARY
  Triton_PROTOBUF_LIBRARY
  Triton_PROTOBUF_LITE_LIBRARY
)

unset(_triton_client_root)
unset(_triton_client_roots)
