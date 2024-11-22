# find dependencies
find_package(Eigen3 REQUIRED)

if(DEFINED ENV{TRITON_CLIENT_DIR})
    set(TRITON_CLIENT_DIR $ENV{TRITON_CLIENT_DIR})
else()
    set(TRITON_CLIENT_DIR /opt/tritonclient)
endif()

set(ARM_TRITON_CLIENT_DIR ${TRITON_CLIENT_DIR}/tritonserver/clients)

# find include directories
find_path(INCLUDE_DIR grpc_client.h HINTS ${TRITON_CLIENT_DIR}/include ${ARM_TRITON_CLIENT_DIR}/include)
list(APPEND INCLUDE_DIRS ${INCLUDE_DIR})

# find libraries
find_library(GrpcLib libgrpcclient.so HINTS ${TRITON_CLIENT_DIR}/lib ${ARM_TRITON_CLIENT_DIR}/lib)
find_library(HttpLib libhttpclient.so HINTS ${TRITON_CLIENT_DIR}/lib ${ARM_TRITON_CLIENT_DIR}/lib)
find_library(ProtobufLib libprotobuf.a HINTS ${TRITON_CLIENT_DIR}/lib ${ARM_TRITON_CLIENT_DIR}/lib)
find_library(ProtobufLiteLib libprotobuf-lite.a HINTS ${TRITON_CLIENT_DIR}/lib ${ARM_TRITON_CLIENT_DIR}/lib)

# handle the QUIETLY and REQUIRED arguments and set *_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Triton DEFAULT_MSG INCLUDE_DIRS GrpcLib HttpLib ProtobufLib ProtobufLiteLib)
mark_as_advanced(INCLUDE_DIRS GrpcLib HttpLib ProtobufLib ProtobufLiteLib)

# set INCLUDE_DIRS and LIBRARIES
if(Triton_FOUND)
    set(Triton_INCLUDE_DIRS ${INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
    set(Triton_LIBRARIES ${GrpcLib} ${HttpLib} ${ProtobufLib} ${ProtobufLiteLib})
endif()
