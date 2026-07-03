# Copyright Institute for Automotive Engineering (ika), RWTH Aachen University
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/package.xml" TRITON_CPP_VERSION_LINE
     REGEX "<version>[0-9]+\\.[0-9]+\\.[0-9]+</version>")
string(REGEX REPLACE ".*<version>([0-9]+\\.[0-9]+\\.[0-9]+)</version>.*" "\\1"
       TRITON_CPP_VERSION "${TRITON_CPP_VERSION_LINE}")
if(NOT TRITON_CPP_VERSION)
  message(FATAL_ERROR "Unable to read triton_cpp version from package.xml")
endif()

project(triton_cpp VERSION "${TRITON_CPP_VERSION}" LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(NOT DEFINED ROS_VERSION)
  find_package(ros_environment QUIET)
  if(ros_environment_FOUND)
    set(ROS_VERSION "$ENV{ROS_VERSION}")
  else()
    set(ROS_VERSION 0)
  endif()
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Triton REQUIRED MODULE)
find_package(CUDAToolkit QUIET)

set(TRITON_CPP_CONFIG_WITH_CUDA OFF)

add_library(triton_cpp INTERFACE)
add_library(triton_cpp::triton_cpp ALIAS triton_cpp)

target_compile_features(triton_cpp INTERFACE cxx_std_17)
target_include_directories(triton_cpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(triton_cpp INTERFACE Triton::Client)

if(CUDAToolkit_FOUND)
  set(TRITON_CPP_CONFIG_WITH_CUDA ON)
  target_link_libraries(triton_cpp INTERFACE CUDA::cudart)
  target_compile_definitions(triton_cpp INTERFACE TRITON_CPP_ENABLE_CUDA_SHM TRITON_ENABLE_GPU)
endif()

install(DIRECTORY include/triton_cpp DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS triton_cpp
  EXPORT triton_cppTargets
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(FILES LICENSE README.md
  DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}"
)

# === ROS 2 / AMENT ============================================================
if(ROS_VERSION EQUAL 2)
  find_package(ament_cmake REQUIRED)

  ament_export_targets(triton_cppTargets HAS_LIBRARY_TARGET)
  if(CUDAToolkit_FOUND)
    # Preserve the legacy variables exported to ament consumers.
    ament_export_definitions(TRITON_CPP_ENABLE_CUDA_SHM TRITON_ENABLE_GPU)
  endif()

  install(FILES cmake/FindTriton.cmake
    DESTINATION share/${PROJECT_NAME}/cmake
  )

  if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
  endif()

  ament_package(
    CONFIG_EXTRAS cmake/triton_cppDependencies.cmake.in
    CONFIG_EXTRAS_POST cmake/triton_cppAliases.cmake
  )

# === STANDARD CMAKE ===========================================================
elseif(ROS_VERSION EQUAL 0)
  set(TRITON_CPP_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/triton_cpp"
      CACHE STRING "Install directory for triton_cpp CMake package files")

  configure_package_config_file(
    cmake/triton_cppConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/triton_cppConfig.cmake"
    INSTALL_DESTINATION "${TRITON_CPP_INSTALL_CMAKEDIR}"
  )
  configure_file(
    cmake/triton_cppDependencies.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/triton_cppDependencies.cmake"
    @ONLY
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/triton_cppConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
  )

  install(EXPORT triton_cppTargets
    FILE triton_cppTargets.cmake
    DESTINATION "${TRITON_CPP_INSTALL_CMAKEDIR}"
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/triton_cppConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/triton_cppConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/triton_cppDependencies.cmake"
    cmake/triton_cppAliases.cmake
    cmake/FindTriton.cmake
    DESTINATION "${TRITON_CPP_INSTALL_CMAKEDIR}"
  )
  install(FILES package.xml
    DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}"
  )

  option(BUILD_TESTING "Build triton_cpp tests" OFF)
  include(CTest)
  if(BUILD_TESTING)
    add_test(
      NAME triton_cpp_installed_package
      COMMAND "${CMAKE_COMMAND}"
              "-DTRITON_CPP_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
              "-DTRITON_CPP_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/test-prefix"
              "-DTRITON_CPP_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_installed_package.cmake"
    )
  endif()
else()
  message(FATAL_ERROR "triton_cpp supports ROS 2 or standalone CMake builds; ROS_VERSION=${ROS_VERSION} is unsupported")
endif()
