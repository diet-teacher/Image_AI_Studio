cmake_minimum_required(VERSION 3.18)
project(image_ai_studio_phase0 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows is the primary target but nothing here is MSVC-specific:
# standard C++17 + CMake + LibTorch's own CMake config, so the same
# CMakeLists.txt builds on macOS/Linux too.
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

option(BUILD_TORCHSCRIPT_RUNNER "Build the TorchScript runner" ON)
option(BUILD_AOTI_PROBE "Build the AOTInductor capability probe" ON)
option(BUILD_AOTI_RUNNER "Build the AOTInductor runner" ON)

# Backend-independent; no Torch dependency, always available.
add_subdirectory(cpp/common)

# Each of the following is independently configurable/buildable. A
# failure in one (most likely BUILD_AOTI_RUNNER on a distribution
# without AOTInductor C++ runtime support) must not block the others --
# do not add cross-dependencies between these subdirectories.
if(BUILD_TORCHSCRIPT_RUNNER)
    add_subdirectory(cpp/torchscript_runner)
endif()

if(BUILD_AOTI_PROBE)
    add_subdirectory(cpp/aoti_probe)
endif()

if(BUILD_AOTI_RUNNER)
    add_subdirectory(cpp/aoti_runner)
endif()
