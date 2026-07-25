# Backend-independent utilities only: CLI parsing, tensor bin+json IO,
# result/log writing, timing stats. Must never depend on Torch or
# AOTInductor headers -- both runners link this unconditionally.
add_library(phase0_common STATIC
    src/CommandLine.cpp
    src/TensorFileIO.cpp
    src/ResultWriter.cpp
    src/TimingStats.cpp
)

target_include_directories(phase0_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(phase0_common PUBLIC cxx_std_17)
