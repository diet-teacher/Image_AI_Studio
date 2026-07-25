# Independent AOTInductor C++ runtime capability probe. Deliberately
# minimal: its only job is to answer HEADER_NOT_FOUND /
# LIBRARY_OR_SYMBOL_NOT_FOUND / COMPILE_FAILED / LINK_FAILED /
# PACKAGE_LOAD_FAILED / SUPPORTED for the currently configured LibTorch.
# A failure here must not affect cpp/torchscript_runner in any way.
find_package(Torch REQUIRED)

add_executable(probe_aoti src/main.cpp)

target_link_libraries(probe_aoti PRIVATE phase0_common "${TORCH_LIBRARIES}")
target_compile_features(probe_aoti PUBLIC cxx_std_17)

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET probe_aoti POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:probe_aoti>
    )
endif()
