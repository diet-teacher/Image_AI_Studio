# Independent target. Built and tested only because cpp/aoti_probe
# reported SUPPORTED on this machine (see docs/phase0_results.md). No
# TorchScript fallback exists anywhere in this target.
find_package(Torch REQUIRED)

add_executable(run_aoti src/main.cpp)

target_link_libraries(run_aoti PRIVATE phase0_common "${TORCH_LIBRARIES}")
target_compile_features(run_aoti PUBLIC cxx_std_17)

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET run_aoti POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:run_aoti>
    )
endif()
