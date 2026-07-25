# Independent target: must configure, build, and run even if
# cpp/aoti_probe or cpp/aoti_runner fail to compile or link. No
# AOTInductor headers may be included from this target.
find_package(Torch REQUIRED)

add_executable(run_torchscript src/main.cpp)

target_link_libraries(run_torchscript PRIVATE phase0_common "${TORCH_LIBRARIES}")
target_compile_features(run_torchscript PUBLIC cxx_std_17)

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET run_torchscript POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:run_torchscript>
    )
endif()
