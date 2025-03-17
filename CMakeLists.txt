cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("metal-matmul" LANGUAGES C CXX OBJCXX)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

# Build metal
file(GLOB METAL_SHADER_FILES "metal/*.metal")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/air_files"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/air_files"
)

set(METAL_AIR_FILES)
foreach(shader_file IN LISTS METAL_SHADER_FILES)
    get_filename_component(shader_name ${shader_file} NAME_WE) # Get filename without extension
    set(_output "${CMAKE_CURRENT_BINARY_DIR}/air_files/${shader_name}.air")
    list(APPEND METAL_AIR_FILES "${_output}")
    add_custom_command(
        OUTPUT ${_output}
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/air_files"
        COMMAND xcrun -sdk macosx metal -c "${shader_file}" -o "${_output}"
    )
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/matmul.metallib
    DEPENDS ${METAL_AIR_FILES}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${CMAKE_CURRENT_BINARY_DIR}/matmul.metallib
    DEPENDS ${METAL_SHADER_FILES}
)
add_custom_target(metallib ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/matmul.metallib)


add_executable(benchmark "cpp/benchmark.cpp")
target_compile_options(benchmark PRIVATE -x objective-c++)
target_link_libraries(
    benchmark
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${MPS_FRAMEWORK}
)
add_dependencies(benchmark metallib)
