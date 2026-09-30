# libdh_spec: speculative decoding for the rewrite (see dh_spec.cpp). Built by build.py after vendor/llama.cpp,
# against that build's libraries, into the same bin/ directory so it is copied to dh_llama/lib with them.
cmake_minimum_required(VERSION 3.14)
project(dh_spec CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT LLAMA_SRC OR NOT LLAMA_BUILD)
    message(FATAL_ERROR "pass -DLLAMA_SRC=<vendor/llama.cpp> -DLLAMA_BUILD=<its build dir>")
endif()
add_library(dh_spec SHARED dh_spec.cpp dh_tree.cpp)
target_include_directories(dh_spec PRIVATE ${LLAMA_SRC}/include ${LLAMA_SRC}/common ${LLAMA_SRC}/ggml/include ${LLAMA_SRC}/vendor)
set(_dirs ${LLAMA_BUILD}/bin ${LLAMA_BUILD}/bin/Release ${LLAMA_BUILD}/lib ${LLAMA_BUILD}/lib/Release
          ${LLAMA_BUILD}/common ${LLAMA_BUILD}/common/Release ${LLAMA_BUILD}/src ${LLAMA_BUILD}/src/Release
          ${LLAMA_BUILD}/ggml/src ${LLAMA_BUILD}/ggml/src/Release)
foreach(_l llama-common llama ggml ggml-base)
    find_library(_lib_${_l} NAMES ${_l} PATHS ${_dirs} NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(dh_spec PRIVATE ${_lib_${_l}})
endforeach()
set_target_properties(dh_spec PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON
    LIBRARY_OUTPUT_DIRECTORY ${LLAMA_BUILD}/bin RUNTIME_OUTPUT_DIRECTORY ${LLAMA_BUILD}/bin
    BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN" MACOSX_RPATH ON)
