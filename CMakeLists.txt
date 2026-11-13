cmake_minimum_required(VERSION 3.25)

project(caudal
        VERSION 0.0.1
        DESCRIPTION "CPU tensor runtime built around the memory wall"
        LANGUAGES CXX)

# --- C++23 estricto -----------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Flags --------------------------------------------------------------------
# CAUDAL_NATIVE=ON compila para ESTA máquina (AVX-512 incluido si existe).
# Para el edge (Atom sin AVX2, Galaxy NEON) usar -DCAUDAL_NATIVE=OFF y añadir
# dispatch por target_clones en los kernels (pendiente, Fase 1).
option(CAUDAL_NATIVE "Compilar con -march=native (binario no portable)" ON)

if(CAUDAL_NATIVE)
    add_compile_options(-march=native)
else()
    add_compile_options(-mavx2 -mfma)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# --- Include ------------------------------------------------------------------
find_package(Threads REQUIRED)

add_library(caudal_flags INTERFACE)
target_include_directories(caudal_flags INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(caudal_flags INTERFACE Threads::Threads)

# --- Probes (mediciones, no aserciones: no van a ctest) -----------------------
add_executable(probe0_bw bench/probe0_bw.cpp)
target_link_libraries(probe0_bw PRIVATE caudal_flags)

add_executable(probe1_gemv bench/probe1_gemv.cpp)
target_link_libraries(probe1_gemv PRIVATE caudal_flags)

add_executable(probe2_group bench/probe2_group.cpp)
target_link_libraries(probe2_group PRIVATE caudal_flags)

# --- Tests (aserciones: van a ctest) ------------------------------------------
enable_testing()

add_executable(test_doublemap tests/test_doublemap.cpp)
target_link_libraries(test_doublemap PRIVATE caudal_flags)
add_test(NAME doublemap COMMAND test_doublemap 64)

add_executable(test_pack_roundtrip tests/test_pack_roundtrip.cpp)
target_link_libraries(test_pack_roundtrip PRIVATE caudal_flags)

find_package(Python3 COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    # El artefacto sintético se genera antes del test, vía fixtures de CTest.
    add_test(NAME gen_artefacto
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tools/pack_synth.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/artefacto)
    set_tests_properties(gen_artefacto PROPERTIES FIXTURES_SETUP artefacto)

    add_test(NAME pack_roundtrip
             COMMAND test_pack_roundtrip ${CMAKE_CURRENT_BINARY_DIR}/artefacto 256 512 4)
    set_tests_properties(pack_roundtrip PROPERTIES FIXTURES_REQUIRED artefacto)

    add_test(NAME gen_artefacto_k3584
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tools/pack_synth.py
                     --N 64 --K 3584 --M 2
                     --out ${CMAKE_CURRENT_BINARY_DIR}/artefacto_k3584)
    set_tests_properties(gen_artefacto_k3584 PROPERTIES FIXTURES_SETUP artefacto_k3584)

    add_test(NAME pack_roundtrip_k3584
             COMMAND test_pack_roundtrip ${CMAKE_CURRENT_BINARY_DIR}/artefacto_k3584 64 3584 2)
    set_tests_properties(pack_roundtrip_k3584 PROPERTIES FIXTURES_REQUIRED artefacto_k3584)
else()
    message(STATUS "Python3 no encontrado: los tests del packer quedan deshabilitados")
endif()

# Pendiente Fase 1: tools/pack_qwen.py (safetensors -> layout A0 real)



add_executable(forward_qwen tools/forward_qwen.cpp)
target_link_libraries(forward_qwen PRIVATE caudal_flags)
