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

# Placeholder de la Fase 1 (se añade cuando exista el archivo):
#   bench/probe1_gemv.cpp   -> tres layouts x M=1,8
#   tests/test_doublemap.cpp-> aserción: los dos mapas de memfd aliasan
