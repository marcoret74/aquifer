cmake_minimum_required(VERSION 3.25)

project(aquifer
        VERSION 0.1.0
        DESCRIPTION "Geometric Tensor Paging Runtime & Hypercube Gatekeeper"
        LANGUAGES CXX)

# --- C++23 estricto -----------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-march=native -O3 -Wall -Wextra -Wpedantic)

find_package(Threads REQUIRED)

add_library(aquifer_flags INTERFACE)
target_include_directories(aquifer_flags INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(aquifer_flags INTERFACE Threads::Threads)

# --- Tests de Arquitectura HiperRAM ---
enable_testing()

add_executable(test_morton tests/test_morton.cpp)
target_link_libraries(test_morton PRIVATE aquifer_flags)
add_test(NAME test_morton COMMAND test_morton)

add_executable(test_portero tests/test_portero.cpp)
target_link_libraries(test_portero PRIVATE aquifer_flags)
add_test(NAME test_portero COMMAND test_portero)

add_executable(test_paging_rss tests/test_paging_rss.cpp)
target_link_libraries(test_paging_rss PRIVATE aquifer_flags)
add_test(NAME test_paging_rss COMMAND test_paging_rss)

add_executable(test_sse_affine tests/test_sse_affine.cpp)
target_link_libraries(test_sse_affine PRIVATE aquifer_flags)
add_test(NAME test_sse_affine COMMAND test_sse_affine)

add_executable(test_affine_g32 tests/test_affine_g32.cpp)
target_link_libraries(test_affine_g32 PRIVATE aquifer_flags)
add_test(NAME test_affine_g32 COMMAND test_affine_g32)

add_executable(test_kmin tests/test_kmin.cpp)
target_link_libraries(test_kmin PRIVATE aquifer_flags)
add_test(NAME test_kmin COMMAND test_kmin)

add_executable(test_doublemap tests/test_doublemap.cpp)
target_link_libraries(test_doublemap PRIVATE aquifer_flags)
add_test(NAME test_doublemap COMMAND test_doublemap)

add_executable(check_g32M tests/check_g32M.cpp)
target_link_libraries(check_g32M PRIVATE aquifer_flags)
add_test(NAME check_g32M COMMAND check_g32M)

# --- Binarios de Inferencia Aquifer ---
add_executable(generate_aquifer tools/generate_aquifer.cpp)
target_link_libraries(generate_aquifer PRIVATE aquifer_flags)
