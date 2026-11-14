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

# --- Binarios de Inferencia Aquifer ---
add_executable(generate_aquifer tools/generate_aquifer.cpp)
target_link_libraries(generate_aquifer PRIVATE aquifer_flags)
