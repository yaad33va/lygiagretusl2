cmake_minimum_required(VERSION 4.1)
project(lygiagretusl2)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)
set(JSON_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${JSON_INCLUDE_DIR})

set(JSON_URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp)
set(JSON_FILE ${JSON_INCLUDE_DIR}/json.hpp)

if (NOT EXISTS ${JSON_FILE})
    message(STATUS "Downloading nlohmann/json single header...")
    file(DOWNLOAD ${JSON_URL} ${JSON_FILE} SHOW_PROGRESS)
endif()

include_directories(${JSON_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_PATH})

add_executable(lygiagretusl2
        main.cpp
        main.h)

target_link_libraries(lygiagretusl2 PRIVATE ${MPI_LIBRARIES})

set_target_properties(lygiagretusl2 PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}"
        LINK_FLAGS "${MPI_LINK_FLAGS}")