cmake_minimum_required(VERSION 3.0)

project(cpp_test)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-elide-constructors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-elide-constructors")


include_directories(include)

add_subdirectory(std_lib)
add_subdirectory(learn)
add_subdirectory(algorithm)

