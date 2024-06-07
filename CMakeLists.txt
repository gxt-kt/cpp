cmake_minimum_required(VERSION 3.0)

project(cpp_test)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-elide-constructors")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-elide-constructors")
set(CMAKE_C_COMPILER "/opt/homebrew/bin/aarch64-apple-darwin23-gcc-14") # gcc clang` 指定c编译器
set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/aarch64-apple-darwin23-g++-14") # g++ clang++` 指定c++编译器


include_directories(include)

add_subdirectory(std_lib)
add_subdirectory(learn)
add_subdirectory(algorithm)

