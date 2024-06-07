return()

cmake_minimum_required(VERSION 3.0)

project(pybindtest)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

find_package(pybind11 REQUIRED)

# 这里的模块名需要和 PYBIND11_MODULE中参数pybind11_test的名字一样
# 这句话会生成动态库
pybind11_add_module(pybind11_test src/libs.cc)

# 指定库的输出路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set_target_properties(pybind11_test PROPERTIES  LIBRARY_OUTPUT_PATH LIBRARY_OUTPUT_PATH)
