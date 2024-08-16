# 最低版本要求
cmake_minimum_required(VERSION 3.00)
# 项目信息
project(demo02 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)  # 用于指定 C++ 编译器应该使用的 C++ 标准版本
set(CMAKE_CXX_STANDARD_REQUIRED ON) # 表明如果找不到指定版本的 C++ 编译器，将发出错误
set(CMAKE_CUDA_STANDARD 14) # 用于指定CUDA编译器应该使用的CUDA C++标准的版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    # 表明如果找不到指定版本的CUDA编译器，将发出错误

set(CMAKE_BUILD_TYPE Release) # Debug Release RelWithDebInfo 设置编译debug

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG="-G -g -O0")
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_60 --ptxas-options=-v")
# elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
# elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
endif()

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
message(FATAL_ERROR "Cannot find CUDA")
endif()

include_directories(include)

add_executable(global_memory src/global_memory.cu)
target_include_directories(global_memory PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(global_memory PRIVATE ${CUDA_LIBRARIES})

add_executable(static_share_memory src/static_share_memory.cu)
target_include_directories(static_share_memory PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(static_share_memory PRIVATE ${CUDA_LIBRARIES})

add_executable(dynamic_share_memory src/dynamic_share_memory.cu)
target_include_directories(dynamic_share_memory PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(dynamic_share_memory PRIVATE ${CUDA_LIBRARIES})

add_executable(constant_memory src/constant_memory.cu)
target_include_directories(constant_memory PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(constant_memory PRIVATE ${CUDA_LIBRARIES})

add_executable(cache src/cache.cu)
target_include_directories(cache PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cache PRIVATE ${CUDA_LIBRARIES})

add_executable(source_print src/source_print.cu)
target_include_directories(source_print PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(source_print PRIVATE ${CUDA_LIBRARIES})
