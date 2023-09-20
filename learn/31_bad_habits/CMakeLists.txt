# cmake_minimum_required(VERSION 3.0)
# project(cpp_bad_habtis)

# must need c++17
set(CMAKE_CXX_STANDARD 17)

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# message(STATUS "This is cpp bad habits project")


file(GLOB_RECURSE SOURCE_FILES "src/*.cc")
foreach(source_file ${SOURCE_FILES})
    get_filename_component(file_name ${source_file} NAME_WE)
    add_executable(${file_name} ${source_file})
endforeach()


