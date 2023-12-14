cmake_minimum_required(VERSION 3.00)
project(ls)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

include_directories(fpm)
include_directories(backend)

include_directories("/usr/include/eigen3/")


# find_package(Sophus REQUIRED)
# find_package(fmt REQUIRED)  
# find_package(fmt REQUIRED)
# include_directories(${fmt_INCLUDE_DIRS})  

# include_directories(${Sophus_INCLUDE_DIRS})  

include_directories(sophus)  

add_subdirectory(backend)

add_executable(least_square least_square_main.cc)
target_link_libraries(least_square ls_solve)
add_executable(least_square2 least_square2_main.cc)
target_link_libraries(least_square2 ls_solve)


add_executable(test_ba test_ba_main.cc edge_reprojection.cc)
# target_link_libraries(test_ba ls_solve ${Sophus_LIBRARIES})
target_link_libraries(test_ba ls_solve)
