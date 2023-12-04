set(CMAKE_CXX_STANDARD 17)

include_directories(fpm)

include_directories("/usr/include/eigen3/")

add_library(ls_solve vertex.cc edge.cc problem.cc)

add_executable(least_square least_square_main.cc)
target_link_libraries(least_square ls_solve)
