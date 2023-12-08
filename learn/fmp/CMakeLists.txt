set(CMAKE_CXX_STANDARD 17)

add_executable(fpm fpm_main.cc)


find_package(Ceres REQUIRED)

add_executable(fpm_ceres fpm_ceres_main.cc)
target_include_directories(fpm_ceres PUBLIC ${CERES_INCLUDE_DIRS})
target_link_libraries(fpm_ceres PUBLIC ${CERES_LIBRARIES})

add_executable(my_fmp my_fmp_main.cc)

# find_package(Boost REQUIRED)  
# include_directories(${Boost_INCLUDE_DIRS})  

add_executable(fix_point fix_point_main.cc)
# target_link_libraries(fix_point ${Boost_LIBRARIES})  

# add_executable(fpm_matrix fpm_matrix_main.cc)

# include eigen lib
include_directories("/usr/include/eigen3/")  

# include fpm lib
# https://github.com/MikeLankamp/fpm/tree/master
include_directories(fpm)
