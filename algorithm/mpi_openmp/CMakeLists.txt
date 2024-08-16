return()
cmake_minimum_required(VERSION 3.00)
project(mytest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_compile_options(-Ofast -lrt -lm)

set(test_bin_name test_${PROJECT_NAME})

include_directories(src)

set(MPI_SRC
src/common/misc.c src/mpi/utils.c src/mpi/main.c src/common/kernel.c
)
add_executable(m_single src/mpi/solver_mpi.c  ${MPI_SRC})
add_executable(m_multiple src/mpi/solver_HRE_tasks.c  ${MPI_SRC})
target_link_libraries(m_single ${MPI_LIBRARIES})
target_link_libraries(m_multiple ${MPI_LIBRARIES})
# add_executable(${test_bin_name} test/main.c ${DIR_TEST_ARMA} ${DIR_PARSER} ${DIR_SOURCE_ARMA} ${DIR_DETAIL_LAYER} ${DIR_ABSTRACT_LAYER})
# target_link_libraries(${test_bin_name} ${link_lib} ${OpenCV_LIBS} ${link_math_lib} OpenMP::OpenMP_CXX)

# target_include_directories(${test_bin_name} PUBLIC ${glog_INCLUDE_DIR})
# target_include_directories(${test_bin_name} PUBLIC ${GTest_INCLUDE_DIR})
# target_include_directories(${test_bin_name} PUBLIC ${Armadillo_INCLUDE_DIR})
# target_include_directories(${test_bin_name} PUBLIC ./include)
#
# enable_testing()

