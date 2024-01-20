set(CMAKE_CXX_STANDARD 17)

# include eigen lib
include_directories("/usr/include/eigen3/")  

add_executable(rotation_convert rotation_convert_test.cc)

