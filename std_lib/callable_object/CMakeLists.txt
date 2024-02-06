# must need c++17
set(CMAKE_CXX_STANDARD 17)
add_executable(std_function std_function_main.cc)
add_executable(std_bind std_bind_main.cc)
add_executable(std_invoke std_invoke_main.cc)
add_executable(std_ref std_ref_main.cc)

add_executable(my_function my_function_main.cc)
add_executable(my_bind my_bind_main.cc)
add_executable(my_invoke my_invoke_main.cc)
add_executable(my_ref my_ref_main.cc)
