return ()

# c version
add_executable(thread_pool_c_main thread_pool_c_main.cc)
target_link_libraries(thread_pool_c_main pthread)

set(CMAKE_CXX_STANDARD 17)
add_executable(thread_pool_cpp_main thread_pool_cpp_main.cc)
target_link_libraries(thread_pool_cpp_main pthread)

add_executable(thread_pool_cpp_advanced_main thread_pool_cpp_advanced_main.cc)
target_link_libraries(thread_pool_cpp_advanced_main pthread)
