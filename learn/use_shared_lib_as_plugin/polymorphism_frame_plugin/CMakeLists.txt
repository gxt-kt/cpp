set(CMAKE_CXX_STANDARD 17)
add_library(polymorphism_plugin SHARED plugin.cc)  # default generate shared_lib name is libplugin.so
# set_target_properties(plugin PROPERTIES OUTPUT_NAME "polymorphism_plugin")  # rename to libplugin.so
add_executable(load_polymorphism_plugin load_plugin_main.cc)
