add_library(plugin SHARED plugin.cc)  # default generate shared_lib name is libplugin.so
# set_target_properties(plugin PROPERTIES OUTPUT_NAME "plugin")  # rename to libplugin.so
add_executable(load_plugin load_plugin_main.cc)
