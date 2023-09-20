#include <dlfcn.h>

#include <iostream>
#include <memory>

#include "plugin_base.hpp"

int main(int argc, char* argv[]) {
  // 动态库的路径和名称
  const char* libraryPath = "./libpolymorphism_plugin.so";

  // 加载动态库
  void* libraryHandle = dlopen(libraryPath, RTLD_LAZY);
  if (!libraryHandle) {
    std::cerr << "Failed to load the dynamic library: " << dlerror()
              << std::endl;
    return 1;
  }

  // 查找要执行的函数
  using CreatePluginFunction = PluginBase* (*)();
  CreatePluginFunction create_plugin =
      (CreatePluginFunction)dlsym(libraryHandle, "CreatePlugin");
  if (!create_plugin) {
    std::cerr << "Failed to find the function: " << dlerror() << std::endl;
    dlclose(libraryHandle);
    return 1;
  }

  std::shared_ptr<PluginBase> plugin(create_plugin());

  // Run
  plugin->Init();
  plugin->SetValue(10, 20);
  plugin->Execute();
  std::cout << "plugin->GetResult()=" << plugin->GetResult() << std::endl;
  plugin->Release();

  // NOTE: You must release the lib source before dlclose or may be cause UB.
  plugin.reset();

  // 卸载动态库
  dlclose(libraryHandle);

  return 0;
}
