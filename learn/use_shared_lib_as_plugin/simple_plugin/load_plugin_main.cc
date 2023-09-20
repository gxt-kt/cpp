#include <dlfcn.h>

#include <iostream>

int main(int argc, char* argv[]) {
  // 动态库的路径和名称
  const char* libraryPath = "./libplugin.so";

  // 加载动态库
  void* libraryHandle = dlopen(libraryPath, RTLD_LAZY);
  if (!libraryHandle) {
    std::cerr << "Failed to load the dynamic library: " << dlerror()
              << std::endl;
    return 1;
  }

  // 查找要执行的函数
  using HelloFunction = void (*)();
  HelloFunction hello = (HelloFunction)dlsym(libraryHandle, "HelloDemo");
  if (!hello) {
    std::cerr << "Failed to find the function: " << dlerror() << std::endl;
    dlclose(libraryHandle);
    return 1;
  }

  // 调用函数
  hello();

  // 查找要执行的函数
  using AddFunction = int (*)(int, int);
  AddFunction add = (AddFunction)dlsym(libraryHandle, "Add");
  if (!add) {
    std::cerr << "Failed to find the function: " << dlerror() << std::endl;
    dlclose(libraryHandle);
    return 1;
  }

  std::cout << "add(10,20)=" << add(10, 20) << std::endl;

  // 卸载动态库
  dlclose(libraryHandle);

  return 0;
}
