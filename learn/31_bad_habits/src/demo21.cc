#include <iostream>
#include <string>

void Test() {
  std::string str = "Hello, world! 123 456";
  // the output is unexpected prior of cpp17
  str.replace(str.find("world"), 5, "123") // second exec this replace
      .replace(str.find("123"), 3, "456") // may first calculate str.find("123")
      .replace(str.find("456"), 3, "world");
  std::cout << str << "\n";
}

void Test2() {
  int val{0};
  auto fun1 = [&]() { return val++; };
  auto fun2 = [&]() { return val++; };
  auto fun3 = [&]() { return val++; };

  auto fun = [](int a, int b, int c) {
    std::cout << a << "\n";
    std::cout << b << "\n";
    std::cout << c << "\n";
  };
  
  // even for cpp20, the function exec order is unexpected
  fun(fun1(),fun2(),fun3());
}

int main() {
  Test();
  Test2();
  return 0;
}
