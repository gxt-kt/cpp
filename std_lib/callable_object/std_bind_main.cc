#include "callable_objects.hpp"

using namespace std::placeholders;

int main(int argc, char *argv[]) {
  gDebug() << "exec" << __FILE__;
  {
    auto fun = std::bind(GXT_NAMESPACE::Foo_1, _1);
    fun(1);
  }
  // suggest 
  {
    GXT_NAMESPACE::Foo_2_1 foo;
    auto fun1 = std::bind(GXT_NAMESPACE::Foo_2_1(), _1);
    fun1(2);

    auto fun2 = std::bind(GXT_NAMESPACE::Foo_2_2, _1);
    fun2(2);
  }
  {
    GXT_NAMESPACE::Foo_3 foo;
    auto fun = std::bind(GXT_NAMESPACE::Foo_3(), _1);
    fun(3);
  }
  {
    GXT_NAMESPACE::Foo_4 foo;
    // Note that the fourth form is special
    auto fun = std::bind(&GXT_NAMESPACE::Foo_4::Foo, &foo, _1);
  }
  {
    // std::placeholders examples:
    // The follow example comes from https://subingwen.cn/cpp/bind/
    auto foo = [](int x, int y) { std::cout << x << " " << y << std::endl; };
    std::bind(foo, 1, 2)();     // 1 2
    std::bind(foo, _1, 2)(10);  // 10 2
    std::bind(foo, 2, _1)(10);  // 2 10

    // std::bind(foo, 2, _2)(10); // err: hasn't the second parameter
    std::bind(foo, 2, _2)(10, 20);  // 2 20 // the first value hasn't use

    std::bind(foo, _1, _2)(10, 20);  // 10 20
    std::bind(foo, _2, _1)(10, 20);  // 20 10
  }
  return 0;
}
