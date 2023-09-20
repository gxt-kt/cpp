#include "callable_objects.hpp"

int main(int argc, char *argv[]) {
  gDebug() << "exec" << __FILE__;
  {
    std::function<void(int)> fun(GXT_NAMESPACE::Foo_1);
    fun(1);
  }
  {
    GXT_NAMESPACE::Foo_2_1 foo;
    std::function<void(int)> fun1(foo);
    fun1(2);

    std::function<void(int)> fun2(GXT_NAMESPACE::Foo_2_2);
    fun2(2);
  }
  {
    GXT_NAMESPACE::Foo_3 foo;
    std::function<void(int)> fun(foo);
    fun(3);
  }
  {
    GXT_NAMESPACE::Foo_4 foo;
    // Note that the fourth form is special
    std::function<void(GXT_NAMESPACE::Foo_4 *, int)> fun(
        &GXT_NAMESPACE::Foo_4::Foo);
    fun(&foo, 4);
  }
  return 0;
}
