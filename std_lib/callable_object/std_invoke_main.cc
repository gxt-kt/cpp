#include "callable_objects.hpp"

int main(int argc, char *argv[]) {
  gDebug() << "exec" << __FILE__;
  { std::invoke(GXT_NAMESPACE::Foo_1, 1); }
  {
    std::invoke(GXT_NAMESPACE::Foo_2_1(), 2);

    std::invoke(GXT_NAMESPACE::Foo_2_2, 2);
  }
  { std::invoke(GXT_NAMESPACE::Foo_3(), 3); }
  {
    GXT_NAMESPACE::Foo_4 foo;
    // // Note that the fourth form is special
    std::invoke(&GXT_NAMESPACE::Foo_4::Foo, foo, 4);
  }
  return 0;
}
