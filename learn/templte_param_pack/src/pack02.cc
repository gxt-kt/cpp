#include "common.h"

template <typename T, typename U>
T Foo(T t, U u) {
  gDebug(__PRETTY_FUNCTION__);
  return t;
}
template <typename... Args>
void Goo(Args... args) {}

template <typename... Args>
struct Test {
  Test(Args... args) {
    // same as Goo(Foo(&arg0,args0),Foo(&arg1,args1));
    Goo(Foo(&args, args)...);
  }
};

int main() {
  // c++17 class can auto deduce template
  Test(1, 2.0f);
}
