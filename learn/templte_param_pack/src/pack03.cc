#include "common.h"

template <typename... Args>
int Foo(Args... t) {
  gDebug(__PRETTY_FUNCTION__);
  return 0;
}

template <typename... Args>
void Goo(Args... args) {}

template <typename... Args>
struct Test {
  Test(Args... args) {
    // same as Goo(Foo(arg0,arg1,arg2)+arg0,Foo(arg0,arg1,arg2)+arg1,Foo(arg0,arg1,arg2)+arg2);
    Goo(Foo(args...) + args...);
  }
};

int main() { 
  Test test(1, 2, 3);
}
