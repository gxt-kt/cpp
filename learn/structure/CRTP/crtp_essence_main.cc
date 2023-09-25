#include <cstdlib>

#include "crtp_essence.hpp"

struct A {
  int a = 1;
};
class B : public A {
  int b = 2;
};

int main(int argc, char* argv[]) {
  Derived derived;
  gDebug(&derived);

  auto fun1 = [](Base<Derived>* base) { base->interface(); };
  fun1(&derived);

  // Not important: same like fun1
  // auto fun2 = [](Base<Derived>& base) { base.interface(); };
  // fun2(derived);
  
  

  return 0;
}
