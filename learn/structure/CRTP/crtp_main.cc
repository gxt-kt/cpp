#include <cstdlib>

#include "crtp.hpp"

int main(int argc, char* argv[]) {
  Derived1 derived1;
  Derived2 derived2;

  auto fun1 = [](Base<Derived1>* base) { base->interface(); };
  auto fun2 = [](Base<Derived1>& base) { base.interface(); };

  fun1(&derived1);
  fun2(derived1);

  auto fun3 = [](Base<Derived2>* base) { base->interface(); };
  auto fun4 = [](Base<Derived2>& base) { base.interface(); };

  fun3(&derived2);
  fun4(derived2);

  return 0;
}
