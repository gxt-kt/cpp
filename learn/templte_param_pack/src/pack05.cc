#include "common.h"

template <template <typename...> class... Args>
struct A : Args<int, double>... {};

template <typename... Args>
struct B {
  void Fun() { gDebug(__PRETTY_FUNCTION__); }
};

template <typename... Args>
struct C {
  void Fun() { gDebug(__PRETTY_FUNCTION__); }
};

int main() {
  A<B, C> a;
  a.B::Fun();
  a.C::Fun();
}
