#include "common.h"

template <typename T>
struct Base {
  Base() {}
  Base(T t) {}
};

template <typename... Args>
struct Derived : Base<Args>... {
  // use using with template fold to introduce Base construct function
  using Base<Args>::Base...;

  // as following
  #if 0
  // inline Derived() noexcept(false) = default;
  inline Derived(int t) noexcept(false)
  : Base<int>(t)
  , Base<double>()
  , Base<bool>()
  {
  }
  
  inline Derived(double t) noexcept(false)
  : Base<int>()
  , Base<double>(t)
  , Base<bool>()
  {
  }
  
  inline Derived(bool t) noexcept(false)
  : Base<int>()
  , Base<double>()
  , Base<bool>(t)
  {
  }
  #endif
};

int main() {
  Derived<int, double, bool> a;
  Derived<int, double, bool> b(10);
  Derived<int, double, bool> c(10.0);
  Derived<int, double, bool> d(true);
}
