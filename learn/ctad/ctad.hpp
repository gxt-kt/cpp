#pragma once

#include "common.h"

// ctad must use at least c++17

template <typename T>
struct CtadTest1 {
  T a_{};
  CtadTest1(){};
  explicit CtadTest1(T a) : a_(a){};
};

template <typename T>
struct CtadTest2 {
  T a_{};
  CtadTest2(){};

  template <typename U>
  CtadTest2(U x) {}
};

template <typename T>
struct CtadTest3 {
  T a_{};
  CtadTest3(){};
  explicit CtadTest3(T a) : a_(a){};

  template <typename U>
  CtadTest3(U x) {}
};

// set for CtadTest3 template construct function
template <typename T>
CtadTest3(T) -> CtadTest3<T>;
