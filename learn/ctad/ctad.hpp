#pragma once

#include "common.h"

// ctad must use at least c++17

template <typename T>
struct CtadTest {
  T a_{};
  CtadTest(){};
  explicit CtadTest(T a) : a_(a){};
  void Fun() { gDebug(TYPET(T)); }
};


// template <typename T>
// struct CtadTest {
//   T a_{};
//   CtadTest(){};
//   explicit CtadTest(T a) : a_(a){};
//   void Fun() { gDebug(TYPET(T)); }
// };
