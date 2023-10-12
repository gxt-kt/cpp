#pragma once

// Ref :
// https://stackoverflow.com/questions/44963201/when-does-an-incomplete-type-error-occur-in-c
// https://learn.microsoft.com/en-us/cpp/c-language/incomplete-types?view=msvc-170

#include "common.h"

namespace demo01 {
struct A;
struct B {
  // A a; // error incomplete-types
};
struct A {
  int a;
};

};  // namespace demo01

namespace demo02 {
struct A;
struct B {
  A* a;  // can pass compile because A* is fixed
};
struct A {
  int a;
};

};  // namespace demo02

namespace demo03 {
struct A;
struct B {
  A* a;  // can use because A* is fixed
  void FooB() {
    // a.a=10;   // error cannot access full A
    // a.FooA(); // error cannot access full A
  }
};
struct A {
  int a;
  void FooA() {}
};

};  // namespace demo03
