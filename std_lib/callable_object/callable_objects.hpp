#pragma once

#include "common.h"

GXT_NAMESPACE_BEGIN

/*
Callable objects are the collection of all C++ structures which can be used as a
function.
And can be classified to follow types.
*/

// 1. Function Pointers
inline void Foo_1(int) { std::cout << __PRETTY_FUNCTION__ << std::endl; }

// 2. Class objects that overload the function call operator `operator()`
//    Note that the lambda's essence is a instantiated object that overload '()'
struct Foo_2_1 {
  void operator()(int) { std::cout << __PRETTY_FUNCTION__ << std::endl; }
};
inline auto Foo_2_2 = [](int) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
};

// 3. Class objects that can be converted to function pointer.
using Foo_3_Fun = void (*)(int);
struct Foo_3 {
  static void Foo(int) { std::cout << __PRETTY_FUNCTION__ << std::endl; }
  operator Foo_3_Fun() {  // if return class function must use static
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    return Foo;
  }
};

// 4. Member Function Pointers
struct Foo_4 {
  void Foo(int) {  // void(Foo_4::*)(int)
    std::cout << __PRETTY_FUNCTION__ << std::endl;
  }
};

GXT_NAMESPACE_END
