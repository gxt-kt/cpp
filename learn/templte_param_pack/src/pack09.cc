#include "common.h"

// ( pack op ... )
// Unary right fold
template <typename... Args>
auto Sum1(Args... args) {
  // like (1+(2+(3+(4))))
  return (args + ...);
}

// ( ... op pack )
// Unary left fold.
template <typename... Args>
auto Sum2(Args... args) {
  // like ((((1)+2)+3)+4)
  return (... + args);
}

// ( pack op ... op init )
// Binary right fold.
template <typename... Args>
auto Sum3(Args... args) {
  // like (1+(2+(3+(4+0))))
  return (args + ... + 0);
}

// ( init op ... op pack )
// Binary left fold.
template <typename... Args>
auto Sum4(Args... args) {
  // like ((((0+1)+2)+3)+4)
  return (0 + ... + args);
}

template <typename... Args>
void Print(Args... args) {
  // use binary left fold here
  (std::cout << ... << args) << std::endl;
}

int main() {
  {
    gDebug(Sum1(1, 2, 3, 4));
    gDebug(Sum2(1, 2, 3, 4));
    gDebug(Sum3(1, 2, 3, 4));
    gDebug(Sum4(1, 2, 3, 4));
  }
  {
    // error because the unary right fold is (string+(char*+char*))
    // gDebug(Sum1(std::string("hello"), " ", "world"));

    gDebug(Sum2(std::string("hello"), " ", "world"));
  }
  {
    Print(1, 2, 3);
    Print(4, 5, 6);
  }
  {
    // error because unary fold cannot deduce the return auto type
    // gDebug(Sum1());
    // gDebug(Sum2());

    // binary can deduce because there is a init value
    gDebug(Sum3());
    gDebug(Sum4());
  }
}
