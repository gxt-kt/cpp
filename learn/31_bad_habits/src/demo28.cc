#include <iostream>

void Test() {
  int a{0};
  const int* b = &a; // apply int
  int* const c = &a; // apply pointer
  int const* d = &a; // apply int
  const int* const e = &a; // first apply int , second apply pointer
}

int main() { 
  return 0;
}
