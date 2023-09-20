#include <iostream>

void Test() {
  int x;                // will be garbage
  int x1{};             // will be initialize to zero
  int* x2 = new int;    // will be garbage
  int* x3 = new int();  // will be initialize to zero
  int* x4 = new int{};  // will be initialize to zero
  // int x5();          // Is a function not a value initialize
}

int main() {
  Test();
  return 0;
}
