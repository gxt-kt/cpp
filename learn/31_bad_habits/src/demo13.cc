#include <iostream>

// Use constexpr to calculate some expression on the compile time.
constexpr int Caculate(int a,int b) {
  return a*100+b;
}

int main() {
  std::cout << Caculate(10,20);
}

