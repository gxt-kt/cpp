#include <iostream>
#include <vector>

std::vector<int> Test() {
  std::vector<int> vector1(10000);
  // return vector1; // The compiler will auto use RVO
  return std::move(vector1);
}

std::vector<int> Test2() {
  std::vector<int> vector1(10000);
  int n = 0;
  // complicate condition cannot use RVO , so you can use std::move manually
  if (n == 0) {
    return std::move(vector1);
  } else {
    return std::move(vector1);
  }
}

int main() {
  std::vector<int> vector1 = Test();
  std::vector<int> vector2 = Test2();
  return 0;
}
