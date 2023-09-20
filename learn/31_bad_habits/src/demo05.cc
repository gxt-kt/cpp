#include <iostream>
#include <array>

// void Input(int *add,int n) {
//   std::cout << add[0];
// }

// use std::array to replace c style array
template <std::size_t n>
void Input(std::array<int, n> array1) {
  std::cout << array1[0];
}

int main() {
}
