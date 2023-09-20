#include <iostream>

struct Point{
  int x,y,z;
};

Point Test() {
  return {1,2,3};
}

int main() {
  auto [x,y,z]=Test();
}

