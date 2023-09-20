#include <iostream>

class Test {
 public:
  Test(int a_, int b_) : a(a_), b(a + b_) {}

 private:
  // If you change the define order will be warning when initialize the b
  int a;
  int b;
};

;

int main() {
  Test(10, 20);
  return 0;
}
