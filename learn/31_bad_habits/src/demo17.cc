#include <iostream>

double CalculateCircumference(double r) {
  constexpr double pai=3.1415926;
  return 2*pai*r;
  // return 2*3.1415926*r; // use constexprt to make your value have a name
}

int main() {
  CalculateCircumference(10.0);
  return 0;
}
