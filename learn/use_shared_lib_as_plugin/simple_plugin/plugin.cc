#include <iostream>

extern "C" {

int Add(int a, int b) { return a + b; }

void HelloDemo() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
}
