#include "common.h"

int Add(int lhs, int rhs) { return lhs + rhs; }
int Sub(int lhs, int rhs) { return lhs - rhs; }

template <typename... Args>
void Foo(Args (*... args)(int, int)) {
  // the same as int tmp[] = {(std::cout << arg0(1, 2) << std::endl, 0),(std::cout << arg1(1, 2) << std::endl, 0)};
  int tmp[] = {(std::cout << args(1, 2) << std::endl, 0)...};
}

int main() {
  gDebug() << (Add(1,2),Sub(1,2));
  Foo(Add,Sub);

}
