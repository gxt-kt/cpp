#include "common.h"


auto Sum() {
  return 0;
}

template <typename T, typename... Args>
auto Sum(T val, Args... args) {
  gDebug(sizeof...(args));
  return val + Sum(args...);
}

int main() {
  gDebug(Sum(1,2,3,4));
}
