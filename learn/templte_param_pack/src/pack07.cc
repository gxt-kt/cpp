#include "common.h"

template <typename F, typename... Args>
[[nodiscard]] auto DelayInvoke(F f, Args... args) {
  return [f, args...]() -> decltype(auto) { return std::invoke(f, args...); };
}

template <typename F, typename... Args>
[[nodiscard]] auto DelayInvokeOpti(F f, Args... args) {
  return [f = std::move(f),
          tup = std::make_tuple(std::move(args)...)]() -> decltype(auto) {
    return std::apply(f, tup);
  };
}

int main() {
  {
    auto max = [](auto a, auto b) { return a > b ? a : b; };
    auto foo = DelayInvoke(max, 10, 20);
    gDebug(foo());
    auto goo = DelayInvokeOpti(max, 10, 20);
    gDebug(goo());
  }
}
