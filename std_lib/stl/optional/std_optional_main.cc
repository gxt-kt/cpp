#include "std_optional.hpp"

// 示例来自官方示例：https://en.cppreference.com/w/cpp/utility/optional

// std::nullopt can be used to create any (empty) std::optional
auto create2(bool b) { return b ? Optional<std::string>{"Godzilla"} : nullopt; }

int main() {
  auto str = create2(true);
  if (str) std::cout << "create(true) returned " << *str << '\n';
}
