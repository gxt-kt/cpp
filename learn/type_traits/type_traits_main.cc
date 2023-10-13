#include <type_traits>
#include <utility>

#include "type_traits.hpp"

namespace demo01 {
// Ref :
// https://stackoverflow.com/questions/27687389/how-do-we-use-void-t-for-sfinae
// use void_t feature to check some code is legal

// the same like std::void_t
template <typename...>
using void_t = void;

// Example1: to check if a class has `demo` member variable
struct A {
  int a;
};
struct B {
  int demo;
};

template <typename T, typename = void>
struct CheckHasMemberSomething : std::false_type {};

template <typename T>
struct CheckHasMemberSomething<T, void_t<decltype(std::declval<T>().demo)>>
    : std::true_type {};

inline void Test() {
  gDebug(CheckHasMemberSomething<A>::value);
  gDebug(CheckHasMemberSomething<B>::value);
}

}  // namespace demo01

namespace demo02 {
// Ref :
// https://stackoverflow.com/questions/66279769/trying-to-understand-partial-template-specialization-with-template-template-para
// To see a type whether is a specialization of a template class

// Some utility structs to check template specialization
template <typename Test, template <typename...> class Ref>
struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

void Test() {
  gDebug(is_specialization<std::vector<int>, std::vector>::value);
  gDebug(is_specialization<std::list<int>, std::vector>::value);
  gDebug(is_specialization<std::tuple<int, double>, std::tuple>::value);
}

}  // namespace demo02

namespace demo03 {
// a little like demo02
// To see whether a type is a function type that has input parameter

template <typename F, typename... Args>
struct IfFunctionHasInputValue : std::false_type {};
template <typename F>
struct IfFunctionHasInputValue<F()> : std::false_type {};
template <typename F, typename... Args>
struct IfFunctionHasInputValue<F(Args...)> : std::true_type {};

void Test() {
  void Fun1(int);
  void Fun2();
  // We use decltype, the code willn't be exec.
  // So we don't need to implement correspond function definition
  gDebug(IfFunctionHasInputValue<decltype(Fun1)>::value);
  gDebug(IfFunctionHasInputValue<decltype(Fun2)>::value);
  gDebug(IfFunctionHasInputValue<int>::value);  // other type

  // But the template struct still have bug for lambda expression
  auto has_input = [](int) {};
  auto hasnt_input = []() {};
  gDebug(IfFunctionHasInputValue<decltype(has_input)>::value);
  gDebug(IfFunctionHasInputValue<decltype(hasnt_input)>::value);

  std::function<int()> aa;
}

}  // namespace demo03

int main(int argc, char *argv[]) {
  demo01::Test();
  gDebugCol1() << G_SPLIT_LINE;
  demo02::Test();
  gDebugCol1() << G_SPLIT_LINE;
  demo03::Test();
  gDebugCol1() << G_SPLIT_LINE;
  return 0;
}
