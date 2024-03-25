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
  gDebug() << is_specialization<std::vector<int>, std::vector>::value;
  gDebug() << is_specialization<std::list<int>, std::vector>::value;
  gDebug() << is_specialization<std::tuple<int, double>, std::tuple>::value;
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

  // But the template struct not support lambda expression
  auto has_input = [](int) {};
  auto hasnt_input = []() {};
  gDebug(IfFunctionHasInputValue<decltype(has_input)>::value);
  gDebug(IfFunctionHasInputValue<decltype(hasnt_input)>::value);
}

}  // namespace demo03

namespace demo04 {
// use std::enable_if with function return type to constraint T

// T must be integer
template <typename T>
typename std::enable_if<std::is_integral_v<T>, T>::type Fun(T t) {
  gDebug("is_integral_v");
  return t;
}
// T must be float type
template <typename T>
typename std::enable_if<std::is_floating_point_v<T>, T>::type Fun(T t) {
  gDebug("is_floating_point_v");
  return t;
}

// the same like Fun
template <typename T, std::enable_if_t<std::is_integral_v<T>, size_t> = 0>
void Fun1(T t) {}
// the same like Fun
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, size_t> = 0>
void Fun1(T t) {}

void Test() {
  Fun(10);
  Fun(10.0f);
  // Fun("123"); // error: input value isn't integer or float type
  Fun1(10);
  Fun1(10.0f);
  // Fun1("123"); // error: input value isn't integer or float type
}

}  // namespace demo04

namespace demo05 {
// use std::enable_if with class template

// If not understand class template ues default parameter and specialization at the same time
// Can read 
// https://blog.csdn.net/my_id_kt/article/details/133820016?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22133820016%22%2C%22source%22%3A%22my_id_kt%22%7D


// This type of A will not be instantiated, so unnecessnary to be definded 
// Just claim to let others to partial specialization
template <typename T,typename U=void>
struct A;

template <typename T>
struct A<T,std::enable_if_t<std::is_integral_v<T>>> : std::true_type {};

template <typename T>
struct A<T,std::enable_if_t<std::is_floating_point_v<T>>> : std::false_type {};

void Test() {
  gDebug(A<int>::type::value);
  gDebug(A<float>::type::value);
  // But if use other type, you need to define basic A class.
  // gDebug(A<char*>::type::value);
}

}  // namespace demo05

int main(int argc, char *argv[]) {
  demo01::Test();
  gDebugCol1() << G_SPLIT_LINE;
  demo02::Test();
  gDebugCol1() << G_SPLIT_LINE;
  demo03::Test();
  gDebugCol1() << G_SPLIT_LINE;
  demo04::Test();
  gDebugCol1() << G_SPLIT_LINE;
  demo05::Test();
  gDebugCol1() << G_SPLIT_LINE;
  return 0;
}
