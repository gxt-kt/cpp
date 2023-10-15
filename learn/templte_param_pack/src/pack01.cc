#include "common.h"

namespace template_class {

// Args... must be the last parameter of class  template
template <typename T, typename... Args>
struct TestClass {};

}  // namespace template_class

namespace template_function {

// Function template not necessary set Args... the last parameter
// Just ensure can deduce the paramter
template <typename... Args, typename T>
void TestFunc(T t, Args... args){};

// cannot deduce, so cannot use sucessfully
template <typename... Args, typename T>
void TestFunc1(Args... args, T t){};

// can deduce
template <typename... Args, typename T, typename U = int>
void TestFunc2(T t, U u, Args... args){};

// cannot deduce, so cannot use sucessfully
template <typename... Args, typename T, typename U = int>
void TestFunc3(T t, Args... args, U u){};

}  // namespace template_function

namespace template_nums {

template <int... Nums>
void TestNums() {
  int tmp[] = {(std::cout << Nums, 0)...};
  std::cout << std::endl;
};

}  // namespace template_nums

namespace template_inherit {

template <typename... Args>
struct Base1 {
  Base1(Args... args) {}
  Base1(const Base1&) { gDebug(__PRETTY_FUNCTION__); }
  void Fun1() { gDebug(__PRETTY_FUNCTION__); }
};
template <typename... Args>
struct Base2 {
  Base2(Args... args) {}
  Base2(const Base2&) { gDebug(__PRETTY_FUNCTION__); }
  void Fun2() { gDebug(__PRETTY_FUNCTION__); }
};

template <typename... Args>
struct Derived : public Args... {
  Derived(const Args&... args) : Args(args)... {}
};
}  // namespace template_inherit

int main() {
  { template_class::TestClass<int, double, bool> testclass; }
  {
    template_function::TestFunc(1, 2);
    // TestFunc1(1, 2);
    template_function::TestFunc2(1, 2);
    template_function::TestFunc2(1, 2, 3);
    // TestFunc3(1, 2, 3);
    // TestFunc3(1, 2, 3, 4);
  }
  { template_nums::TestNums<1, 2, 3>(); }
  {
    using namespace template_inherit;
    Base1<int> base1(10);
    Base2<double, bool> base2(20.0, true);
    Derived<Base1<int>, Base2<double, bool>> test(base1, base2);
    test.Fun1();
    test.Fun2();
  }
}
