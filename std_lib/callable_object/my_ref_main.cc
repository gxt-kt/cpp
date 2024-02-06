#include "callable_objects.hpp"

// Ref: 类型擦除可以参考：

GXT_NAMESPACE_BEGIN

template <typename _Tp>
class reference_wrapper {
 public:
  _Tp* _M_data;

  reference_wrapper(_Tp& __uref) : _M_data(&__uref) {}

  reference_wrapper(const reference_wrapper&) = default;

  reference_wrapper& operator=(const reference_wrapper&) = default;

  operator _Tp&() const noexcept { return this->get(); }

  _Tp& get() const noexcept { return *_M_data; }
};

template <typename _Tp>
_GLIBCXX20_CONSTEXPR inline reference_wrapper<_Tp> ref(_Tp& __t) noexcept {
  return reference_wrapper<_Tp>(__t);
}

GXT_NAMESPACE_END

using namespace std::placeholders;

void Fun(int a) {
  a++;
}
void Fun1(int& a) {
  a++;
}

int main(int argc, char* argv[]) {
  gDebug() << "exec" << __FILE__;
  int a = 10;
  {
    std::function<void()> f = std::bind(Fun, a);    // default is pass by value
    std::function<void()> f1 = std::bind(Fun1, a);  // default is pass by value
    f();
    gDebug(a);
    f1();
    gDebug(a);
  }
  {
    std::function<void()> f = std::bind(Fun, GXT_NAMESPACE::ref(a));  // use ref to pass by ref
    std::function<void()> f1 = std::bind(Fun1, GXT_NAMESPACE::ref(a));  // use ref to pass by ref
    f();
    gDebug(a);
    f1();
    gDebug(a);
  }
  return 0;
}
