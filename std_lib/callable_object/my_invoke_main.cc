#include "callable_objects.hpp"

// TODO: add my invoke code

// template <class F, class D = F>
// struct FuncImpl {
//   static_assert(!std::is_same_v<F, F>, "error");
//   // FuncImpl(F f) : m_f(std::move(f)) {}
// };
// template <class F>
// struct FuncImpl<F, std::enable_if_t<!std::is_member_function_pointer_v<F>, F>>
//      {  // FuncImpl
//   F m_f;
//   FuncImpl(F f) : m_f(std::move(f)) {}
//   void call(Args... args) override {
//     // 完美转发所有参数给构造时保存的仿函数对象
//     gDebug("not mem");
//     return m_f(std::forward<Args>(args)...);
//   }
// };
// template <class F>
// struct FuncImpl<F, std::enable_if_t<std::is_member_function_pointer_v<F>, F>>
//      {
//   F m_f;
//   FuncImpl(F f) : m_f(std::move(f)) {}
//   template <typename First, typename... Rest>
//   Ret callImpl(First&& first, Rest&&... rest) {
//     return (std::forward<First>(first)->*m_f)(std::forward<Rest>(rest)...);
//     gDebug(TYPE(m_f));
//   }
//   void call(Args... args) override {
//     callImpl(std::forward<Args>(args)...);
//   }
// };

struct A{
  A() = delete;
  int a;
  int Fun(){return 1;}
};


template<typename T>
T&& declval() noexcept
{
    // static_assert(false, "declval not allowed in an evaluated context");
}

int main(int argc, char* argv[]) {
  gDebug() << "exec" << __FILE__;

  // decltype(A().Fun()) aa;
  decltype(std::declval<A>().Fun()) ab;
  decltype(declval<A>().a) ac;
  gDebug(TYPE(declval<A>()));
  gDebug(TYPE(std::declval<A>()));
  ab=2;
  gDebug(TYPE(ac));
  gDebug(TYPE(ab));

  // { std::invoke(GXT_NAMESPACE::Foo_1, 1); }
  // {
  //   std::invoke(GXT_NAMESPACE::Foo_2_1(), 2);

  //   std::invoke(GXT_NAMESPACE::Foo_2_2, 2);
  // }
  // { std::invoke(GXT_NAMESPACE::Foo_3(), 3); }
  // {
  //   GXT_NAMESPACE::Foo_4 foo;
  //   // // Note that the fourth form is special
  //   std::invoke(&GXT_NAMESPACE::Foo_4::Foo, foo, 4);
  // }
  return 0;
}
