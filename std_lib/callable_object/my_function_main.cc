#include "callable_objects.hpp"

// Ref: 类型擦除可以参考：
// https://github.com/apachecn/apachecn-c-cpp-zh/blob/master/docs/adv-cpp-prog-cb/09.md
// https://fuzhe1989.github.io/2017/10/29/cpp-type-erasure/

GXT_NAMESPACE_BEGIN

template <class FnSig>
struct function {
  // 只在使用了不符合 Ret(Args...) 模式的 FnSig 时出错
  static_assert(!std::is_same<FnSig, FnSig>::value,
                "not a valid function signature");
};

template <class Ret, class... Args>
struct function<Ret(Args...)> {
 private:
  struct FuncBase {
    virtual Ret Call(Args... args) = 0;  // 类型擦除后的统一接口
    virtual ~FuncBase() = default;  // 应对F可能有非平凡析构的情况
  };

  // FuncImpl会被实例化多次，每个不同的仿函数类都产生一次实例化
  template <class F>
  struct FuncImpl : FuncBase {
    F m_f;

    FuncImpl(F f) : m_f(std::move(f)) {}

    virtual Ret Call(Args... args) override {
      // 完美转发所有参数给构造时保存的仿函数对象
      // 这样有bug,没法使用成员函数的情况，需要判断使用is_member_function_pointer
      // 修复这个bug其实就相当于实现了自己的invoke了
      // return m_f(std::forward<Args>(args)...);
      // 更规范的写法其实是：
      return std::invoke(m_f, std::forward<Args>(args)...);
      // 但为了照顾初学者依然采用朴素的调用方法
    }
  };

  // 使用智能指针管理仿函数对象，用shared而不是unique是为了让function支持拷贝
  std::shared_ptr<FuncBase> m_base;

 public:
  function() = default;  // m_base 初始化为 nullptr

  // 此处 enable_if_t 的作用：阻止 function 从不可调用的对象中初始化
  template <class F, class = typename std::enable_if<
                         std::is_invocable_r<Ret, F &, Args...>::value>::type>
  function(F f)  // 没有 explicit，允许 lambda 表达式隐式转换成 function
      : m_base(std::make_shared<FuncImpl<F>>(std::move(f))) {}

  Ret operator()(Args... args) const {
    if (!m_base) [[unlikely]]
      throw std::runtime_error("function not initialized");
    // 完美转发所有参数，这样即使 Args 中具有引用，也能不产生额外的拷贝开销
    return m_base->Call(std::forward<Args>(args)...);
  }
};

GXT_NAMESPACE_END

int main(int argc, char *argv[]) {
  {
    GXT_NAMESPACE::function<void(int)> fun(GXT_NAMESPACE::Foo_1);
    fun(1);
  }
  {
    GXT_NAMESPACE::Foo_2_1 foo;
    GXT_NAMESPACE::function<void(int)> fun1(foo);
    fun1(2);

    GXT_NAMESPACE::function<void(int)> fun2(GXT_NAMESPACE::Foo_2_2);
    fun2(2);
  }
  {
    GXT_NAMESPACE::Foo_3 foo;
    GXT_NAMESPACE::function<void(int)> fun(foo);
    fun(3);
  }
  {
    GXT_NAMESPACE::Foo_4 foo;
    // Note that the fourth form is special
    GXT_NAMESPACE::function<void(GXT_NAMESPACE::Foo_4 *, int)> fun(
        &GXT_NAMESPACE::Foo_4::Foo);
    fun(&foo, 4);
  }

  return 0;
}
