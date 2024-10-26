#include "common.h"

/*
 * visit实现上就是用来找到variant对应的类型，然后执行对应的lambda表达式（函数指针）
 *
 * 在源码实现上，有两种思路：
 * 1. 采用模板递归调用方法，一个一个检查过去，直到找到对应函数指针输入类型
 * 2. 查表法，预先定义好所有lambda表达式对应类型的数组，运行时直接查表就可以拿到对应的函数指针
 *
 */

//===================
// 递归方式的visit
//===================
namespace digui {
// 基础模板：递归终止条件
template <typename Visitor, typename Variant, std::size_t Index = 0>
decltype(auto) visit_impl(Visitor &&visitor, Variant &&variant) {
  using VariantType = std::decay_t<Variant>;
  // 检查当前 Index 是否是 Variant 中的有效类型索引
  if constexpr (Index < std::variant_size_v<VariantType>) {
    // 如果当前索引匹配，调用访问器
    if (variant.index() == Index) {
      return visitor(std::get<Index>(std::forward<Variant>(variant)));
    } else {
      // 否则递归检查下一个索引
      return visit_impl<Visitor, Variant, Index + 1>(
          std::forward<Visitor>(visitor), std::forward<Variant>(variant));
    }
  } else {
    // 如果索引超出范围，抛出异常
    throw std::runtime_error("Bad variant access");
  }
}

// 用户接口：包装实现
template <typename Visitor, typename Variant>
decltype(auto) visit(Visitor &&visitor, Variant &&variant) {
  return visit_impl(std::forward<Visitor>(visitor),
                    std::forward<Variant>(variant));
}
}  // namespace digui

//===================
// 查表方式的visit
//===================
namespace chabiao {
// 定义一个帮助的函数指针
template <class Lambda, typename... Ts>
using VisitorFunction = typename std::common_type<typename std::invoke_result<
    Lambda, Ts &>::type...>::type (*)(Lambda &&, std::variant<Ts...> &);
// 一个函数，返回值是一个static的数组，数组内容是不同类型构造的lambda对象
template <class Lambda, typename... Ts>
VisitorFunction<Lambda, Ts...> *visitors_table() noexcept {
  static VisitorFunction<Lambda, Ts...> function_ptrs[sizeof...(Ts)] = {
      [](Lambda &&lambda, std::variant<Ts...> &var) ->
      typename std::common_type<
          typename std::invoke_result<Lambda, Ts &>::type...>::type {
        return std::invoke(std::forward<Lambda>(lambda),
                           *std::get_if<Ts>(&var));
      }...};
  return function_ptrs;
}
// 根据构造的表，可以直接找到对应的函数指针
template <class Lambda, typename... Ts>
typename std::common_type<
    typename std::invoke_result<Lambda, Ts &>::type...>::type
visit(Lambda &&lambda, std::variant<Ts...> &var) {
  return visitors_table<Lambda, Ts...>()[var.index()](
      std::forward<Lambda>(lambda), var);
}
};  // namespace chabiao

int main(int argc, char *argv[]) {
  // 定义一个 std::variant，可以存储 int 或 double 或 std::string
  std::variant<int, double, std::string> var;

  var = 42;
  std::visit([](auto val) { std::cout << val << std::endl; }, var);  // 输出：42
  digui::visit([](auto val) { std::cout << val << std::endl; },
               var);  // 输出：42
  chabiao::visit([](auto val) { std::cout << val << std::endl; },
                 var);  // 输出：42
  return 0;
}
