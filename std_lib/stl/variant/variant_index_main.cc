#include "common.h"

/*
 *
 * 这里主要是想要解决，对于一个变长模板参数
 * 1. 给一个类型，怎么知道这个类型是第几个
 * 2. 给一个数字，怎么知道这个第数字个类型
 */

template <typename... T>
struct Types {};

// VariantIndex 作用就是根据类型知道是第几个
template <typename, typename>  // typename -> size_t
struct VariantIndex;
// 递归的终止条件
template <typename T, typename... Ts>
struct VariantIndex<Types<T, Ts...>, T> {
  static constexpr size_t value = 0;
};
// 递归调用自身
template <typename T0, typename T, typename... Ts>
struct VariantIndex<Types<T0, Ts...>, T> {
  static constexpr size_t value = VariantIndex<Types<Ts...>, T>::value + 1;
};

// VariantAlternative 的作用就是知道第几个是什么类型
template <typename, size_t>  // size_t -> typename
struct VariantAlternative;
// 递归的终止条件
template <typename T, typename... Ts>
struct VariantAlternative<Types<T, Ts...>, 0> {
  using type = T;
};
// 递归调用自身
template <typename T, typename... Ts, size_t I>
struct VariantAlternative<Types<T, Ts...>, I> {
  using type = typename VariantAlternative<Types<Ts...>, I - 1>::type;
};

int main(int argc, char *argv[]) {
  using type = Types<int, double, std::string>;

  {
    static_assert(VariantIndex<type, int>::value == 0);
    static_assert(VariantIndex<type, double>::value == 1);
    static_assert(VariantIndex<type, std::string>::value == 2);
  }

  {
    static_assert(std::is_same<VariantAlternative<type, 0>::type, int>::value ==
                  true);
    static_assert(
        std::is_same<VariantAlternative<type, 1>::type, double>::value == true);
    static_assert(
        std::is_same<VariantAlternative<type, 2>::type, std::string>::value ==
        true);
  }

  return 0;
}
