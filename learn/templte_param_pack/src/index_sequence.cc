#include "common.h"

// about implement of index_sequence
// There is no source code in gcc/g++ and the compile did it
// So we haven't the source code to ref
//
// But if you want did it by c++ code
// Can refer abseil of google:
// https://chromium.googlesource.com/external/github.com/abseil/abseil-cpp/+/HEAD/absl/utility/utility.h
// There are also some other web to help you understand the index_sequence
// https://stackoverflow.com/questions/17424477/implementation-c14-make-integer-sequence
// https://en.cppreference.com/w/cpp/utility/integer_sequence
// https://cloud.tencent.com/developer/article/1768976

// generate a array : 0,1,4,9,16...(N-1)^2
template <size_t... N>
constexpr auto GenerateArrayImp(std::index_sequence<N...>) {
  return std::array{N * N...};
}
template <size_t N>
constexpr auto GenerateArray() {
  return GenerateArrayImp(std::make_index_sequence<N>{});
}

template <typename Tuple, typename F, size_t... N>
void TravelTupleImp(const Tuple& tup, F&& f, std::index_sequence<N...>) {
  // the following three method all can use
  int tmp[]{(f(std::get<N>(tup)), 0)...};
  // int tmp[]{(std::invoke(f,std::get<N>(tup)), 0)...};
  // std::initializer_list<int>({(f(std::get<N>(tup)), 0)...});
}

template <typename... Args, typename F>
void TravelTuple(const std::tuple<Args...>& tup, F&& f) {
  TravelTupleImp(tup, std::forward<F>(f),
                 std::make_index_sequence<sizeof...(Args)>{});
}

int main(int argc, char* argv[]) {
  {
    constexpr auto arr = GenerateArray<10>();
    gDebug(arr.size());
    gDebug(arr);
  }

  {
    auto tup = std::make_tuple(1, 2.0, true, "123");
    TravelTuple(tup, [](auto val) { gDebug() << val; });
  }

  return 0;
}
