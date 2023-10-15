#include "common.h"

template <typename... Args>
struct Tuple {
  void Fun() { gDebug(__PRETTY_FUNCTION__); }
};

template <typename T, typename U>
struct pair {};

template <typename... Args>
struct zip {
  template <typename... Argss>
  struct with {
    using type = Tuple<pair<Args, Argss>...>;
  };
};

int main() {
  {
    zip<int, float>::with<double, bool>::type t;
    t.Fun();
  }

  {
    // error: unpack serveral Args... at the same time
    // must satisfy the serveral Args have same size
    // zip<int>::with<double, bool>::type t;
    // t.Fun();
  }
}
