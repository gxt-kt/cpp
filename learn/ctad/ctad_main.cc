#include "ctad.hpp"

int main(int argc, char *argv[]) {
  {
    std::vector<int> a{1, 2, 3};
    std::vector b{1, 2, 3};  // use ctad
  }
  {
    std::pair<std::string, int> a{"123", 4};
    auto b = std::make_pair("123", 4);
    std::pair c{"123", 4};  // use ctad
  }
  {
    std::tuple<std::string, int, float> a{"123", 4, 5.0f};
    auto b = std::make_tuple("123", 4, 5.0f);
    std::tuple c{"123", 4, 5.0f};  // use ctad
  }
  {
    std::array<int, 3> a{1, 2, 3};
    // auto deduce the type and cnt
    std::array b{1, 2, 3};  // use ctad
  }

  {
    CtadTest<int> a(10);
    a.Fun();
    // error because the T cannot be deduced
    // CtadTest b();  // use ctad
    // b.Fun();
    CtadTest c(10);  // use ctad
    c.Fun();
    CtadTest d("123");  // use ctad
    d.Fun();
  }

  return 0;
}
