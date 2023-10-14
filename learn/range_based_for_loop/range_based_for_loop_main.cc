#include "range_based_for_loop.hpp"

int main(int argc, char* argv[]) {
  {
    std::vector<int> aa(10);
    for (auto& val : aa) {
      val = gxt::Random(0, 100);
    }
    std::sort(aa.begin(), aa.end(), std::greater());
    gDebug(aa);
  }

  {
    std::unordered_map<int, std::string> aa;
    aa.insert({10, "123"});
    aa.insert({20, "456"});
    aa.insert({30, "789"});
    for (const auto& val : aa) {
      gDebug(val.first) << gDebug(val.second);
    }
  }

  {
    Test<int, 10> test;
    for (auto& val : test) {
      val = gxt::Random(0, 100);
      // gDebug(val);
    }
    gDebug() << G_SPLIT_LINE;
    for (auto& val : test) {
      gDebug(val);
    }
    gDebug() << G_SPLIT_LINE;
    gDebug() << test;
  }

  {
    gDebugCol3() << G_SPLIT_LINE;
    const Test<int, 10> test{12, 45, 123, 12};
    for (auto& val : test) {
      gDebug(val);
    }
    gDebugCol3() << G_SPLIT_LINE;
    gDebug() << test;
  }

  {
    Test<int, 10> test;
    for (auto& val : test) {
      static int i = 0;
      val = ++i;
    }
    std::cout << test << std::endl;
  }

  {
    const Test<int, 10> test{12, 34, 56, 78, 910};
    for (const auto& val : test) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }

}
