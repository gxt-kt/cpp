#include <algorithm>
#include <iostream>
#include <vector>

int FindTheFirstNumZeroIndex(const std::vector<int>& vector1) {
  // for(int i{0};i<vector1.size();i++) {
  //   if (vector1[i]==0)
  //     return i;
  // }

  auto function = [](const auto& val) -> int { return val == 0; };

  auto num_index = std::find_if(vector1.begin(), vector1.end(), function);
  return num_index - vector1.begin();
}

int main() {
  std::cout << FindTheFirstNumZeroIndex({1, 2, -1, 0, 3});
}
