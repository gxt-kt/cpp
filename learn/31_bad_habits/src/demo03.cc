#include <iostream>
#include <vector>

void PrintVector(const std::vector<int>& vector1) {
  // for(int i{0};i<vector1.size();i++) {
  //   std::cout << vector1[i] << '\n';
  // }

  // use for loop based of auto
  for(const auto& val:vector1) {
    std::cout << val << '\n';
  }

}

int main() {
  PrintVector({1,2,3});
}
