#include <iostream>
#include <vector>

void Test(std::vector<int>& vector1) {
  // This will not be stop because the vector1.size will be change
  for(int i{0};i<vector1.size();i++) {

    static int stop{1};
    if(stop++>10000) {
      std::cout << "will not stop";
      std::terminate();
    }

    vector1.push_back(10);
  }

  // This will not be stop because the vector1.end will move with the push_back
  for(auto it=vector1.begin();it<vector1.end();it++) {

    static int stop{1};
    if(stop++>10000) {
      std::cout << "will not stop";
      std::terminate();
    }

    vector1.push_back(10);
  }

  // Although this can stop but the value is strange because push_back make the vector1 change a new memory address.
  for(auto& val:vector1) {
    vector1.push_back(10);
    std::cout << val << "\n";
  }
}

int main() {
  std::vector<int> vector1={1,2,3};
  Test(vector1);
  return 0;
}
