#include <iostream>
#include <vector>

// Don't forget const if you sure you don't change the parameter
void DontFallConstDown(const std::vector<int>& vector1) {
  for(auto& val:vector1) {
    std::cout << val << "\n";
  }
}

// The string is fixed and alive in all the program life. So use the const
const char* ReturnConstString() {
  return "hello world";
}

int main() {

}

