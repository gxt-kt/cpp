#include <iostream>

// using namespace std; // Do not do like this.

void Test() {
  using namespace std; // only for the function
  cout << "hello" << endl;
}

int main() {
  Test();
  std::cout << "hello" << std::endl;
  // cout << "hello" << endl; // error
}
