#include <iostream>

int main() {
  for (int i{0};i<10;i++) {
    // std::cout << "i=" << i << std::endl; // this will refresh the buffer and take more time
    std::cout << "i=" << i << "\n"; // do it better
  }
}
