#include <iostream>

// can use reinterpret_cast to watch memory bytes
template <typename T>
void PrintMemoryBytes(const T& input) {
  auto* byte_addr=reinterpret_cast<const std::byte*>(&input); // std::byte need cpp17
  for(int i{0};i<sizeof(T);i++) {
    std::cout << byte_addr+i << " : " << static_cast<int>(*(byte_addr+i)) << "\n";
  }
  std::cout << std::endl;
}


int main() {
  float a{1.1};
  int b=*(int*)(&a); // UB
  int c=*reinterpret_cast<int*>(&a); // UB
  PrintMemoryBytes(a);
}
