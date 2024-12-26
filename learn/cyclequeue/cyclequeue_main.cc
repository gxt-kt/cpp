#include "cyclequeue.hpp"

int main() {
  CycleQueue<int> q(5);
  q.Push(1);
  q.Push(2);
  q.Push(3);
  q.Push(4);
  for (int i = 0; i < 4; i++) {
    std::cout << q.Pop() << std::endl;
  }
  q.Push(5);
  q.Push(5);
  q.Push(5);
  std::cout << q.Pop() << std::endl;
  std::cout << q.Pop() << std::endl;
  std::cout << q.Pop() << std::endl;
  std::cout << q.Pop() << std::endl;
  return 0;
}
