#include <iostream>
#include <memory>
#include <thread>
#include <mutex>


std::mutex mtx;

void Work(std::shared_ptr<int> val) {
  for (int i{0}; i < 100000; i++) {
    mtx.lock();
    ++(*val);
    mtx.unlock();
  }
}

int main() {
  auto ptr1 = std::make_shared<int>(0);
  std::thread thread1(Work, ptr1);
  std::thread thread2(Work, ptr1);

  thread1.join();
  thread2.join();

  // If you don't use mutex lock, the cout value will lost.
  std::cout << *ptr1 << "\n";

  return 0;
}
