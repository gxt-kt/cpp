#include <thread>

#include "concurrentqueue.h"
#include "lockfreequeue.hpp"

struct TT {
  int a;
  std::string b;
  double c;
  std::vector<int> d;
  int e[100];
};

using Test = TT;

template <typename T>
using FreeLockQueue=moodycamel::ConcurrentQueue<T>;

int main(int argc, char* argv[]) {
  FreeLockQueue<Test> queue(5000000);

  auto FunInputData = [&]() {
    int n = 100000;
    while (n--) {
      queue.enqueue(Test{n});
    }
  };

  TIME_BEGIN_MS();
  std::vector<std::thread> vec_threads(10);
  for (auto& thread : vec_threads) {
    thread = std::thread(FunInputData);
  }

  for (auto& thread : vec_threads) {
    thread.join();
  }
  TIME_END();

  int sum = 0;
  Test tmp;
  // int cnt{0};
  while (queue.try_dequeue(tmp)) {
    // if(tmp.a==10) {cnt++;}
    sum++;
  }
  gDebug(sum);
  // gDebug(cnt,sum);

  return 0;
}
