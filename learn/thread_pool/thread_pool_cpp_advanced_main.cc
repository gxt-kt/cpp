#include "thread_pool_cpp_advanced.hpp"

// ref: https://gitee.com/lc123/thread-pool

void taskFunc(int arg) {
  int num = arg;
  gxt::Sleep(arg);
  // std::cout << std::this_thread::get_id() << VAR(num) << std::endl;
  gDebug() << std::this_thread::get_id() << VAR(num);
}

int main() {
  // 创建线程池
  ThreadPool pool(3);
  for (int i = 0; i < 40; ++i) {
    pool.commit(taskFunc, gxt::Random(0,10));
  }
  pool.Join();

  // gxt::Sleep(15);

  // threadPoolDestroy(pool);
  return 0;
}
