#include "thread_pool_cpp.hpp"

void taskFunc(void* arg) {
  int num = *(int*)arg;
  printf("thread %ld is working, number = %d\n", pthread_self(), num);
  sleep(1);
}

int main() {
  // 创建线程池
  ThreadPool pool(3, 10);
  for (int i = 0; i < 100; ++i) {
    int* num = (int*)malloc(sizeof(int));
    *num = i + 100;
    pool.addTask(Task(taskFunc, num));
  }

  sleep(20);

  // threadPoolDestroy(pool);
  return 0;
}
