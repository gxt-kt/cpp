#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "common.h"

std::mutex mtx;
std::condition_variable cv;

std::string input_data;
std::string output_data;

bool ready = false;
bool processed = false;

void processData() {
  gDebug("begin sub thread id:") << std::this_thread::get_id();
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [] { return ready; });

  // 在子线程中解析和处理数据
  output_data = input_data + input_data;

  // 模拟处理时间
  std::this_thread::sleep_for(std::chrono::seconds(2));

  processed = true;
  lock.unlock();
  cv.notify_one();
}

/*
*
* 如果处理数据特别耗时，可以用多个线程进行处理。
* 比如主线程接收一次-> 给到一个子线程处理
* 再接受一次，给到另一个子线程处理。
*
* 如果这样的话，就可能要考虑双缓冲区了
*/

int main() {
  std::cout << "请输入要处理的数据: ";
  std::getline(std::cin, input_data);

  std::thread workerThread(processData);

  {
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
  }
  cv.notify_one();

  {
    std::unique_lock<std::mutex> lock(mtx);
    // 这个有个缺点：主线程会一直被阻塞直到子线程执行完
    // 可以使用循环改进，可以在循环内部做别的事情，然后还可以定时检查到子线程是否有数据要输出
    // 但是终究不是那么完美
    cv.wait(lock, [] { return processed; });
    std::cout << "处理后的数据: " << output_data << std::endl;
  }

  workerThread.join();

  return 0;
}

