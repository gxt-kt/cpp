#include <future>
#include <iostream>
#include <string>

#include "common.h"

std::string processData(const std::string& input) {
  gDebug("begin sub thread id:") << std::this_thread::get_id();
  // 在子线程中解析和处理数据
  std::string res = input + input;

  // 模拟处理时间
  std::this_thread::sleep_for(std::chrono::seconds(2));

  return res;
}

int main() {
  std::cout << "请输入要处理的数据: ";
  std::string input;
  std::getline(std::cin, input);

  // 使用 std::async 创建一个异步任务，将处理函数和输入数据绑定
  std::future<std::string> resultFuture =
      std::async(std::launch::async, processData, input);

  // 主线程可以继续执行其他任务

  // 在需要的时候获取子线程的处理结果
  // 这里使用 std::future::wait_for 来检查子线程是否完成
  // 如果子线程未完成，可以继续执行其他任务，避免主线程阻塞
  // 当子线程完成后，可以通过 std::future::get 获取结果
  std::future_status status = resultFuture.wait_for(std::chrono::seconds(0));
  // 其实关于这个循环，和用std::thread加上循环感觉是差不多的
  while (status != std::future_status::ready) {
    // 子线程未完成，可以执行其他任务
    // 这里可以添加你需要的逻辑
    // ...

    // 检查子线程状态
    status = resultFuture.wait_for(std::chrono::seconds(0));
  }

  // 子线程已完成，获取结果并打印
  std::string processedData = resultFuture.get();
  std::cout << "处理后的数据: " << processedData << std::endl;

  return 0;
}

