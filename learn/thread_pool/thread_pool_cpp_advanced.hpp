#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>

#include "common.h"

using Task = std::function<void()>;

class TaskQueue {
 public:
  void AddTask(const Task task) {
    std::lock_guard<std::mutex> lock(que_mutex_);
    tasks_.push(task);
  }
  size_t GetSize() {
    std::lock_guard<std::mutex> lock(que_mutex_);
    return tasks_.size();
  }
  Task GetTask() {
    std::lock_guard<std::mutex> lock(que_mutex_);
    if (tasks_.empty()) {
      return nullptr;
    }
    auto task = tasks_.front();
    tasks_.pop();
    return task;
  }
  bool Empty() {
    std::lock_guard<std::mutex> lock(que_mutex_);
    return tasks_.empty();
  }

 private:
  std::queue<Task> tasks_;
  std::mutex que_mutex_;
};

class WorkThread;
using ThreadPtr = std::shared_ptr<WorkThread>;

class WorkThread {
 public:
  enum Status : int {
    WAIT,
    WORK,
    EXIT,
  };

  WorkThread(TaskQueue &task_queue, std::condition_variable &cond_var,
             std::mutex &mtx)
      : task_queue_(task_queue), cond_var_(cond_var), mutex_(mtx) {
    thread_ = std::thread([this]() {
      while (!finished_) {
        state_ = WAIT;
        {
          std::unique_lock<std::mutex> look(mutex_);
          if (task_queue_.Empty()) {
            cond_var_.wait(look,
                           [&]() { return !task_queue_.Empty() || finished_; });
          }
        }
        if (finished_) {
          break;
        }
        Task task = task_queue_.GetTask();
        if (task != nullptr) {
          state_ = WORK;
          task();
        }
      }
      state_ = EXIT;
    });
  }

  ~WorkThread() {
    gDebugCol2(__PRETTY_FUNCTION__) << GetId();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  int GetState() const { return state_; }

  void Finish() { finished_ = true; }

  std::thread::id GetId() const { return thread_.get_id(); }

  std::thread &GetThread() { return thread_; }

 private:
  TaskQueue &task_queue_;
  std::condition_variable &cond_var_;
  std::mutex &mutex_;
  std::atomic_int state_;
  std::atomic_bool finished_;
  std::thread thread_;
};

class ThreadPool {
 public:
  ThreadPool(int min = 1, int max = std::thread::hardware_concurrency())
      : finished_(false), min_(min), max_(max) {
    for (int i = 0; i < min_; i++) {
      AddThread();
    }

    manage_thread_ = std::thread([this]() {
      while (!finished_) {
        if ((task_queue_.GetSize() > 2 * threads_.size()) &&
            (threads_.size() < max_)) {
          AddThread();
        } else {
          int cnt = 0;
          for (auto &t : threads_) {
            if (t.second->GetState() == WorkThread::WAIT) {
              ++cnt;
            }
          }
          if ((cnt > 2 * task_queue_.GetSize()) && (threads_.size() > min_)) {
            DelThread();
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    });
  }

  ~ThreadPool() {
    Finish();
    if (manage_thread_.joinable()) {
      manage_thread_.join();
    }
  }

  void Finish() {
    finished_ = true;
    for (auto &thread : threads_) {
      thread.second->Finish();
    }
    std::unique_lock<std::mutex> look(mutex_);
    cond_var_.notify_all();
  }

  template <class F, class... Args>
  auto commit(F &&f, Args &&...args) -> std::future<decltype(f(args...))> {
    using RetType = decltype(f(args...));
    auto task = std::make_shared<std::packaged_task<RetType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<RetType> future = task->get_future();
    std::unique_lock<std::mutex> lk(mutex_);
    task_queue_.AddTask([task]() -> void { (*task)(); });
    cond_var_.notify_all();
    return future;
  }

  void Join() {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      // gDebugCol4(__PRETTY_FUNCTION__) << VAR(task_queue_.GetSize());
      std::unique_lock<std::mutex> look(mutex_);
      if (task_queue_.GetSize() == 0) {
        for (const auto &it : threads_) {
          if (it.second->GetState() == WorkThread::WORK) {
            continue;
          }
        }
        break;
      }
    }
  }

  uint64_t Count() { return threads_.size(); }

 private:
  void AddThread() {
    auto tdPtr = std::make_shared<WorkThread>(task_queue_, cond_var_, mutex_);
    threads_[tdPtr->GetId()] = tdPtr;
    gDebugCol1(__PRETTY_FUNCTION__) << tdPtr->GetId() << VAR(Count());
  }

  void DelThread() {
    std::thread::id pid;
    for (auto &it : threads_) {
      if (it.second->GetState() == WorkThread::WAIT) {
        it.second->Finish();
        pid = it.first;
        break;
      }
    }
    gDebugCol3(__PRETTY_FUNCTION__) << pid << VAR(Count());
    threads_.erase(pid);
    std::unique_lock<std::mutex> lock(mutex_);
    cond_var_.notify_all();
  }

  TaskQueue task_queue_;
  std::condition_variable cond_var_;
  std::mutex mutex_;
  std::thread manage_thread_;
  std::atomic_bool finished_;
  std::unordered_map<std::thread::id, ThreadPtr> threads_;
  int min_;
  int max_;
};
