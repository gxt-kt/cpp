#pragma once

#include "common.h"

template <class T>
class CycleQueue {
 private:
  size_t size_;
  size_t front_;
  size_t tail_;
  T* data_;

 public:
  CycleQueue(size_t size) : size_(size), front_(0), tail_(0) {
    data_ = new T[size];
  }

  ~CycleQueue() { delete[] data_; }

  bool Empty() { return front_ == tail_; }

  bool Full() { return front_ == (tail_ + 1) % size_; }

  void Push(T ele) {
    if (Full()) {
      throw std::bad_exception();
    }
    data_[tail_] = ele;
    tail_ = (tail_ + 1) % size_;
  }

  T Pop() {
    if (Empty()) {
      throw std::bad_exception();
    }
    T tmp = data_[front_];
    front_ = (front_ + 1) % size_;
    return tmp;
  }
};
