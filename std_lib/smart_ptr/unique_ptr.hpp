#pragma once
#include "common.h"

GXT_NAMESPACE_BEGIN

template <typename T>
class unique_ptr {
 public:
  using pointer = T *;
  explicit unique_ptr(pointer ptr = nullptr) : ptr_(ptr) {}
  unique_ptr(unique_ptr &&ptr) noexcept { ptr_ = ptr.release(); }
  unique_ptr &operator=(unique_ptr &&ptr) noexcept {
    reset(ptr.release());
    return *this;
  }
  unique_ptr(const unique_ptr &) = delete;
  unique_ptr &operator=(const unique_ptr &) = delete;
  ~unique_ptr() { delete ptr_; }
  void reset(pointer p = nullptr) {
    auto old_ptr = ptr_;
    ptr_ = p;
    delete old_ptr;
  }
  pointer release() {
    auto p = ptr_;
    ptr_ = nullptr;
    return p;
  }
  pointer get() const { return ptr_; }
  T &operator*() const { return *ptr_; }
  pointer operator->() const { return ptr_; }
  explicit operator bool() const { return ptr_; }

 private:
  pointer ptr_;
};

GXT_NAMESPACE_END
