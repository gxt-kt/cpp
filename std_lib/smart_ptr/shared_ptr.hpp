#pragma once
#include "common.h"

GXT_NAMESPACE_BEGIN

class shared_count {
 public:
  shared_count() : count_(1) {}
  void add_count() { count_++; }
  int reduce_count() { return --count_; }
  int get_count() const { return count_; }

 private:
  int count_;
};

template <typename T>
class shared_ptr;
template <typename T>
class weak_ptr;

template <typename T>
class shared_ptr {
 private:
  template <typename U>
  void move_from(shared_ptr<U> &&rhs) noexcept {
    ptr_ = rhs.ptr_;
    if (ptr_) {
      rhs.ptr_ = nullptr;
      shared_count_ = rhs.shared_count_;
    }
  }
  template <typename U>
  void init(const shared_ptr<U> &rhs) noexcept {
    ptr_ = rhs.ptr_;
    if (ptr_) {
      shared_count_ = rhs.shared_count_;
      shared_count_->add_count();
    }
  }

 public:
  using pointer = T *;

  explicit shared_ptr(pointer ptr = nullptr) : ptr_(ptr) {
    if (ptr_) {
      shared_count_ = new shared_count();
    }
  }
  shared_ptr(const shared_ptr &rhs) { init(rhs); }
  shared_ptr(shared_ptr &&rhs) noexcept { move_from(std::move(rhs)); }

  template <typename U>
  friend class shared_ptr;

  template <typename U>
  friend class weak_ptr;

  template <typename U>
  explicit shared_ptr(const shared_ptr<U> &rhs) {
    init(rhs);
  }
  template <typename U>
  explicit shared_ptr(shared_ptr<U> &&rhs) {
    move_from(std::move(rhs));
  }

  template <typename U>
  explicit shared_ptr(const shared_ptr<U> &rhs, pointer ptr) {
    ptr_ = ptr;
    if (ptr_) {
      shared_count_ = rhs.shared_count_;
      shared_count_->add_count();
    }
  }

  void swap(shared_ptr &rhs) {
    using std::swap;
    swap(rhs.ptr_, ptr_);
    swap(rhs.shared_count_, shared_count_);
  }

  shared_ptr &operator=(shared_ptr rhs) {
    rhs.swap(*this);
    return *this;
  }
  template <typename U>
  shared_ptr &operator=(shared_ptr<U> rhs) {
    shared_ptr<T>(std::move(rhs)).swap(*this);
    return *this;
  }

  ~shared_ptr() {
    if (ptr_ && !shared_count_->reduce_count()) {
      delete ptr_;
      delete shared_count_;
      ptr_ = pointer();
    }
  }

  T &operator*() const { return *ptr_; }
  pointer operator->() const { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }
  pointer get() const { return ptr_; }
  int use_count() const {
    if (!ptr_) {
      return 0;
    }
    return shared_count_->get_count();
  }

 private:
  pointer ptr_ = nullptr;
  shared_count *shared_count_ = nullptr;
};

template <typename T, typename U>
shared_ptr<T> dynamic_pointer_cast(const shared_ptr<U> &other) {
  auto ptr = dynamic_cast<T *>(other.get());
  return shared_ptr<T>(other, ptr);
}

template <typename T>
class weak_ptr {
 public:
  weak_ptr() {}
  template <typename U>
  weak_ptr(const shared_ptr<U> &x)
      : ptr_(x.ptr_), shared_count_(x.shared_count_) {}
  weak_ptr(const weak_ptr &obj)
      : ptr_(obj.ptr_), shared_count_(obj.shared_count_) {}
  weak_ptr(weak_ptr &&obj) {
    ptr_ = obj.ptr_;
    shared_count_ = obj.shared_count_;
    obj.ptr_ = nullptr;
    obj.shared_count_ = nullptr;
  }
  weak_ptr<T> &operator=(const weak_ptr<T> &obj) {
    if (&obj == this) {
      return *this;
    }
    ptr_ = obj.ptr_;
    shared_count_ = obj.shared_count_;
    return *this;
  }
  weak_ptr<T> &operator=(weak_ptr<T> &&obj) {
    if (&obj == this) {
      return *this;
    }
    ptr_ = obj.ptr_;
    shared_count_ = obj.shared_count_;
    obj.ptr_ == nullptr;
    obj.shared_count_ == nullptr;
    return *this;
  }
  ~weak_ptr() {
    ptr_ = nullptr;
    shared_count_ = nullptr;
  }
  void reset() {
    ptr_ = nullptr;
    shared_count_ = nullptr;
  }
  bool expired() const {
    return ptr_ == nullptr || shared_count_->get_count() == 0;
  }
  shared_ptr<T> lock() const {
    if (expired()) {
      return shared_ptr<T>();
    }
    shared_ptr<T> res;
    res.ptr_ = ptr_;
    res.shared_count_ = shared_count_;
    if (res.shared_count_ != nullptr) {
      res.shared_count_->add_count();
    }
    return res;
  }
  void swap(weak_ptr<T> &obj) {
    using std::swap;
    swap(obj.ptr_, ptr_);
    swap(obj.shared_count_, shared_count_);
  }
  int use_count() const {
    if (ptr_ == nullptr) {
      return 0;
    } else {
      return shared_count_->get_count();
    }
  }
  weak_ptr<T> &operator=(const shared_ptr<T> &obj) {
    ptr_ = obj.ptr_;
    shared_count_ = obj.shared_count_;
    return *this;
  }
  using pointer = T *;
  pointer ptr_ = nullptr;
  shared_count *shared_count_ = nullptr;
};

GXT_NAMESPACE_END
