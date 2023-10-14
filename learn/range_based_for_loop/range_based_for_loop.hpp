#pragma once

#include "common.h"

// a iterator class must impel three function
// 1. !=
// 2. ++ (prefix++)
// 3. *

/*
 * And the data class must impl begin and end function
 * to return the correspond iterator
 * */



class MyIterTest {
 public:
  MyIterTest(int* p) : p_(p) {}
  bool operator!=(const MyIterTest& obj) { return p_ != (obj.p_); };
  MyIterTest& operator++() {
    p_++;
    return *this;
  };
  int& operator*() { return *p_; };
  int* p_;
};

class MyIterTestConst {
 public:
  MyIterTestConst(const int* p) : p_(p) {}
  bool operator!=(const MyIterTestConst& obj) { return p_ != (obj.p_); };
  MyIterTestConst& operator++() {
    ++p_;
    return *this;
  }
  const int& operator*() const { return *p_; }
  const int* p_;
};

template <size_t N>
class Test {
 public:
  Test() = default;
  Test(std::initializer_list<int> obj) {
    int* cur = str;
    for (const auto& val : obj) {
      *cur = val;
      ++cur;
    }
  }
  using iterator = MyIterTest;
  using const_iterator = MyIterTestConst;

 public:
  iterator begin() { return iterator(str); }
  iterator end() { return iterator(str + N); }
  const_iterator begin() const { return const_iterator(str); }
  const_iterator end() const { return const_iterator(str + N); }

 private:
  int str[N]{0};

 public:
  friend std::ostream& operator<<(std::ostream& os, Test obj) {
    for (const auto& val : obj) {
      os << val << " ";
    }
    return os;
  }
};
