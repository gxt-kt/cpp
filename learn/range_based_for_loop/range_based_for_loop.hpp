#pragma once

#include "common.h"

// range based for loop pseudocode
//
// ```cpp
// auto && __range = range_expression;
// auto __begin = begin_expr;
// auto __end = end_expr;
// for (; __begin != __end; ++__begin) {
//   range_declaration = *__begin;
//   loop_statement
// }
// ```
//

/*
 * 观察伪代码，关于迭代器用到的运算符，用到了 * , != 和 前置++ ,
 *
 * 所以我们需要对迭代器实现以下的内容
 * 1. *
 * 2. !=
 * 3. ++ (prefix++)
 *
 * 然后根据c++标准（编译器）规定，我们还需要实现begin和end函数，返回对应的迭代器
 *
 */

template <typename _Iterator>
class MyIterator {
 protected:
  _Iterator _M_current;

  typedef std::iterator_traits<_Iterator> __traits_type;

  // just use std::iterator_traits to trait the types
 public:
  typedef _Iterator iterator_type;
  typedef typename __traits_type::iterator_category iterator_category;
  typedef typename __traits_type::value_type value_type;
  typedef typename __traits_type::difference_type difference_type;
  typedef typename __traits_type::reference reference;
  typedef typename __traits_type::pointer pointer;

 public:
  MyIterator() noexcept : _M_current(_Iterator()) {}

  explicit MyIterator(const _Iterator& __i) noexcept : _M_current(__i) {}

  reference operator*() const noexcept { return *_M_current; }
  MyIterator& operator++() noexcept {
    ++_M_current;
    return *this;
  }

  // not need base function. Just to expose _M_current
  const _Iterator& base() const noexcept { return _M_current; }
};

// deal with different types of left and right != operators
template <typename _IteratorL, typename _IteratorR>
[[__nodiscard__]] inline bool operator!=(
    const MyIterator<_IteratorL>& __lhs,
    const MyIterator<_IteratorR>& __rhs) noexcept {
  return __lhs.base() != __rhs.base();
}

// deal with same types of left and right != operators
template <typename _Iterator>
[[__nodiscard__]] inline bool operator!=(
    const MyIterator<_Iterator>& __lhs,
    const MyIterator<_Iterator>& __rhs) noexcept {
  return __lhs.base() != __rhs.base();
}

template <typename T, size_t N>
class Test {
 public:
  Test() = default;
  Test(std::initializer_list<T> obj) {
    T* cur = str;
    for (const auto& val : obj) {
      *cur = val;
      ++cur;
    }
  }
  T* data() {return str;}
  const T* data() const { return str; }

  /*
   * And the data class must impl begin and end function
   * to return the correspond iterator
   * */
  using iterator = MyIterator<T*>;
  using const_iterator = MyIterator<const T*>;

 public:
  iterator begin() { return iterator(str); }
  iterator end() { return iterator(str + N); }
  const_iterator begin() const { return const_iterator(str); }
  const_iterator end() const { return const_iterator(str + N); }

 private:
  T str[N]{};

 public:
  friend std::ostream& operator<<(std::ostream& os, const Test& obj) {
    for (const auto& val : obj) {
      os << val << " ";
    }
    return os;
  }
};


// subtitute begin and end function in class Test
// If we cannot chang a class source code to insert begin and end inner the class.
// We can use global begin and end like the following
#if 0
template <typename T, size_t N>
auto begin(Test<T, N>& a) {
  return MyIterator<T*>(a.data());
}
template <typename T, size_t N>
auto end(Test<T, N>& a) {
  return MyIterator<T*>(a.data() + N);
}
template <typename T, size_t N>
auto begin(const Test<T, N>& a) {
  return MyIterator<const T*>(a.data());
}
template <typename T, size_t N>
auto end(const Test<T, N>& a) {
  return MyIterator<const T*>(a.data() + N);
}
#endif
