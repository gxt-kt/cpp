#include "range_based_for_loop.hpp"

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
class Test2 {
 public:
  Test2() = default;
  Test2(std::initializer_list<T> obj) {
    T* cur = str;
    for (const auto& val : obj) {
      *cur = val;
      ++cur;
    }
  }
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
  friend std::ostream& operator<<(std::ostream& os, Test2 obj) {
    for (const auto& val : obj) {
      os << val << " ";
    }
    return os;
  }
};

int main(int argc, char* argv[]) {
  {
    std::vector<int> aa(10);
    for (auto& val : aa) {
      val = gxt::Random(0, 100);
    }
    std::sort(aa.begin(), aa.end(), std::greater());
    gDebug(aa);
  }

  {
    std::unordered_map<int, std::string> aa;
    aa.insert({10, "123"});
    aa.insert({20, "456"});
    aa.insert({30, "789"});
    for (const auto& val : aa) {
      gDebug(val.first) << gDebug(val.second);
    }
  }

  {
    Test<10> test;
    for (auto& val : test) {
      val = gxt::Random(0, 100);
      // gDebug(val);
    }
    gDebug() << G_SPLIT_LINE;
    for (auto& val : test) {
      gDebug(val);
    }
    gDebug() << G_SPLIT_LINE;
    gDebug() << test;
  }
  {
    gDebugCol3() << G_SPLIT_LINE;
    const Test<10> test{12, 45, 123, 12};
    for (auto& val : test) {
      gDebug(val);
    }
    gDebugCol3() << G_SPLIT_LINE;
    gDebug() << test;
  }
  { __gnu_cxx::__normal_iterator<const int*, std::vector<int>> const_iterator; }
  gDebugCol5() << G_SPLIT_LINE;
  {
    Test2<int, 10> test{1, 2, 3, 4, 5};
    for (auto it = test.begin(); it != test.end(); ++it) {
      gDebug(*it);
    }
    gDebug() << G_SPLIT_LINE;
    for (auto& val : test) {
      gDebug(val);
    }
    gDebug() << G_SPLIT_LINE;
    gDebug() << test;
  }
}
