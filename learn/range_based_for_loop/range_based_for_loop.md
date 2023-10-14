# 自己的类支持基于范围的for循环 (深入探索)

## 编译器实际运行伪代码为:

```cpp
auto && __range = range_expression;
auto __begin = begin_expr;
auto __end = end_expr;
for (; __begin != __end; ++__begin) {
  range_declaration = *__begin;
  loop_statement
}
```


观察伪代码，关于迭代器用到的运算符，用到了 `* , != 和 前置++ `

所以我们需要对迭代器实现以下的内容
1. \*
2. !=
3. ++ (prefix++)

然后根据c++标准（编译器）规定，我们还需要对对应的类实现begin和end函数，返回对应的迭代器

## 实现



#### 迭代器类的实现


```cpp
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
```

需要💡几点：
- 这里的const都不是乱加的，都是有讲究的
- 这里用到了标准库的迭代器类型萃取，没啥难度，但是对应特化版本写起来很麻烦，我这就不重新写了，理解上也很简单
- 这里写了一个base函数，不是强制要求的，在这里也只是为了暴露指针，让全局的`重载!=`可以访问到，当然把指针设成`public`也可以
- 重载的`!=`有两个版本，分别对应`!=`左右类型是否一致的情况

---


#### 类的begin和end实现

实际上，我们只需要把下面这段代码插入到我们要的类里就可以了

```cpp

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
```

这里的str和N换成你对应要的内容

当然这里只针对顺序容器，如果是其它类型，会复杂一点，迭代器也需要重新设计，就需要第二个参数了

标准库中普通顺序容器模板也是有两个，但是因为顺序可以直接对指针进行++，第二个用处不大，我这里也就直接省略了

## 示例

**我这里给了一个最基本的使用demo示例，仅供参考：**

```cpp
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
```

**使用示例：**

```cpp
  {
    Test<int, 10> test;
    for (auto& val : test) {
      static int i = 0;
      val = ++i;
    }
    std::cout << test << std::endl;
  }

  {
    const Test<int, 10> test{12, 34, 56, 78, 910};
    for (const auto& val : test) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }
```

**输出：**

```plaintext
12 45 123 12 0 0 0 0 0 0  
1 2 3 4 5 6 7 8 9 10 
12 34 56 78 910 0 0 0 0 0 
```


## 小问题：如果一个类代码已经写好了，我们没法插入自己的begin函数和end函数怎么办


那么c++编译器在看你这个类如果没有begin和end函数，就回去查找全局的begin和end

那么全局的begin和end应该怎么写呢？

我是这么实现的，也顺利通过了，左右值/const也都走了对应版本

```cpp
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
```

但怎么说呢，我觉得应该有更优雅的写法或者规范，但是这一块没有参考

不像前面写迭代器可以看标准库，保证不出错，但是这个标准库里肯定也没有示例，就不好说了

如果有谁看到有官方示例或推荐，欢迎来联系我
