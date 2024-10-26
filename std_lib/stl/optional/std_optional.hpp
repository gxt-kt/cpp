#pragma once
#include "common.h"

/**
 *
 * Ref: https://github.com/parallel101/stl1weekend/blob/main/Optional.hpp
 *      https://www.bilibili.com/video/BV1v6421Z7f8
 *
 * 教程里实现了比较完全的optional，主要包含了一些c++23的api
 * 在这个源码中我把这些删了，只保留了c++17库中有的api
 *
 * 实现：
 *   其实内部成员变量就两个，一个是布尔类型表示是否有有效值，另一个就是实际值
 *   这里的值value采用union的方式存储，可以惰性初始化，和存储T*指针的思路不一样
 *   剩下的就比较简单了，就是正常的思路
 *
 */

// 用来抛出异常
struct BadOptionalAccess : std::exception {
  BadOptionalAccess() = default;
  virtual ~BadOptionalAccess() = default;

  const char *what() const noexcept override { return "BadOptionalAccess"; }
};

// 类似于实现一个nullptr，只是这个是自己定义的类
struct Nullopt {
  explicit Nullopt() = default;
};
// 全局变量 nullopt类似于nullptr
constexpr Nullopt nullopt;

// 用一个空的占位符用来区分不同的构造函数
struct InPlace {
  explicit InPlace() = default;
};
constexpr InPlace inPlace;

template <class T>
struct Optional {
 private:
  bool m_has_value;
  union {
    T m_value;
  };

 public:
  Optional(T &&value) noexcept : m_has_value(true), m_value(std::move(value)) {}

  Optional(T const &value) noexcept
      : m_has_value(true), m_value(std::move(value)) {}

  Optional() noexcept : m_has_value(false) {}

  Optional(Nullopt) noexcept : m_has_value(false) {}

  template <class... Ts>
  explicit Optional(InPlace, Ts &&...value_args)
      : m_has_value(true), m_value(std::forward<Ts>(value_args)...) {}

  template <class U, class... Ts>
  explicit Optional(InPlace, std::initializer_list<U> ilist, Ts &&...value_args)
      : m_has_value(true), m_value(ilist, std::forward<Ts>(value_args)...) {}

  Optional(Optional const &that) : m_has_value(that.m_has_value) {
    if (m_has_value) {
      new (&m_value) T(that.m_value);  // placement-new（不分配内存，只是构造）
      // m_value = T(that.m_value); // m_value.operator=(T const &);
    }
  }

  Optional(Optional &&that) noexcept : m_has_value(that.m_has_value) {
    if (m_has_value) {
      new (&m_value) T(std::move(that.m_value));
    }
  }

  Optional &operator=(Nullopt) noexcept {
    if (m_has_value) {
      m_value.~T();
      m_has_value = false;
    }
    return *this;
  }

  Optional &operator=(T &&value) noexcept {
    if (m_has_value) {
      m_value.~T();
      m_has_value = false;
    }
    new (&m_value) T(std::move(value));
    m_has_value = true;
    return *this;
  }

  Optional &operator=(T const &value) noexcept {
    if (m_has_value) {
      m_value.~T();
      m_has_value = false;
    }
    new (&m_value) T(value);
    m_has_value = true;
    return *this;
  }

  Optional &operator=(Optional const &that) {
    if (this == &that) return *this;

    if (m_has_value) {
      m_value.~T();
      m_has_value = false;
    }
    if (that.m_has_value) {
      new (&m_value) T(that.m_value);
    }
    m_has_value = that.m_has_value;
    return *this;
  }

  Optional &operator=(Optional &&that) noexcept {
    if (this == &that) return *this;

    if (m_has_value) {
      m_value.~T();
      m_has_value = false;
    }
    if (that.m_has_value) {
      new (&m_value) T(std::move(that.m_value));
      that.m_value.~T();
    }
    m_has_value = that.m_has_value;
    that.m_has_value = false;
    return *this;
  }

  template <class... Ts>
  void emplace(Ts &&...value_args) {
    if (m_has_value) {
      m_value.~T();
      m_has_value = false;
    }
    new (&m_value) T(std::forward<Ts>(value_args)...);
    m_has_value = true;
  }

  template <class U, class... Ts>
  void emplace(std::initializer_list<U> ilist, Ts &&...value_args) {
    if (m_has_value) {
      m_value.~T();
      m_has_value = false;
    }
    new (&m_value) T(ilist, std::forward<Ts>(value_args)...);
    m_has_value = true;
  }

  void reset() noexcept {  // 等价于 *this = nullopt;
    if (m_has_value) {
      m_value.~T();
      m_has_value = false;
    }
  }

  ~Optional() noexcept {
    if (m_has_value) {
      m_value.~T();  // placement-delete（不释放内存，只是析构）
    }
  }

  bool has_value() const noexcept { return m_has_value; }

  explicit operator bool() const noexcept { return m_has_value; }

  bool operator==(Nullopt) const noexcept { return !m_has_value; }

  friend bool operator==(Nullopt, Optional const &self) noexcept {
    return !self.m_has_value;
  }

  bool operator!=(Nullopt) const noexcept { return m_has_value; }

  friend bool operator!=(Nullopt, Optional const &self) noexcept {
    return self.m_has_value;
  }

  T const &value() const & {
    if (!m_has_value) throw BadOptionalAccess();
    return m_value;
  }

  T &value() & {
    if (!m_has_value) throw BadOptionalAccess();
    return m_value;
  }

  T const &&value() const && {
    if (!m_has_value) throw BadOptionalAccess();
    return std::move(m_value);
  }

  T &&value() && {
    if (!m_has_value) throw BadOptionalAccess();
    return std::move(m_value);
  }

  T const &operator*() const & noexcept { return m_value; }

  T &operator*() & noexcept { return m_value; }

  T const &&operator*() const && noexcept { return std::move(m_value); }

  T &&operator*() && noexcept { return std::move(m_value); }

  T const *operator->() const noexcept { return &m_value; }

  T *operator->() noexcept { return &m_value; }

  T value_or(T default_value) const & {
    if (!m_has_value) return default_value;
    return m_value;
  }

  T value_or(T default_value) && noexcept {
    if (!m_has_value) return default_value;
    return std::move(m_value);
  }

  bool operator==(Optional<T> const &that) const noexcept {
    if (m_has_value != that.m_has_value) return false;
    if (m_has_value) {
      return m_value == that.m_value;
    }
    return true;
  }

  bool operator!=(Optional const &that) const noexcept {
    if (m_has_value != that.m_has_value) return true;
    if (m_has_value) {
      return m_value != that.m_value;
    }
    return false;
  }

  bool operator>(Optional const &that) const noexcept {
    if (!m_has_value || !that.m_has_value) return false;
    return m_value > that.m_value;
  }

  bool operator<(Optional const &that) const noexcept {
    if (!m_has_value || !that.m_has_value) return false;
    return m_value < that.m_value;
  }

  bool operator>=(Optional const &that) const noexcept {
    if (!m_has_value || !that.m_has_value) return true;
    return m_value >= that.m_value;
  }

  bool operator<=(Optional const &that) const noexcept {
    if (!m_has_value || !that.m_has_value) return true;
    return m_value <= that.m_value;
  }

  void swap(Optional &that) noexcept {
    if (m_has_value && that.m_has_value) {
      using std::swap;  // ADL
      swap(m_value, that.m_value);
    } else if (!m_has_value && !that.m_has_value) {
      // do nothing
    } else if (m_has_value) {
      that.emplace(std::move(m_value));
      reset();
    } else {
      emplace(std::move(that.m_value));
      that.reset();
    }
  }
};

// C++17 才有 CTAD
template <class T>
Optional(T) -> Optional<T>;

template <class T>
Optional<T> makeOptional(T value) {
  return Optional<T>(std::move(value));
}
