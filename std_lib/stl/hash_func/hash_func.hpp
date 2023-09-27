#pragma once
#include "common.h"

// NOTE: use user type for unorderd_map/unorderd_set's key need two function
// 1. Hash function : TestHashFunc(const Test& obj) const // must use const
// 2. == function : to judge whether two objects are the same

// hash 函数的计算结果决定了挂在哪个篮子上：
// 哈希值计算出后对篮子总数取余就是挂在哪个篮子
// 如果当前存储的键值对总数超过篮子数篮子就会进行扩充，并且会重新计算每个的篮子
// hash 函数设计上就是无明显规则略乱略好，倒也不一定非要保证不会重复

// == 函数是用来判断两个key是否相等，相等的key肯定是不会发生重复的

// Ref : https://blog.csdn.net/y109y/article/details/82669620

struct Test {
  int a;
  float b;
  std::string c;
  bool operator==(const Test& obj) const { return this->b == obj.b; }
};

struct TestHashFunc {
  size_t operator()(const Test& obj) const {
    return static_cast<size_t>(obj.a);
  }
};

struct Standard {
  struct Involve {
    float involve;
  };
  int a;
  Involve b;
  std::string c;
  // == type1
  // bool operator==(const Standard& obj) const {
  //   return a == obj.a && b.involve == obj.b.involve && c == obj.c;
  // }
};
// == type2 // actually like type1
inline bool operator==(const Standard& obj1, const Standard& obj2) {
  return obj1.a == obj2.a && obj1.b.involve == obj2.b.involve &&
         obj1.c == obj2.c;
}

// == type3
inline bool My_equal_to(const Standard& obj1, const Standard& obj2) {
  return obj1.a == obj2.a && obj1.b.involve == obj2.b.involve &&
         obj1.c == obj2.c;
}

// == type4
namespace std {
template <>
struct equal_to<Standard> {
  bool operator()(const Standard& obj1, const Standard& obj2) const {
    return obj1.a == obj2.a && obj1.b.involve == obj2.b.involve &&
           obj1.c == obj2.c;
  }
};
}  // namespace std

// hash type1
struct StandardHashFunc1 {
  size_t operator()(const Standard& obj) const {
    return std::hash<int>()(obj.a) + std::hash<float>()(obj.b.involve) -
           std::hash<std::string>()(obj.c) * 3.14;
  }
};

// hash type2
inline size_t StandardHashFunc2(const Standard& obj) {
  return std::hash<int>()(obj.a) + std::hash<float>()(obj.b.involve) -
         std::hash<std::string>()(obj.c) * 3.14;
}

// hash type3
namespace std {
template <>
struct hash<Standard> {
  size_t operator()(const Standard& obj) {
    return std::hash<int>()(obj.a) + std::hash<float>()(obj.b.involve) -
           std::hash<std::string>()(obj.c) * 3.14;
  }
};
}  // namespace std
