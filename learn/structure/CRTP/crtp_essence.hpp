#pragma once

#include "common.h"

template <typename Derived>
class Base {
 public:
  void interface() {
    // 这里的this等不等于子类的地址取决于父类（当前类）有没有占用空间(sizeof)
    gDebug(this);

    // NOTE: 不管上面结果如何，通过static_cast并使用指针或者引用转换后
    // 转换后的地址都会是子类的地址
    gDebug(static_cast<Derived*>(this));
    gDebug(&static_cast<Derived&>(*this));

    // error cannot convert class directly
    // 父类没法直接转给子类，只能通过指针或者引用
    // gDebug(static_cast<Derived>(*this));

    // 既然转换后的地址都是子类地址了，那么调用对应函数也会是子类的函数了
    static_cast<Derived*>(this)->implementation();
    // 和上面效果一样
    // static_cast<Derived&>(*this).implementation();
  }
  int a;  // 如果定义了a，父类的地址就是a的起始地址，如果没有a,父类地址就是子类地址
};

struct PlaceHolder1 {
  int placeholder1_1 = 1;
  int placeholder1_2 = 1;
};
struct PlaceHolder2 {
  int placeholder2_1 = 2;
  int placeholder2_2 = 2;
};

class Derived : public PlaceHolder1, public Base<Derived>, public PlaceHolder2 {
 public:
  void implementation() { gDebug() << __PRETTY_FUNCTION__; }
};
