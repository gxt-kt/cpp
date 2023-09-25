#pragma once

#include "common.h"

template <typename Derived>
class Base {
 public:
  void interface() { static_cast<Derived*>(this)->implementation(); }
};

class Derived1 : public Base<Derived1> {
 public:
  void implementation() { gDebug() << __PRETTY_FUNCTION__; }
};

class Derived2 : public Base<Derived2> {
 public:
  void implementation() { gDebug() << __PRETTY_FUNCTION__; }
};
