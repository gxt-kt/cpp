#pragma once

#include <iostream>

class PluginBase {
 public:
  PluginBase() {}
  virtual ~PluginBase() {}
  virtual void Init() = 0;
  virtual void Execute() = 0;
  virtual void Release() = 0;
  virtual void SetValue(int a, int b) { a_ = a, b_ = b; };
  virtual int GetResult() = 0;

 protected:
  int a_;
  int b_;
};
