#pragma once

#include "plugin_base.hpp"

class PluginDerived : public PluginBase {
 public:
  PluginDerived() { std::cout << __PRETTY_FUNCTION__ << std::endl; }

  virtual void Init() override{};
  virtual void Execute() override;
  virtual void Release() override{};
  virtual int GetResult() override;

 private:
  int AddImpl();
  int res{0};
};

// 创建插件实例的函数声明
extern "C" PluginBase* CreatePlugin();
