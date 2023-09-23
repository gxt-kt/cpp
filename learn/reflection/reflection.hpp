#pragma once

// Ref : https://blog.csdn.net/K346K346/article/details/51698184

#include "common.h"

using PTRCreateObject = void* (*)();

class ClassFactory {
 private:
  std::map<std::string, PTRCreateObject> class_map_;
  ClassFactory() = default;

 public:
  static ClassFactory& GetInstance() {
    static ClassFactory class_factory;
    return class_factory;
  }
  void* GetClassByName(std::string class_name) {
    auto it = class_map_.find(class_name);
    if (it == class_map_.end()) {
      return nullptr;
    } else {
      return it->second();  // NOTE: instante is here :()
    }
  }
  void RegistClass(std::string class_name, PTRCreateObject method) {
    class_map_.insert({class_name, method});
  }
};

class RegistAction {
 public:
  RegistAction(std::string class_name, PTRCreateObject ptr) {
    ClassFactory::GetInstance().RegistClass(class_name, ptr);
  }
};
