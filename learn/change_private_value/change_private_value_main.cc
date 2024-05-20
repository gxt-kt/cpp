#include "common.h"

// Ref:
// https://www.bilibili.com/video/BV11x4y1i7ed/?spm_id_from=333.999.0.0&vd_source=01da08e4487b8e450cf16063029887c6
// C++中使用模板去修改类的私有成员
// 主要利用了模板实例化时不会检查类成员的访问属性

class Test {
 public:
  void PrintValue() { std::cout << value << std::endl; }

 private:
  int value = 0;
};

// change_value是模板参数，类型为指向Test的成员变量的指针
template <int Test::*change_value>
struct Change {
  // 友元，为了获取私有变量
  friend int& test(Test& obj) { return obj.*change_value; }
};

// 在外面声明这个友元函数使用
int& test(Test& obj);

// 模板实例化，指明模板参数为Test的value变量指针
template struct Change<&Test::value>;

int main(int argc, char* argv[]) {
  Test tmp;
  tmp.PrintValue();
  test(tmp) = 1000;
  tmp.PrintValue();
  return 0;
}
