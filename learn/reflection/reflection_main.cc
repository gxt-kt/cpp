#include "reflection.hpp"

class TestClassA {
 public:
  void print() {
    gDebug(__PRETTY_FUNCTION__);
    gDebug(a);
  }
  int a = 10;
};

inline void* CreateTestClassA() { return new TestClassA(); }

RegistAction regist_a("TestClassA", CreateTestClassA);

int main(int argc, char* argv[]) {
  TestClassA* a = static_cast<TestClassA*>(
      ClassFactory::GetInstance().GetClassByName("TestClassA"));
  a->print();
  return 0;
}
