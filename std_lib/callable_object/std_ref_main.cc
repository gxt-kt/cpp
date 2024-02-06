#include "callable_objects.hpp"

using namespace std::placeholders;

void Fun(int a) {
  a++;
}
void Fun1(int& a) {
  a++;
}

int main(int argc, char* argv[]) {
  gDebug() << "exec" << __FILE__;
  int a = 10;
  {
    std::function<void()> f = std::bind(Fun, a);    // default is pass by value
    std::function<void()> f1 = std::bind(Fun1, a);  // default is pass by value
    f();
    gDebug(a);
    f1();
    gDebug(a);
  }
  {
    std::function<void()> f = std::bind(Fun, std::ref(a));  // use ref to pass by ref
    std::function<void()> f1 = std::bind(Fun1, std::ref(a));  // use ref to pass by ref
    f();
    gDebug(a);
    f1();
    gDebug(a);
  }
  return 0;
}
