#include "fun_ptr.hpp"

void Fun() { gDebugCol1() << __PRETTY_FUNCTION__; }

struct Test {
  void Fun() { gDebugCol2() << __PRETTY_FUNCTION__; }
  void Fun2() { gDebugCol2() << __PRETTY_FUNCTION__; }
  static void FunStatic(void) { gDebugCol3() << __PRETTY_FUNCTION__; }
};

int main(int argc, char* argv[]) {
  Test test;

  gDebug() << G_SPLIT_LINE;

  Fun();
  test.Fun();
  Test::FunStatic();

  gDebug() << G_SPLIT_LINE;

  Fun();
  (&Fun)();
  (*Fun)();
  (*(&Fun))();
  (*(*Fun))();
  (*(*(*Fun)))();
  (&(*(&Fun)))();
  (*(&(*(&Fun))))();

  gDebug() << G_SPLIT_LINE;

  gDebug(G_TYPET(decltype(Fun)));
  gDebug(G_TYPET(decltype(&Fun)));
  // gDebug(G_TYPET(decltype(&(&Fun)))); // error
  gDebug(G_TYPET(decltype(*Fun)));
  gDebug(G_TYPET(decltype(*(*Fun))));
  gDebug(G_TYPET(decltype(&(*Fun))));
  gDebug(G_TYPET(decltype((*(&Fun)))));
  gDebug(G_TYPET(decltype((*(&(*(&Fun)))))));

  gDebug() << G_SPLIT_LINE;

  {
    decltype(Fun) a;  // hasn't set value so cannot use
    // a(); // error

    decltype(&Fun) b1 = Fun;
    decltype(&Fun) b2 = *Fun;
    decltype(&Fun) b3 = &Fun;

    b1();
    b2();
    b3();

    decltype(*Fun) c1 = Fun;
    // decltype(*Fun) c2 = &Fun; // error
    decltype(*Fun) c3 = *Fun;
  }

  gDebug() << G_SPLIT_LINE;

  gDebugCol5() << "Test::FunStatic is like Fun";

  {
    gDebug(G_TYPET(decltype(Test::FunStatic)));
    gDebug(G_TYPET(decltype(&Test::FunStatic)));
    // gDebug(G_TYPET(decltype(&(&Test::FunStatic)))); // error
    gDebug(G_TYPET(decltype(*Test::FunStatic)));
    gDebug(G_TYPET(decltype(*(*Test::FunStatic))));
    gDebug(G_TYPET(decltype(&(*Test::FunStatic))));
    gDebug(G_TYPET(decltype((*(&Test::FunStatic)))));
    gDebug(G_TYPET(decltype((*(&(*(&Test::FunStatic)))))));
  }

  gDebug() << G_SPLIT_LINE;

  {
    // gDebug(G_TYPET(decltype(Test::Fun))); //error
    gDebug(G_TYPET(decltype(&Test::Fun)));
    decltype(&Test::Fun) b;
  }
}
