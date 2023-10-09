#include "std_move_forward.hpp"

template <typename T>
void FunT(T&& a) {
  gDebugCol1(TYPET(T));
  gDebugCol1(TYPE(GXT_NAMESPACE::forward<T>(a)));
}


int main(int argc, char* argv[]) {
  gDebug() << G_FILE_LINE;

  gDebug() << G_SPLIT_LINE;
  {
    int a;
    FunT(a);
    FunT(10);
    FunT(GXT_NAMESPACE::move(a));
  }
  gDebug() << G_SPLIT_LINE;
  {
    int t;
    auto&& a = t;
    auto&& b = 10;
    auto&& c = GXT_NAMESPACE::move(t);
    gDebugCol2(TYPE(t));
    gDebugCol2(TYPE(a));
    gDebugCol2(TYPE(b));
    gDebugCol2(TYPE(c));
  }
  {
    auto fun1=[]()->decltype(auto){ int a; return a; };
    auto fun2=[]()->decltype(auto){ return 10; };
    auto fun3=[]()->decltype(auto){ int a; return std::move(a); };
    gDebugCol3(TYPE(fun1()));
    gDebugCol3(TYPE(fun2()));
    gDebugCol3(TYPE(fun3()));
    // add () for return value with decltype(auto)
    auto fun4=[]()->decltype(auto){ int a; return (a); };
    auto fun5=[]()->decltype(auto){ return (10); };
    auto fun6=[]()->decltype(auto){ int a; return (std::move(a)); };
    gDebugCol3(TYPE(fun4()));
    gDebugCol3(TYPE(fun5()));
    gDebugCol3(TYPE(fun6()));
  }
}
