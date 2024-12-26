#include "decltype.hpp"
// #include "decltype.hpp"

// #define SEE(a) gDebug(G_TYPET(decltype(a)))
#define SEE(a) gDebug(G_TYPE(a))
// #define SEE(a) decltype(a)
//
#define TYPELR(a) gDebug(G_TYPE(a))

struct A {
  void Fun() {}
};
struct B : public A {
  void Fun() {}
};

std::string Fun() { return std::string("11"); }
std::string Fun2() {
  std::string a("11");
  return a;
  // return std::string("11");
}

// FIXME: has problem when use on "string"
// because & isn't the last character
char TypeLR(const std::string& str) {
  // gDebug(str);
  auto size = str.size();
  if (size >= 2) {
    if (str.at(size - 1) == '&' && str.at(size - 2) == '&') {
      return 'x';
    } else if (str.at(size - 1) == '&') {
      return 'l';
    } else {
      return 'p';
    }
  } else if (size >= 1) {
    if (str.at(size - 1) == '&') {
      return 'l';
    } else {
      return 'p';
    }
  } else {
    return 'p';
  }
}

int main(int argc, char* argv[]) {
  int a = 0;
  // Test(a);
  SEE(a);
  SEE(+a);
  SEE(a + 0);
  SEE((a));
  SEE(std::move(a));
  SEE(1);
  {
    const char* p = "123";
    SEE(p);
    SEE((p));
  }

  gDebug(TypeLR(G_TYPET(decltype((a)))));
  gDebug(TypeLR(G_TYPET(decltype((1)))));
  gDebug(TypeLR(G_TYPET(decltype((std::string())))));
  gDebug(TypeLR(G_TYPET(decltype((Fun())))));
  gDebug(TypeLR(G_TYPET(decltype((Fun2())))));
  gDebug(TypeLR(G_TYPET(decltype((std::move(a))))));
  gDebug(TypeLR(G_TYPET(decltype((gxt::TestClass())))));
  gDebug(TypeLR(G_TYPET(decltype(("123")))));
  gDebug(G_TYPET(decltype("123")));
  gDebug(G_TYPET(decltype(("123"))));
  gDebug(G_TYPET(decltype((std::string("??")))));
  gDebug(G_TYPET(decltype((Fun()))));
  gDebug(G_TYPET(decltype((Fun2()))));
  int b[2];
  int* p = b;

  gDebug(G_TYPET(decltype((p))));
  gDebug(G_TYPET(decltype((+p))));
  gDebug(G_TYPET(decltype((++p))));
  gDebug(G_TYPET(decltype((p++))));

  gDebug(G_TYPET(decltype((std::move(p)))));
  gDebug(G_TYPET(decltype((std::move(+p)))));
  gDebug(G_TYPET(decltype((std::move(++p)))));
  gDebug(G_TYPET(decltype((std::move(p++)))));
  gDebug(G_TYPET(decltype((std::move(p)[0]))));
  gDebug(G_TYPET(decltype((std::move(+p)[0]))));
  gDebug(G_TYPET(decltype((std::move(++p)[0]))));
  gDebug(G_TYPET(decltype((std::move(p++)[0]))));

  gDebug(G_TYPET(decltype((b))));
  gDebug(G_TYPET(decltype((+b))));
  // gDebug(G_TYPET(decltype((++b))));
  // gDebug(G_TYPET(decltype((b++))));

  gDebug(G_TYPET(decltype((std::move(b)))));
  gDebug(G_TYPET(decltype((std::move(+b)))));
  // gDebug(G_TYPET(decltype((std::move(++b)))));
  // gDebug(G_TYPET(decltype((std::move(b++)))));
  gDebug(G_TYPET(decltype((std::move(b)[0]))));
  gDebug(G_TYPET(decltype((std::move(+b)[0]))));
  // gDebug(G_TYPET(decltype((std::move(++b)[0]))));
  // gDebug(G_TYPET(decltype((std::move(b++)[0]))));
  // gDebug(G_TYPET(decltype((a))));
}
