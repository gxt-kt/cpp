#include "std_tuple.hpp"


int main(int argc, char* argv[]) {
  GXT_NAMESPACE::tuple<int, double, std::string> goo(1, 2.5, "111");

  gDebug() << goo.head();
  gDebug() << goo.tail().head();
  gDebug() << goo.tail().tail().head();
  gDebug() << goo.tail().tail().tail();

  goo.head() = 20;
  gDebug() << goo.head();

  gDebug(GXT_NAMESPACE::get<0>(goo));
  gDebug(GXT_NAMESPACE::get<1>(goo));
  gDebug(GXT_NAMESPACE::get<2>(goo));
  // gDebug(GXT_NAMESPACE::get<3>(goo)); // error

  gDebug(TYPE(GXT_NAMESPACE::get<0>(goo)));
  gDebug(TYPE(GXT_NAMESPACE::get<1>(goo)));
  gDebug(TYPE(GXT_NAMESPACE::get<2>(goo)));
}
