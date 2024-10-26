#include "common.h"
#include "variant.h"

int main(int argc, char *argv[]) {
  Variant<std::string, int, double> v1(inPlaceIndex<0>, "asas");
  gDebug(v1.get<std::string>());
  Variant<std::string, int, double> v2(42);
  gDebug(v2.get<1>());
  Variant<std::string, int, double> v3(3.14);
  gDebug(v3.get<2>());
  return 0;
}
