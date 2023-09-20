#include "shared_ptr.hpp"

struct TmpClass {
  int a;
  void Foo() {}
};

int main(int argc, char* argv[]) {
  gDebug() << "exec" << __FILE__;
  { GXT_NAMESPACE::shared_ptr<int> ptr1(new int(10)); }
  {
    GXT_NAMESPACE::shared_ptr<int> ptr1(new int(10));
    assert(ptr1.use_count() == 1);

    GXT_NAMESPACE::shared_ptr<int> ptr2(ptr1);
    assert(ptr1.use_count() == 2);
    assert(ptr2.use_count() == 2);

    GXT_NAMESPACE::shared_ptr<int> ptr3(new int(10));
    ptr3 = ptr1;
    assert(ptr1.use_count() == 3);

    GXT_NAMESPACE::shared_ptr<int> ptr4(std::move(ptr1));
    assert(ptr1.use_count() == 0);
    assert(ptr3.use_count() == 3);
  }
  {
    GXT_NAMESPACE::shared_ptr<std::vector<int>> ptr1(
        new std::vector<int>{1, 2, 3});
    GXT_NAMESPACE::shared_ptr<std::vector<int>> ptr2(std::move(ptr1));
    assert(static_cast<bool>(ptr1) == false);
    assert(static_cast<bool>(ptr2) == true);
  }
  {
    GXT_NAMESPACE::shared_ptr<int> ptr1(new int(10));
    GXT_NAMESPACE::shared_ptr<int> ptr2(new int(20));
    assert(*ptr1.get() == 10);
    assert(*ptr2 == 20);
    ptr1.swap(ptr2);
    assert(*ptr1.get() == 20);
    assert(*ptr2 == 10);
  }
  {
    GXT_NAMESPACE::shared_ptr<GXT_NAMESPACE::Derived> ptr1(
        new GXT_NAMESPACE::Derived);
    auto ptr2 = GXT_NAMESPACE::dynamic_pointer_cast<GXT_NAMESPACE::Base>(ptr1);
  }
  return 0;
}
