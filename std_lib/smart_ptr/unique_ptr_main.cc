#include "unique_ptr.hpp"

struct TmpClass {
  int a;
  void Foo() {}
};

int main(int argc, char *argv[]) {
  gDebug() << "exec" << __FILE__;
  {
    GXT_NAMESPACE::unique_ptr<int> ptr1(new int(10));
  }
  {
    GXT_NAMESPACE::unique_ptr<int> ptr1(new int(10));
    GXT_NAMESPACE::unique_ptr<int> ptr2(new int(10));
    // auto ptr3=ptr1; // delete function
    // ptr2=ptr1; // delete function
    GXT_NAMESPACE::unique_ptr<int> ptr4(std::move(ptr1));  // can run
    ptr4 = std::move(ptr4);                                // can run
  }
  {
    GXT_NAMESPACE::unique_ptr<TmpClass> ptr1(new TmpClass);
    ptr1->a = 10;
    (*ptr1).a = 10;
    ptr1.get()->a = 10;
    assert(static_cast<bool>(ptr1) == true);
    ptr1.reset();
    assert(static_cast<bool>(ptr1) == false);
    ptr1.reset(new TmpClass);
    assert(static_cast<bool>(ptr1) == true);

    ptr1.release();
    assert(static_cast<bool>(ptr1) == false);
  }
  {
    GXT_NAMESPACE::unique_ptr<std::vector<int>> ptr1(
        new std::vector<int>{1, 2, 3});
    GXT_NAMESPACE::unique_ptr<std::vector<int>> ptr2(std::move(ptr1));
    assert(static_cast<bool>(ptr1) == false);
    assert(static_cast<bool>(ptr2) == true);
  }
  return 0;
}
