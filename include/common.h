#include "/home/gxt_kt/Projects/debugstream/debugstream.hpp"

#define GXT_NAMESPACE gxt

#define GXT_NAMESPACE_BEGIN namespace gxt {
#define GXT_NAMESPACE_END }

GXT_NAMESPACE_BEGIN

struct TestClass {
 public:
  TestClass() { gDebugCol3(__PRETTY_FUNCTION__); }
  TestClass(int, int) { gDebugCol3(__PRETTY_FUNCTION__); }
  TestClass(TestClass &&) { gDebugCol3(__PRETTY_FUNCTION__); }
  TestClass(const TestClass &) { gDebugCol3(__PRETTY_FUNCTION__); }
  TestClass &operator=(TestClass &&) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  }
  TestClass &operator=(const TestClass &) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  };
  ~TestClass() { gDebugCol3(__PRETTY_FUNCTION__); };
};

struct Base {
 public:
  Base() { gDebugCol3(__PRETTY_FUNCTION__); }
  Base(int, int) { gDebugCol3(__PRETTY_FUNCTION__); }
  Base(Base &&) { gDebugCol3(__PRETTY_FUNCTION__); }
  Base(const Base &) { gDebugCol3(__PRETTY_FUNCTION__); }
  Base &operator=(Base &&) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  }
  Base &operator=(const Base &) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  };
  virtual ~Base() { gDebugCol3(__PRETTY_FUNCTION__); };
  virtual int Foo() {
    gDebugCol3(__PRETTY_FUNCTION__);
    return 0;
  }
};

struct Derived : public Base {
 public:
  Derived() { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived(int, int) { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived(Derived &&) { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived(const Derived &) { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived &operator=(Derived &&) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  }
  Derived &operator=(const Derived &) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  };
  virtual ~Derived() { gDebugCol3(__PRETTY_FUNCTION__); };
  virtual int Foo() {
    gDebugCol3(__PRETTY_FUNCTION__);
    return 1;
  }
};

struct Derived1 : public Base {
 public:
  Derived1() { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived1(int, int) { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived1(Derived1 &&) { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived1(const Derived1 &) { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived1 &operator=(Derived1 &&) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  }
  Derived1 &operator=(const Derived1 &) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  };
  virtual ~Derived1() { gDebugCol3(__PRETTY_FUNCTION__); };
  virtual int Foo() {
    gDebugCol3(__PRETTY_FUNCTION__);
    return 1;
  }
};

struct Derived2 : public Base {
 public:
  Derived2() { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived2(int, int) { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived2(Derived2 &&) { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived2(const Derived2 &) { gDebugCol3(__PRETTY_FUNCTION__); }
  Derived2 &operator=(Derived2 &&) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  }
  Derived2 &operator=(const Derived2 &) {
    gDebugCol3(__PRETTY_FUNCTION__);
    return *this;
  };
  virtual ~Derived2() { gDebugCol3(__PRETTY_FUNCTION__); };
  virtual int Foo() {
    gDebugCol3(__PRETTY_FUNCTION__);
    return 1;
  }
};

GXT_NAMESPACE_END
