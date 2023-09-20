#include <iostream>

class Test {
public:
    Test() {
        std::cout << "Parent constructor" << std::endl;
    }

    // ~Parent() { // without virtual will not execute Child destructor function
    virtual ~Test() {
        std::cout << "Parent destructor" << std::endl;
    }
};

class Child : public Test {
public:
    Child() {
        std::cout << "Child constructor" << std::endl;
    }

    // add override you must add virtual for the parent constructor function
    ~Child() override {
        std::cout << "Child destructor" << std::endl;
    }
};

int main() {
    Test* obj = new Child();
    delete obj;

    return 0;
}
