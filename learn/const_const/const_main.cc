#include "const.hpp"

int main(int argc, char* argv[]) {
  int a = 1;
  int b = 2;
  int* pa = &a;
  int* pb = &b;
  int** ppa = &pa;
  int** ppb = &pb;

  {
    const int aa = 10;
    // p=1; // error
  }
  {
    const int* p=pa;
    // int const* p = pa;  // the same as above line
    // *p = 10; // error
    p = pb;
  }
  {
    int* const p = pa;
    *p = 10;  // error
    // p = pb; // error
  }
  {
    const int* const p = pa;
    // int const* const p = pa;  // the same as above line
    // *p = 10; // error
    // p = pb; // error
  }
  {
    int** p = ppa;
    p = ppb;
    *p = pb;
    **p = b;
  }
  {
    // why: because c++ need to protect int here. So disable implicit convert int** to const int**
    // Also disable implicit convert int* to const int* 
    // To avoid bypassing const int** and thus change the constant value.
    // const int ** p=ppa; // error
    // int const ** p=ppa; // the same as above line

    // But this code can run
    int tmp=100;
    const int* pt=&tmp;
    const int** ppt=&pt;
  }
  {
    // think this (int(*->const)) (*) p = ppa;
    int* const* p = ppa;
    p = ppb;
    // *p=pb; // error
    **p = b;
  }
  {
    // think this (int*) (*->const) p = ppa;
    int** const p = ppa;
    // p = ppb;  // error
    *p = pb;
    **p = b;
  }
  return 0;
}
