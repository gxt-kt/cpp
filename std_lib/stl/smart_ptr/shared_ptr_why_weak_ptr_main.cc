#include "shared_ptr.hpp"

int main(int argc, char *argv[]) {
  gDebug() << "exec" << __FILE__;

  {
    // EXAMPLE 1
    struct TmpClass1;
    struct TmpClass2;
    struct TmpClass1 {
      ~TmpClass1() { gDebug(__PRETTY_FUNCTION__); }
      GXT_NAMESPACE::shared_ptr<TmpClass2> ptr;
    };
    struct TmpClass2 {
      ~TmpClass2() { gDebug(__PRETTY_FUNCTION__); }
      GXT_NAMESPACE::shared_ptr<TmpClass1> ptr;
    };

    auto p1 = GXT_NAMESPACE::shared_ptr<TmpClass1>(new TmpClass1);
    auto p2 = GXT_NAMESPACE::shared_ptr<TmpClass2>(new TmpClass2);
    assert(p1.use_count() == 1);
    assert(p2.use_count() == 1);
    p1->ptr = p2;
    p2->ptr = p1;
    assert(p1.use_count() == 2);
    assert(p2.use_count() == 2);
    // NOTE: There will not exec p1/p2 's deconstruct function and
    // lead to memory leak.
    //
  }

  {
    // EXAMPLE IMPROVE 1
    struct TmpClass1;
    struct TmpClass2;
    struct TmpClass1 {
      ~TmpClass1() { gDebug(__PRETTY_FUNCTION__); }
      GXT_NAMESPACE::shared_ptr<TmpClass2> ptr;
    };
    struct TmpClass2 {
      ~TmpClass2() { gDebug(__PRETTY_FUNCTION__); }
      GXT_NAMESPACE::weak_ptr<TmpClass1> ptr;  // change shared_ptr to weak_ptr
    };

    auto p1 = GXT_NAMESPACE::shared_ptr<TmpClass1>(new TmpClass1);
    auto p2 = GXT_NAMESPACE::shared_ptr<TmpClass2>(new TmpClass2);
    assert(p1.use_count() == 1);
    assert(p2.use_count() == 1);
    p1->ptr = p2;
    p2->ptr = p1;
    assert(p1.use_count() == 1);  // this count will change to 1
    assert(p2.use_count() == 2);
  }

  {
    // EXAMPLE 2
    // The example if from below video and time58:00
    // https://www.bilibili.com/video/BV1G94y1x7NX/?spm_id_from=333.880.my_history.page.click&vd_source=01da08e4487b8e450cf16063029887c6
    struct TmpClass {
      using fun_type = std::function<void()>;
      ~TmpClass() { gDebug(__PRETTY_FUNCTION__); }

      void SetFun(fun_type fun) { fun_ = fun; }
      void Goo() { gDebug(__PRETTY_FUNCTION__); }
      fun_type fun_;
    };
    auto bar = GXT_NAMESPACE::shared_ptr<TmpClass>(new TmpClass);
    bar->SetFun([bar]() { bar->Goo(); });

    // NOTE: There will not exec bar's deconstruct function and
    // lead to memory leak.
    // The reason is the lambda expression instantiate a new
    // object and use copy method to capture bar lead to add the bar's count.
    // The essence of the reason is actually circular reference.
  }

  {
    // EXAMPLE IMPROVE 2
    struct TmpClass {
      using fun_type = std::function<void()>;
      ~TmpClass() { gDebug(__PRETTY_FUNCTION__); }

      void SetFun(fun_type fun) { fun_ = fun; }
      void Goo() { gDebug(__PRETTY_FUNCTION__); }
      fun_type fun_;
    };
    auto bar = GXT_NAMESPACE::shared_ptr<TmpClass>(new TmpClass);
    bar->SetFun([tmp = GXT_NAMESPACE::weak_ptr<TmpClass>(bar)]() {
      tmp.lock()->Goo();
    });  // convert oridinary object to weak_ptr
  }

  return 0;
}
