#include <pybind11/pybind11.h>

namespace py = pybind11;

// Ref: https://daobook.github.io/pybind11/index.html
// 暂时只给出了绑定函数和类，更多的可以参考上面的官方文档

int Max(int a, int b) { return a > b ? a : b; }

int MaxWithDefaultParam(int a, int b = 10) { return a > b ? a : b; }

class ClassTest {
 public:
  ClassTest(const std::string &str) : str_(str) {}
  void SetName(const std::string &str) { str_ = str; }
  const std::string &GetName() const { return str_; }

 private:
  std::string str_;
};

PYBIND11_MODULE(pybind11_test, m) {
  m.doc() = "gxt pybind11 example plugin";  // optional module docstring

  // 绑定一个函数示例
  m.def("Max", &Max, "Max function");

  // 绑定lambda表达式
  m.def("Add", [](int a, int b) { return a + b; }, "Add function");

  // 绑定一个类示例
  py::class_<ClassTest>(m, "ClassTest")
      .def(py::init<const std::string &>())
      .def("SetName", &ClassTest::SetName)
      .def("GetName", &ClassTest::GetName)
      .def("__repr__", [](const ClassTest &obj) {
        return obj.GetName();
      });
}
