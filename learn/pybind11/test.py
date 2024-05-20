# 导入对应的动态库即可，或者这个python脚本和动态库放在同一级目录下
import pybind11_test

print(pybind11_test.Max(10,20))
print(pybind11_test.Add(10,20))

tmp = pybind11_test.ClassTest("pybind11_test")
print(tmp)
tmp.SetName("test")
print(tmp.GetName())
print(tmp)
