#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

void Test1() {
  auto vector1 = std::unique_ptr<std::vector<int>>(new std::vector<int>(100));

  auto vector2_tmp = new std::vector<int>(100);
  auto vector2 = std::unique_ptr<std::vector<int>>(vector2_tmp);

  auto vector3 = std::make_unique<std::vector<int>>(100);
}

void Test2() {
  auto vector1 = std::shared_ptr<std::vector<int>>(new std::vector<int>(100));

  auto vector2_tmp = new std::vector<int>(100);
  auto vector2 = std::shared_ptr<std::vector<int>>(vector2_tmp);

  auto vector3 = std::make_shared<std::vector<int>>(100);

  auto vector4 = std::shared_ptr(vector1);
}

class Test {
 public:
  Test() : ptr1(std::make_unique<std::vector<int>>()){};

 private:
  std::unique_ptr<std::vector<int>> ptr1;
};

class FileRAII {
 private:
  std::ofstream file;

 public:
  FileRAII(const std::string& filename) { file.open(filename); }

  ~FileRAII() {
    if (file.is_open()) {
      file.close();
    }
  }
  void write(const std::string& data) {}
};

int main() { 
  return 0; 
}
