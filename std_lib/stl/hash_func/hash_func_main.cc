#include "hash_func.hpp"

int main(int argc, char* argv[]) {
  {
    std::unordered_map<Test, int, TestHashFunc> unordered_map1;
    auto find = unordered_map1.begin();
    Test test1{1, 2.5, "34"};
    unordered_map1[test1] = 10;

    Test test2{1, 3.5, "56"};
    unordered_map1[test2] = 20;

    // this is because hash funcition just judge with mem var a
    // and judge two objects the same also only rely on a
    auto it=unordered_map1.begin();
    it=unordered_map1.find({1,4.5,"78"});
    if(it!=unordered_map1.end()) {
      std::cout << "find : " << it->second << std::endl;
    }
    gDebug(unordered_map1);
  }
  {
    std::unordered_map<Standard,int,StandardHashFunc1> standard1;
    standard1.insert({{1,{1.0f},"1"},10});

    std::unordered_map<Standard,int,decltype(&StandardHashFunc2)> standard2_1(13,StandardHashFunc2);
    std::unordered_map<Standard,int,std::function<size_t(const Standard&)>> standard2_2(13,StandardHashFunc2);
    std::unordered_map<Standard, int, std::function<size_t(const Standard&)>> standard2_3(13, [](const Standard& obj) -> size_t {
          return std::hash<int>()(obj.a) + std::hash<float>()(obj.b.involve) -
                 std::hash<std::string>()(obj.c) * 3.14;
        });
  }
  {
    std::unordered_map<Standard,int,StandardHashFunc1,std::equal_to<Standard>> standard1;

    std::unordered_map<Standard,int,std::function<size_t(const Standard&)>,std::function<bool(const Standard&,const Standard&)>>
      standard2(13,StandardHashFunc2,My_equal_to);
  }
}
