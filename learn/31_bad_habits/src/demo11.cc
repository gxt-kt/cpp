#include <iostream>
#include <unordered_map>


void Test() {
  const std::unordered_map<std::string,std::string> color{
    {"RED","#ff0000"},
    {"GREEN","#00ff00"},
    {"BLUE","#0000ff"},
  };

  // iterate the map
  for (const auto& pair:color ) {
    std::cout << pair.first << " "<< pair.second << "\n";
  }


  // a better way
  for (const auto& [name,col]:color ) {
    std::cout << name << " "<< col << "\n";
  }
}

int main() {
  Test();
}

