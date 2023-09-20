#include <iostream>
#include <unordered_map>

void DiffBetweenAtAndBrackt(const std::unordered_map<std::string,int>&
unordered_map1) {
  // if(unordered_map1["123"]>unordered_map1["456"]) { } // error // because const cannot use [] to index

  auto& unordered_map2=const_cast<std::unordered_map<std::string,int>&>(unordered_map1);
  if(unordered_map2["123"]>unordered_map2["456"]) { } // ok // because now unordered_map2 isn't const
  
  if(unordered_map1.at("123")>unordered_map1.at("456")) { } // ok
}

void DiffBetweenAtAndBrackt(
    std::unordered_map<std::string, int>& unordered_map1) {
  if (unordered_map1["123"] > unordered_map1["456"]) { } // ok
  if (unordered_map1.at("123") > unordered_map1.at("456")) { } // ok
}

int main() {

}

