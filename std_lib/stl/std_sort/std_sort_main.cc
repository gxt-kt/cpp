#include "std_sort.hpp"


// Driver program to test Introsort
int main() {
  int arr[] = {3, 1, 23, -9, 233, 23, -313, 32, -9};
  int n = sizeof(arr) / sizeof(arr[0]);

  // Pass the array, the pointer to the first element and
  // the pointer to the last element
  Introsort(arr, arr, arr + n - 1);
  printArray(arr, n);


  std::vector<int> aa{10,20,40,20,30};
  std::sort(aa.begin(),aa.end());

  return (0);
}
