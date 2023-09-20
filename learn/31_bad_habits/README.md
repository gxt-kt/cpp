# cpp bad habits

>  Ref: [https://www.youtube.com/watch?v=i_wDa2AS_8w](https://www.youtube.com/watch?v=i_wDa2AS_8w)
>  
>  - I have merged some chapters that the content is similar or can be discussed together.
> 
>  - I also delete several chapters that too simple or no necessary to talk about.


## 1. Do not use `using namepsace std` on global scope

Do not use `using namepsace std` on global scope. Do not use it in header files.

You can use it in function scope like the demo. And the effect scope is only the function.

## 2. Use `'\n'` when use cout instead of `std::endl`

If you want to print a new line, use `std::endl` will refresh the buffer and take more time.

It's better using the `'\n'` and faster.

But maybe it won't refresh when you debug with `'\n'`.

## 3. Use for loop based of auto

When you do not need the index. Use for loop based of auto.

## 4. Use the already exist algorithm interface

If you want realize some algorithm. Call the existing algorithm interface first.

## 5. Use std::array to replace c style array

Use std::array to replace c style array

## 6. Don't abuse `reinterpret_cast` to change type

`reinterpret_cast` should used to change type to it's original type. All other using is UB.

The C stype convert type is the same.


## 7_8. The const can be cast away and const unorder_map cannot use `[]` to index

First you can use `const_cast` to cast the const away.

Second For const unorder_map, you need use `.at()` to index.

Use `[]` will create a defaule value and may change the map size when the index isn't exist.

Use `.at()` won't change the map size and will cast a exception when the index isn't exist.

So you cann't use `[]` to index when the map is const.


## 9_10. Don't fall const down if you sure you don't change the parameter

It's better use const if you know you don't change the paramter.

## 11. Use struct form is better than single value

Use the struct make your programs have better readability.

Such as you can use `const auto& [first,second]:map` to iterate map.

Then you can just use first and second instead of val.first and val.second.

## 12. Use struct when you have multiple return values

You can use `auto [val1,val2,...]=ReturnFun()` to get the values and it's no necessary to create a new struct variable.

## 13. Use `constexpr` to calculate some expression on the compile time.

Use `constexpr` for some simple calculation expression when you know the input value.

## 14. You need add `virtual` for the parent class constructor function when inheriting.

If you don't add virtual, the program will not exec the child deconstructor function when the object destruction.

Also it's a good habit to add override behind the child deconstructor function.

Thus it's will be compile error withoud virtual.

## 15. The variables in the class are initialized in the order of definition 

So you can use the forent define variable to initialize the behind variable.

## 16. Always initialize your variable is a good idea

Notice that different initialize way may have different value.

## 17. Let your values have names

Use constexpr or define to substitute the values. And this will make your programs have better readability.

## 18. Be careful to use vector size or iterate in a loop when you may change the vector size

The vector size will change so the `.size()` and `.end()` will change. The loop may cannot stop.

The corrent way is predefine `.size()` when you want loop.

The special `auto val:vector` have a different strange phenomenon that the loop can stop but value is strange.

It's because the vector size change thus the vector will move the data to a new memory address but auto loop still print the original address value.

## 19_20. Use `std::move` to pass a large size value

Althouth the compiler will automatically use RVO(Return Value Optimization), it's a good idea use `std::move` manually.

Sometimes the condition is complicate and the compiler cannot auto use RVO, the std::move is necessary.

## 21. Prior to cpp17, the code exec order cannot be guaranteed

The format like `a.b().c().d()` exec order cannot be guaranteed before cpp17.

But for the format like `fun(fun1(),fun2(),fun3())` exec order cannot be guaranteed even cpp20.

## 22. Reduce unnecessary heap allocation

If new object is only use in the function, it's unnecessary to use new. Just Create the object by normal way.

Unless you sure want them on the heap.

## 23_24_25_26. Less new, more smart pointer

You need to remember to delete manually when using new, otherwise memory leaks may occur.

You don't hvae to worry about this problem with smart pointers.

So less new and more smart pointer.

Also, RAII（Resource Acquisition Is Initialization）thought is important.

## 27. The `shared_ptr` is not thread-safe

Only the reference couts of shared_ptr is thread-safe. The data isn't.

So you also need to add mutex for date of shared_ptr.

## 28. Distinguish the const pointer and pointer const

It's simple to distinguish. Just remember that the const always decorate whatever is immediately to its left. Unless the const is the most left thing, in which it applies to its right.

In chinese name, you need read it from left to right.

So `const int` read `常量指针` and `int *const` read `指针常量`.

