#pragma once

/*
 * Regular:
 * const will always decorate the left one,
 * it will decorate the right one only left has nothing.
 *
 */

/*
 *
 * 小技巧：
 * 首先如果const在最左边，就把它移动它修饰的右边，
 * 也就是把所有const都放在对应修饰的右边
 *
 *
 * 然后这时候
 * const左边的所有内容就是被加上const属性的类型
 * const右边的所有内容就不能被修改了
 *
 * 比如：int* const* p = ppa; (int*)就是const,(*p)就不能改
 * 同理：int** const p = ppa; (int**)就是const,(p)就不能改
 *
 */

#include "common.h"
