#pragma once

#include "common.h"

/*
 *
 * Ref: https://github.com/parallel101/stl1weekend/blob/main/Variant.hpp
 *      https://www.bilibili.com/video/BV1f3CXYtE4x
 *
 */

/*
*
* 读懂这个源码，可以先看variant_index_main.cc文件，那里为一些晦涩内容做了铺垫
* 关于源码的visit部分，也可以先看visit_main.cc文件，那里有单独的visit实现
*
*/

/*
*
* 关于variant的实现，首先要把握大概：
* 1. 底层使用union，但是由于union不支持构造函数，也不够灵活
*    所以我们自己维护一块内存（char的数组），然后每次把这个内存强转成需要的类型
* 2. 成员函数主要有：
*    一块内存（char数组），数组大小是max(sizeof(所有类型))
*    size_t 类型的索引，表明当前实际类型是第几个模板参数
* 3. 由于内存自己管理了，所以我们需要手动管理生命周期
*    会需要自己实例化对象，采用placement new的方式构造对象
*    会需要自己析构对象，生成一个析构函数表，方便调用
*    关于这些查找表，可以参考visit文件，那里有设计查找表的设计
*/



// 今天来实现标准库中的 variant 和 visit

template <size_t I>
struct InPlaceIndex {
    explicit InPlaceIndex() = default;
};

template <size_t I>
constexpr InPlaceIndex<I> inPlaceIndex;

struct BadVariantAccess : std::exception {
    BadVariantAccess() = default;
    virtual ~BadVariantAccess() = default;

    const char *what() const noexcept override {
        return "BadVariantAccess";
    }
};

template <typename, typename> // typename -> size_t
struct VariantIndex;

template <typename, size_t>   // size_t -> typename
struct VariantAlternative;

// VariantAlternative<Variant<int, double>, 1> = int;
// VariantAlternative<Variant<int, double>, 2> = double;
// VariantIndex<Variant<int, double>, int> = 1;
// VariantIndex<Variant<int, double>, double> = 2;

template <typename ...Ts>
struct Variant {
private:
    size_t m_index;

    // 采用alignas为了防止出现Variant a[10];把variant放到数组中
    alignas(std::max({alignof(Ts)...})) char m_union[std::max({sizeof(Ts)...})];

    using DestructorFunction = void(*)(char *) noexcept;

    // 实现析构函数查找表，手动析构
    static DestructorFunction *destructors_table() noexcept {
        static DestructorFunction function_ptrs[sizeof...(Ts)] = {
            [] (char *union_p) noexcept {
                reinterpret_cast<Ts *>(union_p)->~Ts();
            }...
        };
        return function_ptrs;
    }

    using CopyConstructorFunction = void(*)(char *, char const *) noexcept;

    // 实现拷贝构造查找表,使用placement new来进行初始化对象
    static CopyConstructorFunction *copy_constructors_table() noexcept {
        static CopyConstructorFunction function_ptrs[sizeof...(Ts)] = {
            [] (char *union_dst, char const *union_src) noexcept {
                new (union_dst) Ts(*reinterpret_cast<Ts const *>(union_src));
            }...
        };
        return function_ptrs;
    }

    using CopyAssignmentFunction = void(*)(char *, char const *) noexcept;

    // 实现拷贝赋值查找表
    static CopyAssignmentFunction *copy_assigment_functions_table() noexcept {
        static CopyAssignmentFunction function_ptrs[sizeof...(Ts)] = {
            [] (char *union_dst, char const *union_src) noexcept {
                *reinterpret_cast<Ts *>(union_dst) = *reinterpret_cast<Ts const *>(union_src);
            }...
        };
        return function_ptrs;
    }

    using MoveConstructorFunction = void(*)(char *, char *) noexcept;

    // 实现移动构造查找表
    static MoveConstructorFunction *move_constructors_table() noexcept {
        static MoveConstructorFunction function_ptrs[sizeof...(Ts)] = {
            [] (char *union_dst, char *union_src) noexcept {
                new (union_dst) Ts(std::move(*reinterpret_cast<Ts const *>(union_src)));
            }...
        };
        return function_ptrs;
    }

    using MoveAssignmentFunction = void(*)(char *, char *) noexcept;

    // 实现移动赋值查找表
    static MoveAssignmentFunction *move_assigment_functions_table() noexcept {
        static MoveAssignmentFunction function_ptrs[sizeof...(Ts)] = {
            [] (char *union_dst, char *union_src) noexcept {
                *reinterpret_cast<Ts *>(union_dst) = std::move(*reinterpret_cast<Ts *>(union_src));
            }...
        };
        return function_ptrs;
    }

    // 实现visit的部分
    template <class Lambda>
    using ConstVisitorFunction = typename std::common_type<typename std::invoke_result<Lambda, Ts const &>::type...>::type(*)(char const *, Lambda &&);

    template <class Lambda>
    static ConstVisitorFunction<Lambda> *const_visitors_table() noexcept {
        static ConstVisitorFunction<Lambda> function_ptrs[sizeof...(Ts)] = {
            [] (char const *union_p, Lambda &&lambda) -> typename std::invoke_result<Lambda, Ts const &>::type {
                std::invoke(std::forward<Lambda>(lambda),
                            *reinterpret_cast<Ts const *>(union_p));
            }...
        };
        return function_ptrs;
    }

    template <class Lambda>
    using VisitorFunction = typename std::common_type<typename std::invoke_result<Lambda, Ts &>::type...>::type(*)(char *, Lambda &&);

    template <class Lambda>
    static VisitorFunction<Lambda> *visitors_table() noexcept {
        static VisitorFunction<Lambda> function_ptrs[sizeof...(Ts)] = {
            [] (char *union_p, Lambda &&lambda) -> typename std::common_type<typename std::invoke_result<Lambda, Ts &>::type...>::type {
                return std::invoke(std::forward<Lambda>(lambda),
                                   *reinterpret_cast<Ts *>(union_p));
            }...
        };
        return function_ptrs;
    }

public:

    // 构造函数
    template <typename T, typename std::enable_if<
        std::disjunction<std::is_same<T, Ts>...>::value,
        int>::type = 0>
    Variant(T value) : m_index(VariantIndex<Variant, T>::value) {
        T *p = reinterpret_cast<T *>(m_union);
        new (p) T(value);
    }

    Variant(Variant const &that) : m_index(that.m_index) {
        copy_constructors_table()[index()](m_union, that.m_union);
    }

    Variant &operator=(Variant const &that) {
        m_index = that.m_index;
        copy_assigment_functions_table()[index()](m_union, that.m_union);
    }

    Variant(Variant &&that) : m_index(that.m_index) {
        move_constructors_table()[index()](m_union, that.m_union);
    }

    Variant &operator=(Variant &&that) {
        m_index = that.m_index;
        move_assigment_functions_table()[index()](m_union, that.m_union);
    }

    // 指明确定索引类型的构造函数
    template <size_t I, typename ...Args>
    explicit Variant(InPlaceIndex<I>, Args &&...value_args) : m_index(I) {
        new (m_union) typename VariantAlternative<Variant, I>::type
            (std::forward<Args>(value_args)...);
    }

    ~Variant() noexcept {
        destructors_table()[index()](m_union);
    }

    template <class Lambda>
    typename std::common_type<typename std::invoke_result<Lambda, Ts &>::type...>::type visit(Lambda &&lambda) {
        // 由于时间原因，暂时没有实现支持多参数的std::visit，决定留作回家作业，供学有余力的同学自己尝试实现
        return visitors_table<Lambda>()[index()](m_union, std::forward<Lambda>(lambda));
    }

    template <class Lambda>
    typename std::common_type<typename std::invoke_result<Lambda, Ts const &>::type...>::type visit(Lambda &&lambda) const {
        return const_visitors_table<Lambda>()[index()](m_union, std::forward<Lambda>(lambda));
    }

    constexpr size_t index() const noexcept {
        return m_index;
    }

    template <typename T>
    constexpr bool holds_alternative() const noexcept {
        return VariantIndex<Variant, T>::value == index();
    }

    template <size_t I>
    typename VariantAlternative<Variant, I>::type &get() {
        static_assert(I < sizeof...(Ts), "I out of range!");
        if (m_index != I)
            throw BadVariantAccess();
        return *reinterpret_cast<typename VariantAlternative<Variant, I>::type *>(m_union);
    }

    template <typename T>
    T &get() {
        return get<VariantIndex<Variant, T>::value>();
    }

    template <size_t I>
    typename VariantAlternative<Variant, I>::type const &get() const {
        static_assert(I < sizeof...(Ts), "I out of range!");
        if (m_index != I)
            throw BadVariantAccess();
        return *reinterpret_cast<typename VariantAlternative<Variant, I>::type const *>(m_union);
    }

    template <typename T>
    T const &get() const {
        return get<VariantIndex<Variant, T>::value>();
    }

    template <size_t I>
    typename VariantAlternative<Variant, I>::type *get_if() {
        static_assert(I < sizeof...(Ts), "I out of range!");
        if (m_index != I)
            return nullptr;
        return reinterpret_cast<typename VariantAlternative<Variant, I>::type *>(m_union);
    }

    template <typename T>
    T *get_if() {
        return get_if<VariantIndex<Variant, T>::value>();
    }

    template <size_t I>
    typename VariantAlternative<Variant, I>::type const *get_if() const {
        static_assert(I < sizeof...(Ts), "I out of range!");
        if (m_index != I)
            return nullptr;
        return reinterpret_cast<typename VariantAlternative<Variant, I>::type const *>(m_union);
    }

    template <typename T>
    T const *get_if() const {
        return get_if<VariantIndex<Variant, T>::value>();
    }
};

template <typename T, typename ...Ts>
struct VariantAlternative<Variant<T, Ts...>, 0> {
    using type = T;
};

template <typename T, typename ...Ts, size_t I>
struct VariantAlternative<Variant<T, Ts...>, I> {
    using type = typename VariantAlternative<Variant<Ts...>, I - 1>::type;
};

template <typename T, typename ...Ts>
struct VariantIndex<Variant<T, Ts...>, T> {
    static constexpr size_t value = 0;
};

template <typename T0, typename T, typename ...Ts>
struct VariantIndex<Variant<T0, Ts...>, T> {
    static constexpr size_t value = VariantIndex<Variant<Ts...>, T>::value + 1;
};
