#pragma once
#include "common.h"

GXT_NAMESPACE_BEGIN

template <typename T>
struct remove_reference {
  using type = T;
};

template <typename T>
struct remove_reference<T&> {
  using type = T;
};

template <typename T>
struct remove_reference<T&&> {
  using type = T;
};

/**
 *  @brief  Convert a value to an rvalue.
 *  @param  __t  A thing of arbitrary type.
 *  @return The parameter cast to an rvalue-reference to allow moving it.
 */
template <typename T>
constexpr typename GXT_NAMESPACE::remove_reference<T>::type&& move(T&& __t) noexcept {
  return static_cast<typename GXT_NAMESPACE::remove_reference<T>::type&&>(__t);
}

/**
 *  @brief  Forward an lvalue.
 *  @return The parameter cast to the specified type.
 *
 *  This function is used to implement "perfect forwarding".
 */
template <typename T>
constexpr T&& forward(typename GXT_NAMESPACE::remove_reference<T>::type& __t) noexcept {
  return static_cast<T&&>(__t);
}

/**
 *  @brief  Forward an rvalue.
 *  @return The parameter cast to the specified type.
 *
 *  This function is used to implement "perfect forwarding".
 */
template <typename T>
constexpr T&& forward(typename GXT_NAMESPACE::remove_reference<T>::type&& __t) noexcept {
  return static_cast<T&&>(__t);
}

GXT_NAMESPACE_END
