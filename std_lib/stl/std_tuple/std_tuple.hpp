#pragma once
#include "common.h"


GXT_NAMESPACE_BEGIN

template <typename... Values>
class tuple;

template <>
class tuple<> {};

template <typename Head, typename... Tail>
class tuple<Head, Tail...> : private tuple<Tail...> {
  typedef tuple<Tail...> inherited;

 public:
  tuple() {}
  tuple(Head v, Tail... vtail) : m_head(v), inherited(vtail...) {}

  Head& head() { return m_head; }
  inherited& tail() { return *this; }

 protected:
  Head m_head;
};

template <std::size_t Index, typename Head, typename... Tail>
struct tuple_get_helper {
  static auto& get(tuple<Head, Tail...>& t) {
    return tuple_get_helper<Index - 1, Tail...>::get(t.tail());
  }
};

template <typename Head, typename... Tail>
struct tuple_get_helper<0, Head, Tail...> {
  static auto& get(tuple<Head, Tail...>& t) { return t.head(); }
};

template <std::size_t Index, typename... Values>
auto& get(tuple<Values...>& t) {
  return tuple_get_helper<Index, Values...>::get(t);
}

GXT_NAMESPACE_END

