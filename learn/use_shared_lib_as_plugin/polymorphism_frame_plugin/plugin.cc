#include "plugin.h"

#include <iostream>

int PluginDerived::AddImpl() { return a_ + b_; };

void PluginDerived::Execute() { res = AddImpl(); };
int PluginDerived::GetResult() { return res; };

extern "C" PluginBase* CreatePlugin() { return new PluginDerived(); }
