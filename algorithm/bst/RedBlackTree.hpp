#pragma once

#include "common.h"

/*
Rules That Every Red-Black Tree Follows:
1. Every node has a color either red or black.
2. The root of the tree is always black.
3. There are no two adjacent red nodes (A red node cannot have a red parent or
red child).
4. Every path from a node (including root) to any of its descendants NULL nodes
has the same number of black nodes.
5. Every leaf (e.i. NULL node) must be colored BLACK.
*/
