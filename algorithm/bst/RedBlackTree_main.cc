#include "RedBlackTree.hpp"

void PrintTree_1(Node* root) {
  gxt::leetcode::PrintTree2(
      root, [](Node* node) { return node->left; },
      [](Node* node) { return node->right; },
      [](Node* node) { return node->val; });
}

int main(int argc, char* argv[]) {
  gDebug("???");
  gDebugLog("???");
  RBTree tree;

  tree.insert(7);
  tree.insert(3);
  tree.insert(18);
  tree.insert(10);
  tree.insert(22);
  tree.insert(8);
  tree.insert(11);
  tree.insert(26);
  tree.insert(2);
  tree.insert(6);
  tree.insert(13);

  PrintTree_1(tree.getRoot());

  tree.printInOrder();
  tree.printLevelOrder();

  std::cout << std::endl << "Deleting 18, 11, 3, 10, 22" << std::endl;

  tree.deleteByVal(18);
  tree.deleteByVal(11);
  tree.deleteByVal(3);
  tree.deleteByVal(10);
  tree.deleteByVal(22);

  PrintTree_1(tree.getRoot());
  tree.printInOrder();
  tree.printLevelOrder();
  return 0;
}
