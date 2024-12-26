#include "AVL.hpp"

void PrintTree_1(Node* root) {
  gxt::leetcode::PrintTree2(
      root, [](Node* node) { return node->left; },
      [](Node* node) { return node->right; },
      [](Node* node) { return node->key; });
}
void PrintTree_2(Node* root) {
  std::cout << gxt::leetcode::PrintTree2(
      root, [](Node* node) { return node->left; },
      [](Node* node) { return node->right; },
      [](Node* node) { return node->key; });
}

int main() {
  Node* root = NULL;

  /* Constructing tree given in
  the above figure */
  root = insert(root, 10);
  root = insert(root, 20);
  root = insert(root, 30);
  root = insert(root, 40);
  root = insert(root, 50);
  root = insert(root, 25);

  /* The constructed AVL Tree would be
                   30
                  / \
                 20 40
                / \  \
               10 25 50
  */
  std::cout << "Preorder traversal of the constructed AVL tree is \n";
  preOrder(root);
  std::cout << std::endl;

  PrintTree_1(root);
  deleteNode(root, 20);
  PrintTree_2(root);

  return 0;
}
