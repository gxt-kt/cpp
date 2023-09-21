#pragma once

// C++ program to insert a node in AVL tree
#include "common.h"

// Ref: https://www.geeksforgeeks.org/introduction-to-avl-tree/
//
/*
在AVL二叉树中，判断是否需要进行左旋或右旋的依据是节点的平衡因子（balance
factor）。
平衡因子是指节点的左子树高度减去右子树高度的值，用来衡量节点的平衡状态。

对于一个节点，其平衡因子可以有以下三种情况：
- 平衡因子为0：表示节点的左右子树高度相等，即节点是平衡的。
- 平衡因子为正数：表示节点的左子树高度大于右子树高度，即左子树比右子树更重。
- 平衡因子为负数：表示节点的右子树高度大于左子树高度，即右子树比左子树更重。

当插入或删除一个节点后，如果导致了某个节点的平衡因子超过了1或小于-1，就表示该节点失去了平衡。这时就需要通过旋转操作来恢复平衡。

具体判断左旋或右旋的依据如下：

- 左旋：当一个节点的平衡因子为正数时，表示左子树更重，需要进行左旋操作。
左旋会将该节点的左子节点提升为新的根节点，原来的根节点成为新根节点的右子节点，新根节点的右子节点成为原来根节点的左子节点。

- 右旋：当一个节点的平衡因子为负数时，表示右子树更重，需要进行右旋操作。
右旋会将该节点的右子节点提升为新的根节点，原来的根节点成为新根节点的左子节点，新根节点的左子节点成为原来根节点的右子节点。

有时候，单次旋转操作可能无法完全恢复平衡，这时就需要进行双旋操作，包括左右旋和右左旋。

-
左右旋：当一个节点的左子树高度大于右子树高度，并且该节点的左子节点的右子树高度大于左子树高度时，需要进行左右旋操作。
左右旋先对左子节点进行左旋，然后再对失衡节点进行右旋。

-
右左旋：当一个节点的右子树高度大于左子树高度，并且该节点的右子节点的左子树高度大于右子树高度时，需要进行右左旋操作。
右左旋先对右子节点进行右旋，然后再对失衡节点进行左旋。

通过左旋、右旋、左右旋和右左旋等操作，可以保持AVL树的平衡性质，使得树的高度保持在较小的范围内，提高了搜索、插入和删除等操作的效率。
 */

// An AVL tree node
struct Node {
 public:
  int key;
  Node *left;
  Node *right;
  int height;
};

// A utility function to get the
// height of the tree
inline int height(Node *N) {
  if (N == NULL) return 0;
  return N->height;
}

/* Helper function that allocates a
new node with the given key and
NULL left and right pointers. */
inline Node *newNode(int key) {
  Node *node = new Node();
  node->key = key;
  node->left = NULL;
  node->right = NULL;
  node->height = 1;  // new node is initially
                     // added at leaf
  return (node);
}

//     y                               x
//    / \     Right Rotation          / \
//   x   T3   - - - - - - - >        T1  y
//  / \       < - - - - - - -           / \
// T1  T2     Left Rotation            T2  T3

// A utility function to right
// rotate subtree rooted with y
// See the diagram given above.
inline Node *rightRotate(Node *y) {
  Node *x = y->left;
  Node *T2 = x->right;

  // Perform rotation
  x->right = y;
  y->left = T2;

  // Update heights
  y->height = std::max(height(y->left), height(y->right)) + 1;
  x->height = std::max(height(x->left), height(x->right)) + 1;

  // Return new root
  return x;
}

// A utility function to left
// rotate subtree rooted with x
// See the diagram given above.
inline Node *leftRotate(Node *x) {
  Node *y = x->right;
  Node *T2 = y->left;

  // Perform rotation
  y->left = x;
  x->right = T2;

  // Update heights
  x->height = std::max(height(x->left), height(x->right)) + 1;
  y->height = std::max(height(y->left), height(y->right)) + 1;

  // Return new root
  return y;
}

// Get Balance factor of node N
inline int getBalance(Node *N) {
  if (N == NULL) return 0;
  return height(N->left) - height(N->right);
}

// Recursive function to insert a key
// in the subtree rooted with node and
// returns the new root of the subtree.
inline Node *insert(Node *node, int key) {
  /* 1. Perform the normal BST insertion */
  if (node == NULL) return (newNode(key));

  if (key < node->key)
    node->left = insert(node->left, key);
  else if (key > node->key)
    node->right = insert(node->right, key);
  else  // Equal keys are not allowed in BST
    return node;

  /* 2. Update height of this ancestor node */
  node->height = 1 + std::max(height(node->left), height(node->right));

  /* 3. Get the balance factor of this ancestor
          node to check whether this node became
          unbalanced */
  int balance = getBalance(node);

  // If this node becomes unbalanced, then
  // there are 4 cases

  // Left Left Case
  if (balance > 1 && key < node->left->key) return rightRotate(node);

  // Right Right Case
  if (balance < -1 && key > node->right->key) return leftRotate(node);

  // Left Right Case
  if (balance > 1 && key > node->left->key) {
    node->left = leftRotate(node->left);
    return rightRotate(node);
  }

  // Right Left Case
  if (balance < -1 && key < node->right->key) {
    node->right = rightRotate(node->right);
    return leftRotate(node);
  }

  /* return the (unchanged) node pointer */
  return node;
}

/* Given a non-empty binary search tree,
return the node with minimum key value
found in that tree. Note that the entire
tree does not need to be searched. */
inline Node *minValueNode(Node *node) {
  Node *current = node;

  /* loop down to find the leftmost leaf */
  while (current->left != NULL) current = current->left;

  return current;
}

// Recursive function to delete a node
// with given key from subtree with
// given root. It returns root of the
// modified subtree.
// Unlike insertion, in deletion, after we perform a rotation at z, we may have
// to perform a rotation at ancestors of z. Thus, we must continue to trace the
// path until we reach the root.
inline Node *deleteNode(Node *root, int key) {
  // STEP 1: PERFORM STANDARD BST DELETE
  if (root == NULL) return root;

  // If the key to be deleted is smaller
  // than the root's key, then it lies
  // in left subtree
  if (key < root->key) root->left = deleteNode(root->left, key);

  // If the key to be deleted is greater
  // than the root's key, then it lies
  // in right subtree
  else if (key > root->key)
    root->right = deleteNode(root->right, key);

  // if key is same as root's key, then
  // This is the node to be deleted
  else {
    // node with only one child or no child
    if ((root->left == NULL) || (root->right == NULL)) {
      Node *temp = root->left ? root->left : root->right;

      // No child case
      if (temp == NULL) {
        temp = root;
        root = NULL;
      } else            // One child case
        *root = *temp;  // Copy the contents of
                        // the non-empty child
      free(temp);
    } else {
      // node with two children: Get the inorder
      // successor (smallest in the right subtree)
      Node *temp = minValueNode(root->right);

      // Copy the inorder successor's
      // data to this node
      root->key = temp->key;

      // Delete the inorder successor
      root->right = deleteNode(root->right, temp->key);
    }
  }

  // If the tree had only one node
  // then return
  if (root == NULL) return root;

  // STEP 2: UPDATE HEIGHT OF THE CURRENT NODE
  root->height = 1 + std::max(height(root->left), height(root->right));

  // STEP 3: GET THE BALANCE FACTOR OF
  // THIS NODE (to check whether this
  // node became unbalanced)
  int balance = getBalance(root);

  // If this node becomes unbalanced,
  // then there are 4 cases

  // Left Left Case
  if (balance > 1 && getBalance(root->left) >= 0) return rightRotate(root);

  // Left Right Case
  if (balance > 1 && getBalance(root->left) < 0) {
    root->left = leftRotate(root->left);
    return rightRotate(root);
  }

  // Right Right Case
  if (balance < -1 && getBalance(root->right) <= 0) return leftRotate(root);

  // Right Left Case
  if (balance < -1 && getBalance(root->right) > 0) {
    root->right = rightRotate(root->right);
    return leftRotate(root);
  }

  return root;
}

// A utility function to print preorder
// traversal of the tree.
// The function also prints height
// of every node
inline void preOrder(Node *root) {
  if (root != NULL) {
    gDebug(root->key) << VAR(root->height);
    // std::cout << root->key << " ";
    preOrder(root->left);
    preOrder(root->right);
  }
}

// This code is contributed by
// rathbhupendra
