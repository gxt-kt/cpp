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

// Ref: https://www.geeksforgeeks.org/introduction-to-red-black-tree/?ref=lbp

#include <iostream>
#include <queue>

class Node {
 public:
  enum COLOR : int { RED, BLACK };

 public:
  int val;
  COLOR color;
  Node *left, *right, *parent;

  Node(int val) : val(val) {
    parent = left = right = NULL;

    // Node is created during insertion
    // Node is Node::RED at insertion
    color = Node::RED;
  }

  // returns pointer to uncle
  Node *uncle() {
    // If no parent or grandparent, then no uncle
    if (parent == NULL or parent->parent == NULL) return NULL;

    if (parent->isOnLeft())
      // uncle on right
      return parent->parent->right;
    else
      // uncle on left
      return parent->parent->left;
  }

  // check if node is left child of parent
  bool isOnLeft() { return this == parent->left; }

  // returns pointer to sibling
  Node *sibling() {
    // sibling null if no parent
    if (parent == NULL) return NULL;

    if (isOnLeft()) return parent->right;

    return parent->left;
  }

  // moves node down and moves given node in its place
  void moveDown(Node *nParent) {
    if (parent != NULL) {
      if (isOnLeft()) {
        parent->left = nParent;
      } else {
        parent->right = nParent;
      }
    }
    nParent->parent = parent;
    parent = nParent;
  }

  bool hasREDChild() {
    return (left != NULL and left->color == Node::RED) or
           (right != NULL and right->color == Node::RED);
  }
};

class RBTree {
  Node *root;

  // left rotates the given node
  void leftRotate(Node *x) {
    // new parent will be node's right child
    Node *nParent = x->right;

    // update root if current node is root
    if (x == root) root = nParent;

    x->moveDown(nParent);

    // connect x with new parent's left element
    x->right = nParent->left;
    // connect new parent's left element with node
    // if it is not null
    if (nParent->left != NULL) nParent->left->parent = x;

    // connect new parent with x
    nParent->left = x;
  }

  void rightRotate(Node *x) {
    // new parent will be node's left child
    Node *nParent = x->left;

    // update root if current node is root
    if (x == root) root = nParent;

    x->moveDown(nParent);

    // connect x with new parent's right element
    x->left = nParent->right;
    // connect new parent's right element with node
    // if it is not null
    if (nParent->right != NULL) nParent->right->parent = x;

    // connect new parent with x
    nParent->right = x;
  }

  void swapColors(Node *x1, Node *x2) {
    Node::COLOR temp;
    temp = x1->color;
    x1->color = x2->color;
    x2->color = temp;
  }

  void swapValues(Node *u, Node *v) {
    int temp;
    temp = u->val;
    u->val = v->val;
    v->val = temp;
  }

  // fix Node::RED Node::RED at given node
  void fixREDRED(Node *x) {
    // if x is root color it Node::BLACK and return
    if (x == root) {
      x->color = Node::BLACK;
      return;
    }

    // initialize parent, grandparent, uncle
    Node *parent = x->parent, *grandparent = parent->parent,
         *uncle = x->uncle();

    if (parent->color != Node::BLACK) {
      if (uncle != NULL && uncle->color == Node::RED) {
        // uncle Node::RED, perform recoloring and recurse
        parent->color = Node::BLACK;
        uncle->color = Node::BLACK;
        grandparent->color = Node::RED;
        fixREDRED(grandparent);
      } else {
        // Else perform LR, LL, RL, RR
        if (parent->isOnLeft()) {
          if (x->isOnLeft()) {
            // for left right
            swapColors(parent, grandparent);
          } else {
            leftRotate(parent);
            swapColors(x, grandparent);
          }
          // for left left and left right
          rightRotate(grandparent);
        } else {
          if (x->isOnLeft()) {
            // for right left
            rightRotate(parent);
            swapColors(x, grandparent);
          } else {
            swapColors(parent, grandparent);
          }

          // for right right and right left
          leftRotate(grandparent);
        }
      }
    }
  }

  // find node that do not have a left child
  // in the subtree of the given node
  Node *successor(Node *x) {
    Node *temp = x;

    while (temp->left != NULL) temp = temp->left;

    return temp;
  }

  // find node that replaces a deleted node in BST
  Node *BSTreplace(Node *x) {
    // when node have 2 children
    if (x->left != NULL and x->right != NULL) return successor(x->right);

    // when leaf
    if (x->left == NULL and x->right == NULL) return NULL;

    // when single child
    if (x->left != NULL)
      return x->left;
    else
      return x->right;
  }

  // deletes the given node
  void deleteNode(Node *v) {
    Node *u = BSTreplace(v);

    // True when u and v are both Node::BLACK
    bool uvBLACK =
        ((u == NULL or u->color == Node::BLACK) and (v->color == Node::BLACK));
    Node *parent = v->parent;

    if (u == NULL) {
      // u is NULL therefore v is leaf
      if (v == root) {
        // v is root, making root null
        root = NULL;
      } else {
        if (uvBLACK) {
          // u and v both Node::BLACK
          // v is leaf, fix double Node::BLACK at v
          fixDoubleBLACK(v);
        } else {
          // u or v is Node::RED
          if (v->sibling() != NULL)
            // sibling is not null, make it Node::RED"
            v->sibling()->color = Node::RED;
        }

        // delete v from the tree
        if (v->isOnLeft()) {
          parent->left = NULL;
        } else {
          parent->right = NULL;
        }
      }
      delete v;
      return;
    }

    if (v->left == NULL or v->right == NULL) {
      // v has 1 child
      if (v == root) {
        // v is root, assign the value of u to v, and delete u
        v->val = u->val;
        v->left = v->right = NULL;
        delete u;
      } else {
        // Detach v from tree and move u up
        if (v->isOnLeft()) {
          parent->left = u;
        } else {
          parent->right = u;
        }
        delete v;
        u->parent = parent;
        if (uvBLACK) {
          // u and v both Node::BLACK, fix double Node::BLACK at u
          fixDoubleBLACK(u);
        } else {
          // u or v Node::RED, color u Node::BLACK
          u->color = Node::BLACK;
        }
      }
      return;
    }

    // v has 2 children, swap values with successor and recurse
    swapValues(u, v);
    deleteNode(u);
  }

  void fixDoubleBLACK(Node *x) {
    if (x == root)
      // Reached root
      return;

    Node *sibling = x->sibling(), *parent = x->parent;
    if (sibling == NULL) {
      // No sibling, double Node::BLACK pushed up
      fixDoubleBLACK(parent);
    } else {
      if (sibling->color == Node::RED) {
        // Sibling Node::RED
        parent->color = Node::RED;
        sibling->color = Node::BLACK;
        if (sibling->isOnLeft()) {
          // left case
          rightRotate(parent);
        } else {
          // right case
          leftRotate(parent);
        }
        fixDoubleBLACK(x);
      } else {
        // Sibling Node::BLACK
        if (sibling->hasREDChild()) {
          // at least 1 Node::RED children
          if (sibling->left != NULL and sibling->left->color == Node::RED) {
            if (sibling->isOnLeft()) {
              // left left
              sibling->left->color = sibling->color;
              sibling->color = parent->color;
              rightRotate(parent);
            } else {
              // right left
              sibling->left->color = parent->color;
              rightRotate(sibling);
              leftRotate(parent);
            }
          } else {
            if (sibling->isOnLeft()) {
              // left right
              sibling->right->color = parent->color;
              leftRotate(sibling);
              rightRotate(parent);
            } else {
              // right right
              sibling->right->color = sibling->color;
              sibling->color = parent->color;
              leftRotate(parent);
            }
          }
          parent->color = Node::BLACK;
        } else {
          // 2 Node::BLACK children
          sibling->color = Node::RED;
          if (parent->color == Node::BLACK)
            fixDoubleBLACK(parent);
          else
            parent->color = Node::BLACK;
        }
      }
    }
  }

  // prints level order for given node
  void levelOrder(Node *x) {
    if (x == NULL)
      // return if node is null
      return;

    // queue for level order
    std::queue<Node *> q;
    Node *curr;

    // push x
    q.push(x);

    while (!q.empty()) {
      // while q is not empty
      // dequeue
      curr = q.front();
      q.pop();

      // print node value
      std::cout << curr->val << " ";

      // push children to queue
      if (curr->left != NULL) q.push(curr->left);
      if (curr->right != NULL) q.push(curr->right);
    }
  }

  // prints inorder recursively
  void inorder(Node *x) {
    if (x == NULL) return;
    inorder(x->left);
    std::cout << x->val << " ";
    inorder(x->right);
  }

 public:
  // constructor
  // initialize root
  RBTree() { root = NULL; }

  Node *getRoot() { return root; }

  // searches for given value
  // if found returns the node (used for delete)
  // else returns the last node while traversing (used in insert)
  Node *search(int n) {
    Node *temp = root;
    while (temp != NULL) {
      if (n < temp->val) {
        if (temp->left == NULL)
          break;
        else
          temp = temp->left;
      } else if (n == temp->val) {
        break;
      } else {
        if (temp->right == NULL)
          break;
        else
          temp = temp->right;
      }
    }

    return temp;
  }

  // inserts the given value to tree
  void insert(int n) {
    Node *newNode = new Node(n);
    if (root == NULL) {
      // when root is null
      // simply insert value at root
      newNode->color = Node::BLACK;
      root = newNode;
    } else {
      Node *temp = search(n);

      if (temp->val == n) {
        // return if value already exists
        return;
      }

      // if value is not found, search returns the node
      // where the value is to be inserted

      // connect new node to correct node
      newNode->parent = temp;

      if (n < temp->val)
        temp->left = newNode;
      else
        temp->right = newNode;

      // fix Node::RED Node::RED violation if exists
      fixREDRED(newNode);
    }
  }

  // utility function that deletes the node with given value
  void deleteByVal(int n) {
    if (root == NULL)
      // Tree is empty
      return;

    Node *v = search(n), *u;

    if (v->val != n) {
      std::cout << "No node found to delete with value:" << n << std::endl;
      return;
    }

    deleteNode(v);
  }

  // prints inorder of the tree
  void printInOrder() {
    std::cout << "Inorder: " << std::endl;
    if (root == NULL)
      std::cout << "Tree is empty" << std::endl;
    else
      inorder(root);
    std::cout << std::endl;
  }

  // prints level order of the tree
  void printLevelOrder() {
    std::cout << "Level order: " << std::endl;
    if (root == NULL)
      std::cout << "Tree is empty" << std::endl;
    else
      levelOrder(root);
    std::cout << std::endl;
  }
};
