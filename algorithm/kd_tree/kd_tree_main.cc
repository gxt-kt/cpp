#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

struct Point {
  std::vector<double> coordinates;
};

struct Node {
  Point point;
  Node* left;
  Node* right;

  Node(const Point& p) : point(p), left(nullptr), right(nullptr) {}
};

class KDTree {
 private:
  Node* root;

  Node* buildTree(std::vector<Point>& points, int depth) {
    if (points.empty()) {
      return nullptr;
    }

    int dim = points[0].coordinates.size();
    int axis = depth % dim;

    std::sort(points.begin(), points.end(),
              [axis](const Point& a, const Point& b) {
                return a.coordinates[axis] < b.coordinates[axis];
              });

    int median = points.size() / 2;
    Node* node = new Node(points[median]);

    std::vector<Point> leftPoints(points.begin(), points.begin() + median);
    std::vector<Point> rightPoints(points.begin() + median + 1, points.end());

    node->left = buildTree(leftPoints, depth + 1);
    node->right = buildTree(rightPoints, depth + 1);

    return node;
  }

  void nearestNeighborSearch(Node* node, const Point& target, Node*& best,
                             double& bestDistance, int depth) {
    if (node == nullptr) {
      return;
    }

    double distance = calculateDistance(node->point, target);
    if (distance < bestDistance) {
      best = node;
      bestDistance = distance;
    }

    int dim = target.coordinates.size();
    int axis = depth % dim;

    if (target.coordinates[axis] < node->point.coordinates[axis]) {
      nearestNeighborSearch(node->left, target, best, bestDistance, depth + 1);
      if (std::abs(target.coordinates[axis] - node->point.coordinates[axis]) <
          bestDistance) {
        nearestNeighborSearch(node->right, target, best, bestDistance,
                              depth + 1);
      }
    } else {
      nearestNeighborSearch(node->right, target, best, bestDistance, depth + 1);
      if (std::abs(target.coordinates[axis] - node->point.coordinates[axis]) <
          bestDistance) {
        nearestNeighborSearch(node->left, target, best, bestDistance,
                              depth + 1);
      }
    }
  }

  double calculateDistance(const Point& p1, const Point& p2) {
    double distance = 0.0;
    for (size_t i = 0; i < p1.coordinates.size(); ++i) {
      double diff = p1.coordinates[i] - p2.coordinates[i];
      distance += diff * diff;
    }
    return std::sqrt(distance);
  }

 public:
  KDTree() : root(nullptr) {}

  void build(std::vector<Point>& points) { root = buildTree(points, 0); }

  Point findNearestNeighbor(const Point& target) {
    Node* best = nullptr;
    double bestDistance = std::numeric_limits<double>::max();
    nearestNeighborSearch(root, target, best, bestDistance, 0);
    return best->point;
  }
};

int main() {
  // 示例用法
  std::vector<Point> points = {{{2, 3}}, {{5, 4}}, {{9, 6}},
                               {{4, 7}}, {{8, 1}}, {{7, 2}}};

  KDTree kdTree;
  kdTree.build(points);

  Point target = {{9, 2}};
  Point nearestNeighbor = kdTree.findNearestNeighbor(target);

  std::cout << "Nearest neighbor: (" << nearestNeighbor.coordinates[0] << ", "
            << nearestNeighbor.coordinates[1] << ")" << std::endl;

  return 0;
}
