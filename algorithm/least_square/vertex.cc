#include "vertex.h"

unsigned long global_vertex_id = 0;

Vertex::Vertex(int num_dimension, int local_dimension) {
  parameters_.resize(num_dimension, 1);
  local_dimension_ = local_dimension_ > 0 ? local_dimension_ : num_dimension;
  id_ = global_vertex_id++;

  // std::cout << "Vertex construct num_dimension: " << num_dimension
  //           << " local_dimension: " << local_dimension << " id_: " << id_
  //           << std::endl;
}
