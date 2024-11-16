/*
* Copyright (C) 2024 by the INTELLI team
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */
#ifndef COMPUTATION_H
#define COMPUTATION_H

#include <cmath>  // For std::sqrt
#include <numeric>
#include <vector>
#include <torch/torch.h>
namespace CANDY {
typedef enum AMMTYPE{NONE_AMM,AMM_CRS,AMM_SMPPCA}AMMTYPE;
inline float computeL2Distance(const float* a, const float* b,
                               const size_t size) {
  return std::inner_product(
      a, a + size, b, 0.0f, std::plus<float>(),
      [](const float x, const float y) { return (x - y) * (x - y); });
}

inline float computeL2Distance(const std::vector<float>& a,
                               const std::vector<float>& b) {
  return computeL2Distance(a.data(), b.data(), a.size());
}

inline float euclidean_distance(const std::vector<float>& a,
                                const std::vector<float>& b) {
  return std::sqrt(computeL2Distance(a, b));
}

inline float euclidean_distance(const torch::Tensor& a,
                                const torch::Tensor& b) {
  return torch::norm(a - b).item<float>();
}

//Coordinate-wise Random Sampling, CRS
inline torch::Tensor amm_crs(torch::Tensor &a, torch::Tensor &b, int64_t ss) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "amm_crs:Both a and b must be 2D tensors.");
  TORCH_CHECK(a.size(1) == b.size(0), "amm_crs:Shape mismatch: a.size(1) must equal b.size(0).");

  int64_t n = a.size(1);
  ss = std::min(ss, n); // Ensure ss is within bounds
  // Probability distribution
  torch::Tensor probs = torch::ones(n) / n;  // default: uniform
  auto crsIndices = torch::multinomial(probs, ss, true);
  auto A_sampled = a.index_select(1, crsIndices);
  auto B_sampled = b.index_select(0, crsIndices);
  return torch::matmul(A_sampled, B_sampled);
}

//Subspace-preserving Projection PCA, SMP-PCA
//using Gaussian distribution to conduct tilde matrix
inline torch::Tensor amm_smppca(torch::Tensor &A, torch::Tensor &B, int64_t sketchsize) {
  // Step 1: Input A:n1*d B:d*n2
  A = A.t(); // d*n1
  TORCH_CHECK(A.size(0) == B.size(0), "amm_smppca:Shape mismatch: A and B must have the same feature dimension.");

  int64_t d = A.size(0);
  int64_t n1 = A.size(1);
  int64_t n2 = B.size(1);
  int64_t k = (int64_t) sketchsize;

  // Step 2: Get sketched matrix
  torch::Tensor pi = 1 / std::sqrt(k) * torch::randn({k, d}); // Gaussian sketching matrix
  torch::Tensor A_tilde = torch::matmul(pi, A); // k*n1
  torch::Tensor B_tilde = torch::matmul(pi, B); // k*n2

  torch::Tensor A_tilde_B_tilde = torch::matmul(A_tilde.t(), B_tilde);

  // Step 3: Compute column norms
  // 3.1 column norms of A and B
  torch::Tensor col_norm_A = torch::linalg::vector_norm(A, 2, {0}, false, c10::nullopt); // ||Ai|| for i in [n1]
  torch::Tensor col_norm_B = torch::linalg::vector_norm(B, 2, {0}, false, c10::nullopt); // ||Bj|| for j in [n2]

  // 3.2 column norms of A_tilde and B_tilde
  torch::Tensor col_norm_A_tilde = torch::linalg::vector_norm(A_tilde, 2, {0}, false,
                                                              c10::nullopt); // ||Ai|| for i in [n1]
  torch::Tensor col_norm_B_tilde = torch::linalg::vector_norm(B_tilde, 2, {0}, false,
                                                              c10::nullopt); // ||Bj|| for j in [n2]

  // Step 4: Compute M_tilde
  torch::Tensor col_norm_A_col_norm_B = torch::matmul(col_norm_A.reshape({n1, 1}), col_norm_B.reshape({1, n2}));

  torch::Tensor col_norm_A_tilde_col_norm_B_tilde =
      torch::matmul(col_norm_A_tilde.reshape({n1, 1}), col_norm_B_tilde.reshape({1, n2}));
  torch::Tensor mask = (col_norm_A_tilde_col_norm_B_tilde == 0);
  col_norm_A_tilde_col_norm_B_tilde.masked_fill_(mask, 1e-6); // incase divide by 0 in next step

  torch::Tensor ratio = torch::div(col_norm_A_col_norm_B, col_norm_A_tilde_col_norm_B_tilde);

  torch::Tensor M_tilde = torch::mul(A_tilde_B_tilde, ratio);

  return M_tilde;
}

//compute Dot_product_similarity via AMM
//tensor a is the database 2D-tensor,while b is the transpose of quested tensor
inline torch::Tensor AMM_Compute_DotproductSimilarity(torch::Tensor &a, torch::Tensor &b, int64_t ss, AMMTYPE ammtype) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "AMM_Compute:Inputs a and b must be 2D tensors.");
  TORCH_CHECK(a.size(1) == b.size(0), "AMM_Compute:Shape mismatch: a.size(1) must equal b.size(0).");

  switch (ammtype) {
    case NONE_AMM:return torch::matmul(a, b);
    case AMM_CRS: return amm_crs(a, b, ss);
    case AMM_SMPPCA: return amm_smppca(a, b, ss);
  }
  return torch::matmul(a, b);
}

// From dot product similarity to L2 distance
// TOO SLOW!
inline torch::Tensor pairwise_euclidean_distance(torch::Tensor A, torch::Tensor B, AMMTYPE ammtype,int64_t sketchsize) {
  // ensure 2D tensor input
  A = A.contiguous();// n x d
  torch::Tensor A_tran = A.t().contiguous();//d x n
  B = B.contiguous();// d x m
  torch::Tensor B_tran = B.t().contiguous();// m x d
  torch::Tensor A_norm,B_norm,AB_dot;
  // Step 1,2: compute squared norms and dot product similarity
  A_norm = torch::sum(A * A, 1).reshape({-1, 1}); // ||A_i||^2, shape: n x d
  B_norm = torch::sum(B * B, 0).reshape({1, -1}); // ||B_j||^2, shape: d x m
  if (!ammtype) {
    AB_dot = torch::matmul(A, B); // shape: n x m
  }
  else {
    AB_dot=AMM_Compute_DotproductSimilarity(A,B,sketchsize,ammtype);
  }
  std::cout<<"AB_dot:"<<AB_dot.sizes();
  // Step 3: calculate squared L2 distance
  torch::Tensor D_squared = A_norm + B_norm - 2 * AB_dot; // shape: n x m

  // Step 4: ensure the distance is positive
  D_squared = torch::clamp(D_squared, 0);

  // Step 5: return L2 distance 2D-tensor
  return torch::sqrt(D_squared); // matrix of l2-distance
}

}  // namespace CANDY
#endif  // COMPUTATION_H
