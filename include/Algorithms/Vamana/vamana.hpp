//
// Created by LIUJUN on 21/11/2024.
//

#ifndef VAMANA_HPP
#define VAMANA_HPP
#include <torch/torch.h>
#include <Utils/ConfigMap.hpp>
#include <unordered_set>
#include <vector>
#include "Algorithms/ANNSBase.hpp"
#include "vamana_base.hpp"

namespace CANDY_ALGO {
  class Vamana final : public ANNSBase{
  public:
    Vamana() = default;
    ~Vamana() override = default;
    bool setConfig(INTELLI::ConfigMapPtr cfg) override;
    void reset() override;

    struct distAndId {
      float dist;
      idx_t id;

      distAndId(const float d, const idx_t i) : dist(d), id(i) {}

      bool operator<(const distAndId& other) const { return dist < other.dist; }

      bool operator>(const distAndId& other) const { return dist > other.dist; }

      bool operator==(const distAndId& other) const{return id == other.id;}
    };

    struct compByDistLess {
      bool operator()(const distAndId& a, const distAndId& b) const {
        return a.dist < b.dist;
      }
    };

    struct compByDistGreater {
      bool operator()(const distAndId& a, const distAndId& b) const {
        return a.dist > b.dist;
      }
    };
	//堆顶距离最小
    using priority_of_distAndId_Greater =
      std::priority_queue<distAndId, std::vector<distAndId>, compByDistGreater>;
    //堆顶距离最大
    using priority_of_distAndId_Less =
      std::priority_queue<distAndId, std::vector<distAndId>, compByDistLess>;

    bool insertTensor(const torch::Tensor& t) override;
    auto deleteTensor(torch::Tensor& t, int64_t k ) -> bool override;
    bool reviseTensor(torch::Tensor& t, torch::Tensor& w) override;
    std::vector<torch::Tensor> searchTensor(const torch::Tensor& q,
                                         int64_t k) override;
    bool loadInitialTensor(torch::Tensor& t) override;

    float getAlpha() const { return alpha_; }
    void setAlpha(float alpha) { alpha_ = alpha; }
  protected:
    int64_t M_{};
    int64_t Mmax_{};
    int64_t Mmax0_{};
    int64_t efConstruction_{};
    int64_t efSearch_{};
    float alpha_{};
    long size_ = -1;  // how many vectors in the db
    long Index = -1;  // 已分配的索引值
    long initialVolume_{};  //
    long dim_{};
    static constexpr int expandStep = 1000;

    torch::Tensor dbTensor_;
    unordered_map<idx_t, VertexPtr> index_;
    unordered_set<idx_t> deleteList_;
    long entry_point_ = -1;
    long max_level_ = 0;
//    std::vector<Vertex> vertexes_;
//    std::vector<idx_t> free_list_;
//    std::default_random_engine level_generator_;

    bool insert(const torch::Tensor& tensor);
    bool delete_(const idx_t idx);
    bool delete_(torch::Tensor& t);
    bool deleteBatch();
    void greedy_search(idx_t s , const torch::Tensor& q, priority_of_distAndId_Less& results_, int64_t k);
    void robust_prune(idx_t p , unordered_map<idx_t,float> &V,float alpha,int64_t R);
    void add_links_starting_from(idx_t start_id,idx_t nearest_id,float d_nearest);
    void shrink_neighbor_list(priority_of_distAndId_Less &resultSet1, int64_t R);
    void shrink_neighbor_list_robust(priority_of_distAndId_Greater& input,vector<distAndId>&output,int64_t R);
    void add_link(idx_t src , idx_t dest);
    torch::Tensor search(const torch::Tensor& t, int64_t k);
    priority_of_distAndId_Less search_base_layerST(idx_t ep,
                                                   const torch::Tensor&q, long ef);
    void greedy_update_nearest(idx_t nearest , float & d_nearest,const torch::Tensor& q);
  };
}
#endif //VAMANA_HPP