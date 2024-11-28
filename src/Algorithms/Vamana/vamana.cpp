//
// Created by LIUJUN on 23/11/2024.
//
#include <Algorithms/Vamana/vamana.hpp>
#include <Utils/Computation.hpp>

#include <Utils/TensorOP.hpp>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <unordered_set>
#include <vector>
#include "Algorithms/Utils/metric_type.hpp"
using namespace CANDY_ALGO;

bool Vamana::setConfig(const INTELLI::ConfigMapPtr cfg) {
  if (cfg == nullptr) {
    return false;
  }

  M_ = cfg->tryI64("M", 8, true);
  Mmax_ = cfg->tryI64("Mmax", 16, true);
  efConstruction_ = cfg->tryI64("efConstruction", 50, true);
  efSearch_ = cfg->tryI64("efSearch", 200, true);
  Mmax0_ = Mmax_ * 2;
  // set database parameters
  initialVolume_ = cfg->tryI64("initialVolume", 1000, true);
  dim_ = cfg->tryI64("vecDim", 768, true);
  dbTensor_ = torch::zeros({initialVolume_, dim_});
  size_ = 0;
  Index =0;
  alpha_ =1.2;
  return true;
}
bool Vamana::loadInitialTensor(torch::Tensor& t){

  for(int64_t idx = 0 ; idx < t.size(0); idx++){
    insert(t[idx]);
  }


  return true ;
}


bool Vamana::insert(const torch::Tensor & t){
  idx_t insert_pos = Index;

  auto vertexs = std::make_shared<vertex>(insert_pos, t);
  index_[insert_pos] = vertexs;


  if(entry_point_ == -1){
    entry_point_ = insert_pos;
  }else {
    priority_of_distAndId_Less results;
    idx_t nearest = entry_point_;
    float d_nearest = CANDY::euclidean_distance(index_[entry_point_]->vector_,t);
    greedy_update_nearest(nearest,d_nearest,t.contiguous());
    // greedy_search(entry_point_, t, results, 1);
    //
    // if(results.empty()){
    //   throw std::runtime_error("No neighbors found");
    // }
    // // priority_of_distAndId_Greater candidates;
    // auto [d_nearest,nearest] = results.top();

    add_links_starting_from(insert_pos,nearest,d_nearest);

  }


  size_ = size_ + 1 ;
  Index = Index + 1 ;


  return true ;
}
void Vamana::add_links_starting_from(idx_t start_id,idx_t nearest_id,float d_nearest) {
  priority_of_distAndId_Less link_targets;
  const int64_t k = efConstruction_;

  greedy_search(nearest_id,index_[start_id]->vector_,link_targets,k);

  const int64_t R = Mmax_;

  shrink_neighbor_list(link_targets,R);
  std::vector<idx_t> neighbors ;
  neighbors.reserve(link_targets.size());
  while(!link_targets.empty()) {
    idx_t other_id = link_targets.top().id;
    add_link(start_id,other_id);
    neighbors.push_back(other_id);
    link_targets.pop();
  }

  for (const auto other_id : neighbors) {
    add_link(other_id,start_id);
  }
}

void Vamana::add_link(idx_t src , idx_t dest) {
  const int64_t R = Mmax_ ;
  if(index_[src]->neighbors_.size() < R ) {
    index_[src]->neighbors_.push_back(index_[dest]);
    return ;
  }
  priority_of_distAndId_Less resultSet ;
  auto dist  = CANDY::euclidean_distance(index_[src]->vector_,index_[dest]->vector_);
  resultSet.emplace(dist,dest);
  for(const auto& nei : index_[src]->neighbors_) {
    dist = CANDY::euclidean_distance(nei->vector_,index_[src]->vector_);
    resultSet.emplace(dist,nei->id_);
  }
  shrink_neighbor_list(resultSet,R);

  index_[src]->neighbors_.clear();
  while(!resultSet.empty()) {
    auto [dist,idx] = resultSet.top();
    resultSet.pop();
    index_[src]->neighbors_.push_back(index_[idx]);

  }

}
void Vamana::shrink_neighbor_list(priority_of_distAndId_Less &resultSet1, int64_t R) {
  if(resultSet1.size()< R) {
    return ;
  }
  priority_of_distAndId_Greater resultSet;
  std::vector<distAndId> returnList;
  while(!resultSet1.empty()) {
    resultSet.emplace(resultSet1.top().dist,resultSet1.top().id);
    resultSet1.pop();
  }

  shrink_neighbor_list_robust(resultSet,returnList,R);

  for(distAndId& item : returnList) {
    resultSet1.emplace(item);
  }

}
void Vamana::shrink_neighbor_list_robust(priority_of_distAndId_Greater& input,vector<distAndId>&output,int64_t R) {
  const float alpha = alpha_;
  priority_of_distAndId_Less input2;
  priority_of_distAndId_Greater input1;
  vector<distAndId>V;
  while(!input.empty()) {
    auto _ = input.top();
    input.pop();
    input1.push(_);
    input2.push(_);
    V.push_back(_);
  }
  ranges::reverse(V);
  auto p_star = input2.top();
  while((output.size() < R) && (!input1.empty())) {
    auto p_prime = input1.top();
    input1.pop();
    float d_pstar_pprime = CANDY::euclidean_distance(index_[p_star.id]->vector_,index_[p_prime.id]->vector_);
    if(alpha * d_pstar_pprime > p_prime.dist) {
      output.push_back(p_prime);
    }
  }
  // while((output.size() < R) && (!V.empty())) {
  //   auto p_star = V.back();
  //   V.pop_back();
  //   output.push_back(p_star);
  //   unordered_set<idx_t>to_erase;
  //   for(auto p_prime : V ) {
  //     float d_pstar_pprime = CANDY::euclidean_distance(index_[p_star.id]->vector_,index_[p_prime.id]->vector_);
  //       if(alpha * d_pstar_pprime <= p_prime.dist) {
  //           to_erase.insert(p_prime.id);
  //       }
  //   }
  //   if(to_erase.empty()) {
  //     continue;
  //   }
  //   std::erase_if(V, [&](distAndId x) { return to_erase.contains(x.id); });
  // }
}


void Vamana::greedy_search(idx_t s , const torch::Tensor& q, priority_of_distAndId_Less& results_,
                           const int64_t k){
    priority_of_distAndId_Greater candidates;
    std::unordered_set<idx_t> visited;
    priority_of_distAndId_Less results;
    float dist = CANDY::euclidean_distance(index_[s]->vector_,q);
    candidates.emplace(dist,s);
    results.emplace(dist,s);

    visited.insert(s);
    const auto efConstruction = std::max(efConstruction_,k);
    while(!candidates.empty()){
        auto [_, id] = candidates.top();
        if(_ > results.top().dist){
          break;
        }
        candidates.pop();
        for (auto neighbors = index_[id]->neighbors_;
             const auto& neighbor : neighbors){
            if (!visited.contains(neighbor->id_)) {
              visited.insert(neighbor->id_);
              if (float d = CANDY::euclidean_distance(neighbor->vector_, q);
                  results.size() < efConstruction || d < results.top().dist){
                candidates.emplace(d, neighbor->id_);
                results.emplace(d, neighbor->id_);
                if(results.size() > efConstruction){
                  results.pop();
                }
              }
            }
        }
    }

    while(!results.empty()){
      if (!deleteList_.contains(results.top().id)) {
        results_.emplace(results.top());
        if(results_.size() == k){
          return ;
        }
      }
      results.pop();
    }
}

void Vamana::robust_prune(idx_t p , unordered_map<idx_t,float>& V,float alpha,int64_t R){
  for(auto& neighbor : index_[p]->neighbors_){
		if(V.find(neighbor->id_) == V.end()){
            float d = CANDY::euclidean_distance(index_[neighbor->id_]->vector_, index_[p]->vector_);
            V[neighbor->id_] = d;
        }
    }
    if(V.find(p) != V.end()){
        V.erase(p);
    }
	index_[p]->neighbors_.clear();

    vector<distAndId> V_;
    for(auto& [id, d] : V){
        V_.push_back({d, id});
    }
    std::sort(V_.begin(), V_.end(), [](const distAndId& a, const distAndId& b){
        return a.dist > b.dist;
    });

    while(!V.empty() ){
		idx_t q = V_.back().id;
        V_.pop_back();
        if(V.find(q) == V.end()){
            continue;
        }
        V.erase(q);
        index_[p]->neighbors_.push_back(index_[q]);
        if(index_[p]->neighbors_.size() == R){
            break;
        }
		for(auto & p_ : V_){
			if(V.find(p_.id) == V.end()){
                continue;
            }
            float d1 = CANDY::euclidean_distance(index_[p_.id]->vector_, index_[q]->vector_);
            float d2 = CANDY::euclidean_distance(index_[p_.id]->vector_, index_[p]->vector_);
            if(alpha * d1 < d2 ){
                V.erase(p_.id);
            }
		}

    }

}

bool Vamana::delete_(const idx_t idx){
    if(index_.find(idx) == index_.end()){
        return false;
    }
    deleteList_.insert(idx);
    --size_;
    if(deleteList_.size() > size_ * 0.05) {
      deleteBatch();
    }
    return true;
}

bool Vamana::delete_(torch::Tensor& t){
    idx_t idx = -1;
    for(auto& [id, vertex] : index_){
        if(vertex->vector_.equal(t)){
            idx = id;
            break;
        }
    }
    if(idx == -1){
        return false;
    }
    delete_(idx);
    return true;
}

bool Vamana::deleteTensor(torch::Tensor& t, int64_t k ){
  const std::vector<torch::Tensor> idxToDeleteTensors = searchTensor(t, k);

  // Flatten idxToDeleteTensors into a single vector of int64_t
  std::vector<int64_t> idxToDelete;
  for (const auto& tensor : idxToDeleteTensors) {
    auto tensorAccessor = tensor.accessor<int64_t, 1>();
    for (int64_t i = 0; i < tensor.size(0); ++i) {
      idxToDelete.push_back(tensorAccessor[i]);
    }
  }
  // we use mark to delete the tensor
  for (const auto& idx : idxToDelete) {
    delete_(idx);
  }

    return true;
}
bool Vamana::deleteBatch(){
    for(auto x : index_){
      auto & p = x.first;
      priority_of_distAndId_Less resultSet;
      for(auto& neighbor : x.second->neighbors_){
        if(!deleteList_.contains(neighbor->id_)) {
          float dist = CANDY::euclidean_distance(neighbor->vector_,x.second->vector_);
          resultSet.emplace(dist,neighbor->id_);
        }else {
          for(auto & del_neighbor : index_[neighbor->id_]->neighbors_){
            if(!deleteList_.contains(del_neighbor->id_)){
              float dist = CANDY::euclidean_distance(del_neighbor->vector_,x.second->vector_);
              resultSet.emplace(dist,del_neighbor->id_);
            }
          }
        }
      }
      shrink_neighbor_list(resultSet,Mmax_);
      index_[p]->neighbors_.clear();
      while(!resultSet.empty()){
        auto [dist,id] = resultSet.top();
        resultSet.pop();
        index_[p]->neighbors_.push_back(index_[id]);
      }
    }

    for(auto & id : deleteList_){
      index_.erase(id);
    }
    return true;
}

bool Vamana::insertTensor(const torch::Tensor& t){
    for(int64_t i = 0; i < t.size(0); i++){
        insert(t[i]);
        // std::cout<<i<<std::endl;
    }
    return true;
}

bool Vamana::reviseTensor(torch::Tensor& t, torch::Tensor& w) {
  if (t.size(0) > w.size(0) || t.size(1) != w.size(1)) {
    return false;
  }
  const auto idx2reviseTensors = searchTensor(t, 1);
  std::vector<int64_t> idx2revise;
  for (auto& tensor : idx2reviseTensors) {
    auto tensorAccessor = tensor.accessor<int64_t, 1>();
    for (int64_t i = 0; i < tensor.size(0); ++i) {
      auto idx = tensorAccessor[i];
      idx2revise.push_back(idx);
    }
  }
  int index = 0;
  for (const auto& idx : idx2revise) {
    delete_(idx), insert(w[index++]);
  }
  return true;
}

void Vamana::reset() {

}

std::vector<torch::Tensor> Vamana::searchTensor(const torch::Tensor& q, int64_t k) {
  const auto to_search = q.size(0);
  std::vector<torch::Tensor> to_return;
  to_return.reserve(to_search);
  if (size_ == 0)
    return {};
  std::vector<std::future<torch::Tensor>> futures;
  for (int i = 0; i < to_search; ++i) {
    auto future = std::async(std::launch::async,
                             [this, &q, i, k]() { return search(q[i], k); });
    futures.emplace_back(std::move(future));
  }
  for (auto& future : futures) {
    to_return.push_back(future.get());
  }
  return to_return;
}
torch::Tensor Vamana::search(const torch::Tensor& t, int64_t k) {
  torch::Tensor result = torch::zeros({k}, torch::kInt64);
  const auto tensor = t.contiguous();


  const idx_t & ids = entry_point_ ;
  float dist = CANDY::euclidean_distance(index_[entry_point_]->vector_,tensor);
  greedy_update_nearest(ids,dist,tensor);
  // priority_of_distAndId_Less results_;
  // greedy_search(entry_point_,tensor,results_,1);
  // if(results_.empty()) {
  //   throw runtime_error("results_ empty");
  // }
  // auto [_,ids] = results_.top();
  // while(!results_.empty()) results_.pop();
  // greedy_search(id,tensor,results_,efSearch_);
  priority_of_distAndId_Less top_candidates = search_base_layerST(ids,tensor,std::max(efSearch_,k));
  while (top_candidates.size() > k) {
    top_candidates.pop();
  }
  long index = k - 1;
  while (!top_candidates.empty()) {
    auto [_, id] = top_candidates.top();
    result.index_put_({index--}, id);
    top_candidates.pop();
  }
  // priority_of_distAndId_Less top_candidates_all ;
  // for(const auto& node : index_ ) {
  //
  //     float dist = CANDY::euclidean_distance(node.second->vector_,tensor);
  //     if(top_candidates_all.size() == k && dist < top_candidates_all.top().dist) {
  //       top_candidates.pop();
  //     }
  //     top_candidates_all.emplace(node.second->id_,dist);
  // }
  // for(int i = 0 ; i < k ; i++) {
  //   auto [_,id] = top_candidates_all.top();
  //   result.index_put_({i},id);
  //   top_candidates_all.pop();
  // }
  return result;
}

Vamana::priority_of_distAndId_Less Vamana::search_base_layerST(idx_t ep,
                                                     const torch::Tensor& q,
                                                     const long ef) {
  priority_of_distAndId_Less top_candidates;
  priority_of_distAndId_Less candidates;
  float lowerBound = CANDY::euclidean_distance(dbTensor_[ep], q);
  candidates.emplace(-lowerBound, ep);
  top_candidates.emplace(lowerBound, ep);
  std::unordered_set<idx_t> visited;
  visited.insert(ep);
  while (!candidates.empty()) {
    auto [d, i] = candidates.top();
    if (const auto dist2query = -d;
        dist2query > lowerBound && top_candidates.size() == ef) {
      break;
        }
    candidates.pop();
    for (const auto& neighbor : index_[i]->neighbors_) {
      if (visited.contains(neighbor->id_))
        continue;
      visited.insert(neighbor->id_);
      if (const auto dist = CANDY::euclidean_distance(
              neighbor->vector_, q.contiguous());
          top_candidates.size() < ef || dist < lowerBound) {
        candidates.emplace(-dist, neighbor->id_);
        top_candidates.emplace(dist, neighbor->id_);
        if (top_candidates.size() > ef) {
          top_candidates.pop();
        }
        if (!top_candidates.empty()) {
          lowerBound = top_candidates.top().dist;
        }
          }
    }
  }
  return top_candidates;
}

void Vamana::greedy_update_nearest(idx_t nearest , float & d_nearest,const torch::Tensor& q) {
  for(;;) {
    const idx_t prev_nearest = nearest;
    for (const auto & v : index_[nearest]->neighbors_) {
      if (const float d = CANDY::euclidean_distance(v->vector_, q); d < d_nearest) {
        d_nearest = d;
        nearest = v->id_;
      }
    }
    if(nearest == prev_nearest) {
      return ;
    }
  }
}

