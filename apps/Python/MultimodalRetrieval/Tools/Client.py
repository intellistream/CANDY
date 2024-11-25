import json
import os
import sys

import torch
from pycandy import VectorDB



class DBClient:
    def __init__(self, vec_dim, search_algorithm, config):
        self.search_algorithm = search_algorithm
        self.vec_dim = vec_dim
        self.config = config
        self.db = VectorDB(vec_dim, search_algorithm, config)

    def add_tensor(self, tensor_data):
        self.db.insert_tensor(tensor_data.clone())

    def load_batch_tensor(self, tensor_data):
        for tensor in tensor_data:
            self.db.insert_tensor(tensor.clone())
        return True

    def get_tensor(self, query_text):
        results = self.db.query_nearest_tensors(query_text.clone(), k=1)
        if results:
            return results[0]
        else:
            print(f"Error fetching tensor for query: '{query_text}'")
            return None

    def get_batch_tensors(self, query_texts):
        results = []
        for query_text in query_texts:
            result = self.db.query_nearest_tensors(query_text.clone(), k=1)
            if result:
                results.append(result)
            else:
                print(f"Error fetching tensor for query: '{query_text}'")
                results.append(None)  # 如果没有找到结果，可以添加 None 或其他占位符
        return results

