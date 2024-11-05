/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */

#ifndef INTELLISTREAM_SRC_UTILS_THREAD_POOL_HPP_
#define INTELLISTREAM_SRC_UTILS_THREAD_POOL_HPP_

#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "safe_queue.hpp"

class ThreadPool {
 private:
  bool m_shutdown;
  SafeQueue<std::function<void()>> m_queue;
  std::vector<std::thread> m_threads;
  std::mutex m_conditional_mutex;
  std::condition_variable m_conditional_lock;

  class thread_worker {
   private:
    int m_id;
    ThreadPool* m_pool;

   public:
    thread_worker(ThreadPool* pool, const int id) : m_pool(pool), m_id(id) {}

    void operator()();
  };

 public:
  ThreadPool(const int n_threads)
      : m_threads(std::vector<std::thread>(n_threads)), m_shutdown(false) {}

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;

  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  // init thread pool
  void init();

  // wait until threads finish their current task and shutdowns the pool
  void shutdown();

  // submit a function to be executed asynchronously by the pool
  template <typename F, typename... Args>
  auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
    // crete a function with bounded parameters ready to execute
    std::function<decltype(f(args...))()> func =
        std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    auto task_ptr =
        std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);
    std::function<void()> wrapper_func = [task_ptr]() {
      (*task_ptr)();
    };

    m_queue.enqueue(wrapper_func);
    m_conditional_lock.notify_one();

    return task_ptr->get_future();
  }
};

#endif  //INTELLISTREAM_SRC_UTILS_THREAD_POOL_HPP_
