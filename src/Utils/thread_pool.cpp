/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Shuhao Zhang
 * Created on: 2024/10/9
 * Description: [Provide description here]
 */

#include <Utils/thread_pool.hpp>

void ThreadPool::init() { 
  for (int i = 0; i < m_threads.size(); i++) { 
    m_threads[i] = std::thread(thread_worker(this, i));
  }
}

void ThreadPool::shutdown() { 
  m_shutdown = true;
  m_conditional_lock.notify_all();

  for (int i = 0; i < m_threads.size(); i++) {
    if (m_threads[i].joinable()) { 
      m_threads[i].join();
    }
  }
}

void ThreadPool::thread_worker::operator()() { 
  std::function<void()> func;
  bool dequeued;

  while (!m_pool->m_shutdown) { 
    {
      std::unique_lock<std::mutex> lock(m_pool->m_conditional_mutex);
      if (m_pool->m_queue.empty()) {
        m_pool->m_conditional_lock.wait(lock);
      }
      dequeued = m_pool->m_queue.dequeue(func);          
    }

    if (dequeued) { 
      func();
    }
  }
}
