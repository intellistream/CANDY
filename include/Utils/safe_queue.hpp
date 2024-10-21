/*
 * Copyright (C) 2024 by the INTELLI team
 * Created by: Junyao Dong
 * Created on: 2024/10/14
 * Description: [Provide description here]
 */

#ifndef INTELLISTREAM_SRC_UTILS_SAFE_QUEUE_HPP_
#define INTELLISTREAM_SRC_UTILS_SAFE_QUEUE_HPP_

#include <mutex>
#include <queue>

template <typename T>
class SafeQueue {
private:
	std::queue<T> m_queue;
	std::mutex m_mutex;

public:
	SafeQueue() {
	}

	~SafeQueue() {
	}

	bool empty() {
		std::unique_lock<std::mutex> lock(m_mutex);
		return m_queue.empty();
	}

	int size() {
		std::unique_lock<std::mutex> lock(m_mutex);
		return m_queue.size();
	}

	void enqueue(T &t) {
		std::unique_lock<std::mutex> lock(m_mutex);
		m_queue.push(t);
	}

	bool dequeue(T &t) {
		std::unique_lock<std::mutex> lock(m_mutex);

		if (m_queue.empty()) {
				return false;
		}
		t = std::move(m_queue.front());

		m_queue.pop();
		return true;
	}
};

#endif //INTELLISTREAM_SRC_UTILS_SAFE_QUEUE_HPP_