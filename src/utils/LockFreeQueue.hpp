#pragma once
#include <atomic>
#include <memory>
#include <optional>

namespace jazelle {
namespace utils {

/**
 * @brief Single-producer, single-consumer lock-free queue
 * Perfect for parallel event reading where multiple readers feed one writer
 */
template<typename T>
class SPSCQueue {
public:
    explicit SPSCQueue(size_t capacity) 
        : buffer_(capacity + 1)
        , capacity_(capacity + 1)
        , head_(0)
        , tail_(0) 
    {}
    
    bool tryPush(T&& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = (head + 1) % capacity_;
        
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }
        
        buffer_[head] = std::move(item);
        head_.store(next_head, std::memory_order_release);
        return true;
    }
    
    std::optional<T> tryPop() {
        size_t tail = tail_.load(std::memory_order_relaxed);
        
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt; // Queue empty
        }
        
        T item = std::move(buffer_[tail]);
        tail_.store((tail + 1) % capacity_, std::memory_order_release);
        return item;
    }
    
    bool empty() const {
        return tail_.load(std::memory_order_acquire) == 
               head_.load(std::memory_order_acquire);
    }

private:
    std::vector<T> buffer_;
    size_t capacity_;
    alignas(64) std::atomic<size_t> head_;  // Avoid false sharing
    alignas(64) std::atomic<size_t> tail_;
};

/**
 * @brief Multi-producer, single-consumer queue using multiple SPSC queues
 */
template<typename T>
class MPSCQueue {
public:
    explicit MPSCQueue(size_t num_producers, size_t queue_capacity)
        : queues_(num_producers)
        , current_queue_(0)
    {
        for (auto& q : queues_) {
            q = std::make_unique<SPSCQueue<T>>(queue_capacity);
        }
    }
    
    bool tryPush(T&& item, size_t producer_id) {
        return queues_[producer_id]->tryPush(std::move(item));
    }
    
    std::optional<T> tryPop() {
        // Round-robin across producer queues for fairness
        size_t start = current_queue_;
        do {
            auto result = queues_[current_queue_]->tryPop();
            current_queue_ = (current_queue_ + 1) % queues_.size();
            if (result.has_value()) {
                return result;
            }
        } while (current_queue_ != start);
        
        return std::nullopt;
    }
    
    bool allEmpty() const {
        for (const auto& q : queues_) {
            if (!q->empty()) return false;
        }
        return true;
    }

private:
    std::vector<std::unique_ptr<SPSCQueue<T>>> queues_;
    size_t current_queue_;
};

} // namespace utils
} // namespace jazelle