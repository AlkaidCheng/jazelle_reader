// In src/utils/ObjectPool.hpp
#pragma once
#include <vector>
#include <memory>
#include <mutex>

namespace jazelle {
namespace utils {

/**
 * @brief Simple object pool to avoid repeated allocations
 */
template<typename T>
class ObjectPool {
public:
    explicit ObjectPool(size_t initial_size = 0) {
        pool_.reserve(initial_size);
        for (size_t i = 0; i < initial_size; ++i) {
            pool_.push_back(std::make_unique<T>());
        }
    }
    
    std::unique_ptr<T> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.empty()) {
            return std::make_unique<T>();
        }
        auto obj = std::move(pool_.back());
        pool_.pop_back();
        return obj;
    }
    
    void release(std::unique_ptr<T> obj) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(std::move(obj));
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }

private:
    std::vector<std::unique_ptr<T>> pool_;
    mutable std::mutex mutex_;
};

} // namespace utils
} // namespace jazelle