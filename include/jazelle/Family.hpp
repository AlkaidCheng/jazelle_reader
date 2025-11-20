#pragma once
#include "Bank.hpp"
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace jazelle
{
    template <typename T>
    class Family
    {
        static_assert(std::is_base_of_v<Bank, T>, "T must derive from Bank");

    public:
        // Access by raw pointer for Cython
        T* add(int32_t id)
        {
            // Optimization: Most banks are added sequentially. 
            // If id > last_id, just push_back.
            if (m_banks.empty() || id > m_banks.back().getId()) {
                m_banks.emplace_back(id);
                return &m_banks.back();
            }
            
            // Fallback for out-of-order IDs (rare in Jazelle)
            auto it = std::lower_bound(m_banks.begin(), m_banks.end(), id, 
                [](const T& bank, int32_t searchId) { return bank.getId() < searchId; });
            
            if (it != m_banks.end() && it->getId() == id) {
                 throw std::runtime_error("Duplicate bank id " + std::to_string(id));
            }
            return &(*m_banks.emplace(it, id));
        }

        T* find(int32_t id)
        {
            // Binary search is O(log N) but on contiguous memory (fast)
            auto it = std::lower_bound(m_banks.begin(), m_banks.end(), id, 
                 [](const T& bank, int32_t searchId) { return bank.getId() < searchId; });
            
            if (it != m_banks.end() && it->getId() == id) {
                return &(*it);
            }
            return nullptr;
        }

        /**
         * @brief Access a bank by its zero-based index in the storage vector.
         * @param index The index [0, count()).
         * @return Pointer to the bank, or nullptr if index >= count().
         */
        T* at(size_t index)
        {
            if (index < m_banks.size()) {
                return &m_banks[index];
            }
            return nullptr;
        }

        void clear() { m_banks.clear(); } // Keeps capacity, avoids reallocation next event
        size_t count() const { return m_banks.size(); }

    private:
        // Contiguous memory storage. 
        // Note: T must be move-constructible (generated default is usually fine).
        std::vector<T> m_banks; 
    };
}