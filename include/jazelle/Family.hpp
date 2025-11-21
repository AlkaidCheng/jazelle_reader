#pragma once
#include "Bank.hpp"
#include "BankTraits.hpp"
#include <vector>
#include <string_view>
#include <algorithm>
#include <stdexcept>

namespace jazelle
{

    // Interface: Returns generic Bank*
    class IFamily
    {
    public:
        virtual ~IFamily() = default;
        
        // Clean names!
        virtual Bank* at(size_t index) = 0;
        virtual Bank* find(int32_t id) = 0;
        virtual size_t size() const = 0;

        virtual std::string_view name() const = 0;
    };

    template <typename T>
    class Family : public IFamily
    {
        static_assert(std::is_base_of_v<Bank, T>, "T must derive from Bank");

    public:

        using BankType = T;

        std::string_view name() const override {
            return bank_name<T>;
        }

        /**
         * @brief Access a bank by its zero-based index in the storage vector.
         * @param index The index [0, size()).
         * @return Pointer to the bank, or nullptr if index is out of bounds.
         */
        T* at(size_t index) override {
            if (index < m_banks.size()) {
                return &m_banks[index];
            }
            return nullptr;
        }

        T* find(int32_t id) override {
            // Binary search is O(log N) but on contiguous memory (fast)
            auto it = std::lower_bound(m_banks.begin(), m_banks.end(), id, 
                 [](const T& bank, int32_t searchId) { return bank.getId() < searchId; });
            if (it != m_banks.end() && it->getId() == id) {
                return &(*it);
            }
            return nullptr;
        }

        size_t size() const override {
            return m_banks.size();
        }

        // Access by raw pointer for Cython
        T* add(int32_t id) {
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

        void clear() { m_banks.clear(); } // Keeps capacity, avoids reallocation next event

        /**
         * @brief Pre-allocates memory for the underlying vector.
         * Used by JazelleFile to prevent reallocations when the size is known from the TOC.
         */
        void reserve(size_t capacity) {
            m_banks.reserve(capacity);
        }

    private:
        // Contiguous memory storage. 
        // Note: T must be move-constructible (generated default is usually fine).
        std::vector<T> m_banks; 
    };
}