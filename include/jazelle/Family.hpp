/**
 * @file Family.hpp
 * @brief A templated manager for a collection of banks of the same type.
 *
 * This class replaces the reflection-based Family.java. It uses C++
 * templates and a std::map to provide type-safe, efficient storage
 * and lookup of banks by their ID.
 *
 * @see hep.sld.jazelle.Family
 */

#pragma once

#include "Bank.hpp"
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace jazelle
{
    /**
     * @class Family
     * @brief A type-safe container for all banks of a specific type (e.g., PHCHRG).
     *
     * Replaces the Java Family's ArrayList and manual-linking logic with
     * a simple and fast std::map for ID-based lookup.
     *
     * @tparam T The concrete Bank type (e.g., MCHEAD, PHCHRG).
     */
    template <typename T>
    class Family
    {
        // Ensure that T is actually a type derived from Bank.
        static_assert(std::is_base_of_v<Bank, T>, "T must derive from jazelle::Bank");

    public:
        /**
         * @brief Default constructor.
         */
        Family() = default;

        /**
         * @brief Adds a new, empty bank of type T to the family.
         *
         * This creates the bank object and stores it in the map, ready
         * for its read() method to be called.
         *
         * @param id The unique ID for the new bank.
         * @return A raw pointer to the newly created bank.
         * @see hep.sld.jazelle.Family#add(int)
         */
        T* add(int32_t id)
        {
            // Create a new bank object of type T, passing the id to its constructor
            auto newBank = std::make_unique<T>(id);
            T* bankPtr = newBank.get();
            
            // Add it to our map. emplace checks for duplicates.
            auto [it, inserted] = m_banks.emplace(id, std::move(newBank));
            
            if (!inserted)
            {
                // This faithfully translates the duplicate check in Family.java
                throw std::runtime_error("Duplicate bank id " + std::to_string(id) + " in family.");
            }
            return bankPtr;
        }

        /**
         * @brief Finds a bank by its ID.
         *
         * @param id The ID of the bank to find.
         * @return A raw pointer to the bank, or nullptr if not found.
         * @see hep.sld.jazelle.Family#find(int)
         */
        T* find(int32_t id)
        {
            auto it = m_banks.find(id);
            if (it != m_banks.end())
            {
                return it->second.get();
            }
            return nullptr;
        }
        
        /**
         * @brief Finds a bank by its ID (const version).
         */
        const T* find(int32_t id) const
        {
            auto it = m_banks.find(id);
            if (it != m_banks.end())
            {
                return it->second.get();
            }
            return nullptr;
        }

        /**
         * @brief Clears all banks from this family.
         * @see hep.sld.jazelle.Family#clear()
         */
        void clear()
        {
            m_banks.clear();
        }

        /**
         * @brief Returns the number of banks in this family.
         * @return The bank count.
         * @see hep.sld.jazelle.Family#count()
         */
        size_t count() const
        {
            return m_banks.size();
        }

        // --- Iterators for range-based 'for' loops ---
        
        auto begin() { return m_banks.begin(); }
        auto end()   { return m_banks.end(); }
        auto begin() const { return m_banks.cbegin(); }
        auto end()   const { return m_banks.cend(); }

    private:
        /// @brief The storage for all banks, mapped by their ID.
        std::map<int32_t, std::unique_ptr<T>> m_banks;
    };

} // namespace jazelle