/**
 * @file JazelleEvent.hpp
 * @brief The main event data store
 */

#pragma once

#include "Family.hpp"
#include "banks/AllBanks.hpp"
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace jazelle
{

    /**
     * @class JazelleEvent
     * @brief Modernized event container using tuples to reduce boilerplate.
     */
    class JazelleEvent
    {
    public:

        using BankTypes = std::tuple<
            MCHEAD, MCPART, PHPSUM, PHCHRG, PHKLUS, 
            PHWIC, PHCRID, PHKTRK, PHKELID
        >;

        // Public Header
        IEVENTH ieventh;

        JazelleEvent() = default;

        /**
         * @brief Clears all families. 
         * Uses a C++17 fold expression to iterate the tuple at compile time.
         */
        void clear()
        {
            // "Visit every member of the tuple and call .clear()"
            std::apply([](auto&... families) {
                (families.clear(), ...); 
            }, m_families);
            // Note: ieventh is intentionally not cleared here,
            // as it will be overwritten during the next read operation.
        }

        /**
         * @brief Universal Accessor.
         * Replaces findMCHEAD(), findPHCHRG(), etc.
         * Usage: event.get<PHCHRG>().find(id);
         */
        template <typename T>
        Family<T>& get()
        {
            return std::get<Family<T>>(m_families);
        }

        /**
         * @brief Adds a bank by string name.
         * Iterates over the tuple types at compile-time to find the matching string.
         */
        Bank* add(std::string_view name, int32_t id)
        {
            Bank* result = nullptr;

            // Iterate over the tuple elements
            std::apply([&](auto&... families) {
                // Fold expression: Check every family
                ((bank_name<typename std::decay_t<decltype(families)>::BankType> == name 
                  ? (result = families.add(id), true) // Found match: add and stop
                  : false) || ...); 
            }, m_families);

            if (result) return result;
            throw std::runtime_error("Unknown bank family name: " + std::string(name));
        }

        IFamily* getFamily(std::string_view name)
        {
            IFamily* result = nullptr;
            // Compile-time iteration over the tuple
            std::apply([&](auto&... families) {
                // Check matching name
                ((families.name() == name 
                    ? (result = &families, true) 
                    : false) || ...); 
            }, m_families);
            
            if (!result) {
                throw std::invalid_argument("Unknown family: " + std::string(name));
            }
            return result;
        }

        /**
         * @brief Returns a list of IFamily* for Python reflection.
         */
        std::vector<IFamily*> getFamilies()
        {
            std::vector<IFamily*> result;
            result.reserve(std::tuple_size_v<decltype(m_families)>);
        
            std::apply([&](auto&... families) {
                (result.push_back(&families), ...);
            }, m_families);
        
            return result;
        }

        static std::vector<std::string_view> getKnownBankNames()
        {
            std::vector<std::string_view> names;
            names.reserve(std::tuple_size_v<BankTypes>);
            
            // Iterate generic tuple types, not instances
            // We use a dummy tuple pointer to drive the template expansion
            apply_to_types<BankTypes>([&](auto tag) {
                using T = typename decltype(tag)::type;
                names.emplace_back(bank_name<T>);
            });
            
            return names;
        }

    private:
        // 3. STORAGE: Transform BankTypes tuple (T...) into tuple of Families (Family<T>...)
        //    Helper meta-function to wrap types
        template <typename... Ts>
        static std::tuple<Family<Ts>...> make_family_tuple(std::tuple<Ts...>*);

        using FamiliesTuple = decltype(make_family_tuple((BankTypes*)nullptr));

        // Helper tag to carry the type information
        template <typename T> struct type_tag { using type = T; };

        // Implementation details for type iteration
        // FIX: Use std::tuple_element_t to access types by index, avoiding instantiation.
        template <typename Tuple, typename Func, std::size_t... Is>
        static void apply_to_types_impl(Func f, std::index_sequence<Is...>) {
            (f(type_tag<std::tuple_element_t<Is, Tuple>>{}), ...);
        }

        // Public helper to iterate types in a tuple without an instance
        template <typename Tuple, typename Func>
        static void apply_to_types(Func f) {
            apply_to_types_impl<Tuple>(f, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
        }
        
        FamiliesTuple m_families;
    };

} // namespace jazelle