/**
 * @file MCPART.hpp
 * @brief Definition of the MCPART (Monte Carlo Particle) bank.
 *
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <array>

namespace jazelle
{
    class DataBuffer; // Forward-declaration
    class JazelleEvent; // Forward-declaration

    /**
     * @struct MCPART
     * @brief Holds the Monte Carlo truth information for a single particle.
     */
    struct MCPART : public Bank
    {
        // --- Member Variables ---

        /// "X,Y,Z momentum of particle at track origin"
        std::array<float, 3> p;
        
        float   e;        ///< "Energy of particle"
        float   ptot;     ///< "Total momentum at track origin"
        int32_t ptype;    ///< "Particle type" (PDG code, etc.)
        float   charge;   ///< "Charge of particle"
        int32_t origin;   ///< "Where did this particle come from/go to" (bitmask)
        
        /// "X,Y,Z of termination"
        std::array<float, 3> xt;
        
        int32_t parent_id; ///< "Key giving parent particle" (ID of another MCPART)
        MCPART* parent = nullptr;

        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit MCPART(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (48).
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };

} // namespace jazelle