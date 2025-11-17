/**
 * @file PHPSUM.hpp
 * @brief Definition of the PHPSUM (Particle Sum) bank.
 *
 * @see hep.sld.jazelle.family.PHPSUM
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <cmath> // For std::sqrt

namespace jazelle
{
    class DataBuffer; // Forward-declaration
    class JazelleEvent; // Forward-declaration
    
    /**
     * @struct PHPSUM
     * @brief The Particle Sum bank.
     */
    struct PHPSUM : public Bank
    {
        // --- Member Variables ---
        float   px;  // X momentum of particle at track origin
        float   py;  // Y momentum of particle at track origin
        float   pz;  // Z momentum of particle at track origin
        float   x;  // X position corresponding to momentum measurement
        float   y;  // Y position corresponding to momentum measurement
        float   z;  // Z position corresponding to momentum measurement
        float   charge;  // Particle Charge
        int32_t status;  // Bits (e.g., CUTFLAG)

        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHPSUM(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (32).
         * @see hep.sld.jazelle.family.PHPSUM#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;

        /**
         * @brief Calculates the total momentum.
         * @return The total momentum.
         * @see hep.sld.jazelle.family.PHPSUM#getPTot()
         */
        double getPTot() const
        {
            return std::sqrt(static_cast<double>(px) * px +
                             static_cast<double>(py) * py +
                             static_cast<double>(pz) * pz);
        }
    };

} // namespace jazelle