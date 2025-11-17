/**
 * @file PHKLUS.hpp
 * @brief Definition of the PHKLUS (Calorimeter Cluster) bank.
 *
 * @see hep.sld.jazelle.family.PHKLUS
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
     * @struct PHKLUS
     * @brief The Calorimeter Cluster bank.
     */
    struct PHKLUS : public Bank
    {
        // --- Member Variables ---
        int32_t status;
        float   eraw;
        float   cth;
        float   wcth;
        float   phi;
        float   wphi;
        std::array<float, 8> elayer; // elayer(0-7)
        int32_t nhit2;
        float   cth2;
        float   wcth2;
        float   phi2;
        float   whphi2; // Note: Java name wHphi2, likely typo
        int32_t nhit3;
        float   cth3;
        float   wcth3;
        float   phi3;
        float   wphi3;
        
        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHKLUS(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (96).
         * @see hep.sld.jazelle.family.PHKLUS#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };

} // namespace jazelle