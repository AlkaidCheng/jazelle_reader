/**
 * @file PHCHRG.hpp
 * @brief Definition of the PHCHRG (Charged Particle) bank.
 *
 * This is a large, complex bank with many fields.
 *
 * @see hep.sld.jazelle.family.PHCHRG
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
     * @struct PHCHRG
     * @brief The Charged Particle bank.
     */
    struct PHCHRG : public Bank
    {
        // --- Member Variables ---
        // We use std::array for fixed-size C-style arrays
        
        std::array<float, 6>  hlxpar;   // hlxpar(0-5)
        std::array<float, 15> dhlxpar;  // dhlxpar(0-14)
        float   bnorm;
        float   impact;
        float   b3norm;
        float   impact3;
        int16_t charge;
        int16_t smwstat;
        int32_t status;
        float   tkpar0;
        std::array<float, 5>  tkpar;    // tkpar(0-4)
        std::array<float, 15> dtkpar;   // dtkpar(0-14)
        float   length;
        float   chi2dt;
        int16_t imc;
        int16_t ndfdt;
        int16_t nhit;
        int16_t nhite;
        int16_t nhitp;
        int16_t nmisht;
        int16_t nwrght;
        int16_t nhitv;
        float   chi2;
        float   chi2v;
        int32_t vxdhit;
        int16_t mustat;
        int16_t estat;
        int32_t dedx;
        
        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHCHRG(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (236).
         * @see hep.sld.jazelle.family.PHCHRG#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };

} // namespace jazelle