/**
 * @file PHWIC.hpp
 * @brief Definition of the PHWIC (Warm Iron Calorimeter) bank.
 *
 * @see hep.sld.jazelle.family.PHWIC
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
     * @struct PHWIC
     * @brief The Warm Iron Calorimeter bank.
     */
    struct PHWIC : public Bank
    {
        // --- Member Variables ---
        int16_t idstat;
        int16_t nhit;
        int16_t nhit45;
        int16_t npat;
        int16_t nhitpat;
        int16_t syshit;
        float   qpinit;
        float   t1;
        float   t2;
        float   t3;
        int32_t hitmiss;
        float   itrlen;
        int16_t nlayexp;
        int16_t nlaybey;
        float   missprob;
        int32_t phwicid;
        int16_t nhitshar;
        int16_t nother;
        int32_t hitsused;
        std::array<float, 3>  pref1;    // pref1(0-2)
        std::array<float, 4>  pfit;     // pfit(0-3)
        std::array<float, 10> dpfit;    // dpfit(0-9)
        float   chi2;
        int16_t ndf;
        int16_t punfit;
        float   matchChi2;
        int16_t matchNdf;
        // Note: Java file ends at offset 136, 140-byte size implies 4 bytes spare
        
        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHWIC(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (140).
         * @see hep.sld.jazelle.family.PHWIC#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };

} // namespace jazelle