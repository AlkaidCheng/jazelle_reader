/**
 * @file PHCRID.hpp
 * @brief Definition of the PHCRID (Cerenkov Ring Imaging Detector) bank.
 *
 * This is a variable-length bank that contains LIQ and/or GAS
 * hypothesis data (CRIDHYP).
 *
 * @see hep.sld.jazelle.family.PHCRID
 */

#pragma once

#include "../Bank.hpp"
#include "CRIDHYP.hpp" // Includes PIDVEC.hpp
#include <cstdint>

namespace jazelle
{
    class DataBuffer; // Forward-declaration
    class JazelleEvent; // Forward-declaration

    /**
     * @struct PHCRID
     * @brief The Cerenkov Ring Imaging Detector bank (variable-length).
     */
    struct PHCRID : public Bank
    {
        // --- Member Variables ---
        int32_t ctlword; // Control word read at the start
        float   norm;    // Normalization value
        int16_t rc;
        int16_t geom;
        int16_t trkp;
        int16_t nhits;
        
        CRIDHYP liq;     ///< LIQ hypothesis data
        CRIDHYP gas;     ///< GAS hypothesis data
        PIDVEC  llik;    ///< Combined likelihood vector

        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHCRID(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (variable).
         * @see hep.sld.jazelle.family.PHCRID#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };

} // namespace jazelle