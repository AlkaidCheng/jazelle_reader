/**
 * @file CRIDHYP.hpp
 * @brief Helper struct for CRID hypotheses (LIQ/GAS).
 *
 * Corresponds to CRIDHYP.java. This struct is variable-sized
 * depending on whether it's a "full" hypothesis or not.
 *
 * @see hep.sld.jazelle.family.CRIDHYP
 */

#pragma once

#include "PIDVEC.hpp"
#include <cstdint>
#include <optional>

namespace jazelle
{
    class DataBuffer; // Forward-declaration

    /**
     * @struct CRIDHYP
     * @brief A struct holding CRID hypothesis data.
     * Used by PHCRID for both LIQ and GAS components.
     */
    struct CRIDHYP
    {
        // --- Member Variables ---
        // Public for easy access, populated by read()
        
        bool m_full; ///< Was this read as a "full" hypothesis?
        
        /// Likelihood vector (only present if m_full is true)
        std::optional<PIDVEC> llik;
        
        int16_t rc;
        int16_t nhits;
        int32_t besthyp;
        int16_t nhexp;
        int16_t nhfnd;
        int16_t nhbkg;
        int16_t mskphot;

        /**
         * @brief Default constructor.
         */
        CRIDHYP() = default;

        /**
         * @brief Reads the hypothesis data from the buffer.
         *
         * This method implements the variable-length logic from the Java
         * original. It populates all member variables based on the 'full' flag.
         *
         * @param data The raw data buffer.
         * @param offset The starting offset for this data.
         * @param full Whether to read the full 36-byte or 4-byte version.
         * @return The number of bytes read (36 or 4).
         * @see hep.sld.jazelle.family.CRIDHYP#read(DataBuffer, int, boolean)
         */
        int32_t read(const DataBuffer& data, int32_t offset, bool full);
    };

} // namespace jazelle