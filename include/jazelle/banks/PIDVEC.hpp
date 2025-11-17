/**
 * @file PIDVEC.hpp
 * @brief Helper struct for Particle ID likelihoods.
 *
 * Corresponds to PIDVEC.java. This is a simple data-only struct
 * used by CRIDHYP and PHCRID.
 *
 * @see hep.sld.jazelle.family.PIDVEC
 */

#pragma once

#include <cstdint>

namespace jazelle
{
    class DataBuffer; // Forward-declaration

    /**
     * @struct PIDVEC
     * @brief A struct holding 5 PID likelihood values.
     */
    struct PIDVEC
    {
        float e;  ///< Electron likelihood
        float mu; ///< Muon likelihood
        float pi; ///< Pion likelihood
        float k;  ///< Kaon likelihood
        float p;  ///< Proton likelihood

        /**
         * @brief Default constructor (all values uninitialized).
         */
        PIDVEC() = default;

        /**
         * @brief Deserialization constructor.
         * Reads the 5 float values from the data buffer.
         * @param data The raw data buffer.
         * @param offset The starting offset to read from.
         * @see hep.sld.jazelle.family.PIDVEC#PIDVEC(DataBuffer, int)
         */
        PIDVEC(const DataBuffer& data, int32_t offset);

        /**
         * @brief Combination constructor.
         * Creates a new PIDVEC by summing LIQ and GAS hypotheses.
         * @param liq Pointer to the LIQ PIDVEC (can be null).
         * @param gas Pointer to the GAS PIDVEC (can be null).
         * @param norm The normalization value.
         * @see hep.sld.jazelle.family.PIDVEC#PIDVEC(PIDVEC, PIDVEC, float)
         */
        PIDVEC(const PIDVEC* liq, const PIDVEC* gas, float norm);
    };

} // namespace jazelle