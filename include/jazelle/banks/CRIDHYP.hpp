/**
 * @file CRIDHYP.hpp
 * @brief Helper struct for Cherenkov Ring Imaging Detector (CRID) hypotheses.
 *
 * This struct evaluates particle identification 
 * (PID) by counting expected vs. found photons in either the Liquid or Gas radiators.
 */

#pragma once

#include "PIDVEC.hpp"
#include <cstdint>
#include <optional>

namespace jazelle
{
    class DataBuffer;
    
    /**
     * @struct CRIDHYP
     * @brief A struct holding CRID hypothesis data.
     * Used by PHCRID for both LIQ and GAS components.
     */
    struct CRIDHYP
    {
        bool m_full; ///< True if the full 36-byte hypothesis is present, False if just 4 bytes

        /// Log-likelihood vector for (e, mu, pi, k, p) - only present if m_full is true
        std::optional<PIDVEC> llik;
        
        int16_t rc;       ///< Return Code / Status flag for the CRID reconstruction
        int16_t nhits;    ///< Total number of Cherenkov photons (hits) associated with this ring
        int32_t besthyp;  ///< Best particle hypothesis ID (e.g., matching PDG codes or internal PID)
        int16_t nhexp;    ///< Number of expected hits for the best hypothesis
        int16_t nhfnd;    ///< Number of expected hits actually found
        int16_t nhbkg;    ///< Number of hits attributed to background noise
        int16_t mskphot;  ///< Bitmask of photons masked out or shared with other tracks

        CRIDHYP() = default;

        int32_t read(const DataBuffer& data, int32_t offset, bool full);
    };
} // namespace jazelle