/**
 * @file PHCRID.hpp
 * @brief Definition of the PHCRID (Cherenkov Ring Imaging Detector) bank.
 *
 * Links a charged track to its Cherenkov rings, providing particle ID 
 * likelihoods for both the Liquid (pi/K separation at low p) and 
 * Gas (pi/K/p separation at high p) radiators.
 */

#pragma once

#include "../Bank.hpp"
#include "CRIDHYP.hpp" 
#include <cstdint>

namespace jazelle
{
    class DataBuffer;
    class JazelleEvent;

    struct PHCRID : public Bank
    {
        int32_t ctlword; ///< Control word defining which sub-components (Liq/Gas) were successfully read
        float   norm;    ///< Normalization factor for the likelihood calculations
        int16_t rc;      ///< Global return code/status of the CRID reconstruction for this track
        int16_t geom;    ///< Geometry region flag (e.g., Barrel vs. Endcap)
        int16_t trkp;    ///< Track momentum bin/flag used during ring resolution
        int16_t nhits;   ///< Total CRID hits (Liquid + Gas) associated with the track
        
        CRIDHYP liq;     ///< Hypothesis data for the Liquid radiator (low momentum PID)
        CRIDHYP gas;     ///< Hypothesis data for the Gas radiator (high momentum PID)
        PIDVEC  llik;    ///< Final combined Log-Likelihoods (e, mu, pi, k, p) from both radiators

        explicit PHCRID(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };
} // namespace jazelle