/**
 * @file PHWIC.hpp
 * @brief Definition of the PHWIC (Warm Iron Calorimeter) bank.
 *
 * The WIC acts as the primary muon identifier at SLD. This bank stores 
 * tracking information for particles that penetrate through the calorimeter 
 * and into the instrumented iron flux return.
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <array>

namespace jazelle
{
    class DataBuffer;
    class JazelleEvent;

    struct PHWIC : public Bank
    {
        int16_t idstat;   ///< Muon identification status and quality flag
        int16_t nhit;     ///< Total WIC layers hit by the track
        int16_t nhit45;   ///< Hits specifically in 45-degree stereo strips
        int16_t npat;     ///< Number of distinct patterns/sub-clusters found
        int16_t nhitpat;  ///< Number of hits within the chosen primary pattern
        int16_t syshit;   ///< System bitmask (barrel, endcap, specific octants)
        
        // --- Extrapolation ---
        float   qpinit;   ///< Initial momentum (q/p) expected as it enters the WIC
        float   t1;       ///< Trajectory parameter 1
        float   t2;       ///< Trajectory parameter 2
        float   t3;       ///< Trajectory parameter 3
        int32_t hitmiss;  ///< Bitmask mapping which specific WIC layers had expected hits vs actual misses
        float   itrlen;   ///< Total interaction length (amount of iron) penetrated by the track
        
        int16_t nlayexp;  ///< Number of WIC layers expected to be hit based on momentum
        int16_t nlaybey;  ///< Number of layers hit beyond the expected penetration depth (punch-through)
        float   missprob; ///< Probability that this track is a hadron punch-through misidentified as a muon
        
        int32_t phwicid;  ///< WIC cluster ID
        int16_t nhitshar; ///< Number of WIC hits shared with another adjacent track
        int16_t nother;   ///< Number of hits in the vicinity not assigned to the track
        int32_t hitsused; ///< Mask of hits used in the internal WIC track fit
        
        // --- Fit Results ---
        std::array<float, 3>  pref1; ///< Reference point 1 (X, Y, Z) for the internal WIC track
        std::array<float, 4>  pfit;  ///< WIC-only track fit parameters
        std::array<float, 10> dpfit; ///< Error matrix for the WIC track fit
        
        float   chi2;      ///< Chi-squared of the internal WIC track fit
        int16_t ndf;       ///< Degrees of freedom for internal WIC fit
        int16_t punfit;    ///< Points deemed unusable and discarded from the fit
        float   matchChi2; ///< Chi-squared of the geometric match between the CDC track and the WIC track
        int16_t matchNdf;  ///< Degrees of freedom for the matching process
        
        explicit PHWIC(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };
} // namespace jazelle