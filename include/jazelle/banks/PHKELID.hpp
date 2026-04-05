/**
 * @file PHKELID.hpp
 * @brief Definition of the PHKELID (Calorimeter/Electron ID) bank.
 *
 * Evaluates how well a charged track matches the profile of an electron 
 * by comparing its momentum to the shape and depth of its calorimeter shower.
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>

namespace jazelle
{
    class DataBuffer;  
    class JazelleEvent; 
    struct PHCHRG;      

    struct PHKELID : public Bank
    {
        PHCHRG* phchrg = nullptr; ///< The central drift chamber track evaluated by this bank
        
        int16_t idstat;   ///< Electron ID status/quality flag
        int16_t prob;     ///< Final computed probability that the track is an electron
        
        // --- Kinematic Matching ---
        float   phi;      ///< Azimuthal angle of the track at the calorimeter face
        float   theta;    ///< Polar angle of the track at the calorimeter face
        float   qp;       ///< Track momentum (q/p)
        float   dphi;     ///< Error/Spread in phi match
        float   dtheta;   ///< Error/Spread in theta match
        float   dqp;      ///< Error on momentum
        
        float   tphi;     ///< Azimuthal angle of the associated calorimeter cluster centroid
        float   ttheta;   ///< Polar angle of the associated calorimeter cluster centroid
        float   isolat;   ///< Isolation metric (energy surrounding the electron candidate)
        
        // --- Shower Shape ---
        float   em1;      ///< Energy deposited in the first electromagnetic (EM) layer
        float   em12;     ///< Total energy deposited in EM layers 1 and 2
        float   dem12;    ///< Uncertainty on the EM1+2 energy
        float   had1;     ///< Energy deposited in the first hadronic (HAD) layer (used as a veto for electrons)
        
        float   emphi;    ///< Phi width of the EM shower
        float   emtheta;  ///< Theta width of the EM shower
        float   phiwid;   ///< Overall transverse width in phi
        float   thewid;   ///< Overall transverse width in theta
        
        // --- Tower Clusters ---
        float   em1x1;    ///< Energy in the central 1x1 calorimeter tower block
        float   em2x2a;   ///< Energy in the 2x2 block (configuration A)
        float   em2x2b;   ///< Energy in the 2x2 block (configuration B)
        float   em3x3a;   ///< Energy in the 3x3 block (captures full EM shower core)
        float   em3x3b;   ///< Energy in the extended 3x3 block

        int32_t m_phchrg_id = 0; ///< Raw bank ID used internally to resolve the phchrg pointer

        explicit PHKELID(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };
} // namespace jazelle