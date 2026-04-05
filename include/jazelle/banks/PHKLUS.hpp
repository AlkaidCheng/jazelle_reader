/**
 * @file PHKLUS.hpp
 * @brief Definition of the PHKLUS (Calorimeter Cluster) bank.
 *
 * Represents an isolated grouping of energy deposited in the Liquid Argon 
 * Calorimeter (LAC)
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <array>

namespace jazelle
{
    class DataBuffer; 
    class JazelleEvent;

    struct PHKLUS : public Bank
    {
        int32_t status;  ///< Cluster quality and region status (e.g., barrel vs endcap)
        float   eraw;    ///< Raw, uncalibrated energy sum of the cluster
        
        // --- Global Centroid ---
        float   cth;     ///< Cosine(theta) of the geometric cluster centroid
        float   wcth;    ///< Energy-weighted cosine(theta) of the centroid
        float   phi;     ///< Geometric azimuthal angle
        float   wphi;    ///< Energy-weighted azimuthal angle
        
        // --- Longitudinal Profile ---
        std::array<float, 8> elayer; ///< Energy deposited in each specific calorimeter depth layer (EM1, EM2, HAD1, etc.)
        
        // --- Sub-Cluster (EM section) ---
        int32_t nhit2;   ///< Number of hits in the electromagnetic sections
        float   cth2;    ///< Geometric cosine(theta) of the EM sub-cluster
        float   wcth2;   ///< Energy-weighted cosine(theta) of the EM sub-cluster
        float   phi2;    ///< Geometric phi of the EM sub-cluster
        float   whphi2;  ///< Energy-weighted phi of the EM sub-cluster
        
        // --- Sub-Cluster (Hadronic section) ---
        int32_t nhit3;   ///< Number of hits in the hadronic sections
        float   cth3;    ///< Geometric cosine(theta) of the HAD sub-cluster
        float   wcth3;   ///< Energy-weighted cosine(theta) of the HAD sub-cluster
        float   phi3;    ///< Geometric phi of the HAD sub-cluster
        float   wphi3;   ///< Energy-weighted phi of the HAD sub-cluster
        
        explicit PHKLUS(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };
} // namespace jazelle