/**
 * @file PHKLUS.cpp
 * @brief Implementation of the PHKLUS bank read method. Store for calorimetry information.
 * * The PHKLUS family contains findings of the Calorimeter, specifically reconstructed 
 * clusters of energy deposition in the Liquid Argon Calorimeter (LAC). [cite: 1877, 1878]
 */

#include "jazelle/banks/PHKLUS.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHKLUS::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;

        /* General Cluster Parameters */
        status = buffer.readInt(o);   o += 4; // Cluster status flags
        eraw   = buffer.readFloat(o); o += 4; // Raw cluster energy before e/pi correction
        cth    = buffer.readFloat(o); o += 4; // Mean cluster cos(theta), energy weighted
        wcth   = buffer.readFloat(o); o += 4; // Width of cluster energy in cos(theta)
        phi    = buffer.readFloat(o); o += 4; // Mean cluster phi, energy weighted
        wphi   = buffer.readFloat(o); o += 4; // Width of cluster energy in phi

        /* Layer-by-Layer Energy Distribution (MIP scale) */
        // Indices 0-7 correspond to: Lum Em1, Lum Em2, LAC Em1, LAC Em2, 
        // LAC Had1, LAC Had2, WIC Pads 1, WIC Pads 2.
        buffer.readFloats(o, elayer.data(), 8);
        o += 32;
        
        /* Layer 2 (LAC Electromagnetic Section 1) Details */
        nhit2  = buffer.readInt(o);   o += 4; // Number of towers hit in LAC EM1
        cth2   = buffer.readFloat(o); o += 4; // Mean cos(theta) in LAC EM1, energy weighted
        wcth2  = buffer.readFloat(o); o += 4; // Width of energy in cos(theta) for LAC EM1
        phi2   = buffer.readFloat(o); o += 4; // Mean phi in LAC EM1, energy weighted
        whphi2 = buffer.readFloat(o); o += 4; // Width of energy in phi for LAC EM1

        /* Layer 3 (LAC Electromagnetic Section 2) Details */
        nhit3  = buffer.readInt(o);   o += 4; // Number of towers hit in LAC EM2
        cth3   = buffer.readFloat(o); o += 4; // Mean cos(theta) in LAC EM2, energy weighted
        wcth3  = buffer.readFloat(o); o += 4; // Width of energy in cos(theta) for LAC EM2
        phi3   = buffer.readFloat(o); o += 4; // Mean phi in LAC EM2, energy weighted
        wphi3  = buffer.readFloat(o); o += 4; // Width of energy in phi for LAC EM2
        
        return 96; // Fixed size
    }

} // namespace jazelle