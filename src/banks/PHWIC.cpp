/**
 * @file PHWIC.cpp
 * @brief Implementation of the PHWIC bank read method. Store for muon detector information.
 * * The PHWIC family stores muon identification information derived from the 
 * Warm Iron Calorimeter (WIC) and its association with Central Drift Chamber (CDC) tracks. [cite: 1423, 2726, 2789]
 */

#include "jazelle/banks/PHWIC.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHWIC::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;

        /* General Status and Hit Information */
        idstat   = buffer.readShort(o); o += 2; // Identification status flag
        nhit     = buffer.readShort(o); o += 2; // Total hits in WIC associated with this track
        nhit45   = buffer.readShort(o); o += 2; // Hits in specific angular layers (e.g., 45-degree)
        npat     = buffer.readShort(o); o += 2; // Number of patterns/clusters found in WIC
        nhitpat  = buffer.readShort(o); o += 2; // Number of hits within the identified WIC pattern
        syshit   = buffer.readShort(o); o += 2; // System-specific hit flag or counter

        /* Kinematics and Energy Tracking */
        qpinit   = buffer.readFloat(o); o += 4; // Initial charge/momentum (q/p) from tracking
        t1       = buffer.readFloat(o); o += 4; // Track matching parameter/direction at calorimeter layers
        t2       = buffer.readFloat(o); o += 4; // Track matching parameter/direction at calorimeter layers
        t3       = buffer.readFloat(o); o += 4; // Track matching parameter/direction at calorimeter layers
        hitmiss  = buffer.readInt(o);   o += 4; // Bitmask of hit/miss patterns across WIC layers
        itrlen   = buffer.readFloat(o); o += 4; // Interaction length/track length through the iron

        /* Layer Penetration and Quality */
        nlayexp  = buffer.readShort(o); o += 2; // Number of WIC layers expected to be hit
        nlaybey  = buffer.readShort(o); o += 2; // Layers hit beyond the expected penetration depth
        missprob = buffer.readFloat(o); o += 4; // Misidentification probability (non-muon probability)
        phwicid  = buffer.readInt(o);   o += 4; // Unique ID linking to the corresponding PHCHRG track
        nhitshar = buffer.readShort(o); o += 2; // Number of WIC hits shared with another track
        nother   = buffer.readShort(o); o += 2; // Counter for other associated clusters/hits
        hitsused = buffer.readInt(o);   o += 4; // Count or bitmask of hits used in the final fit
    
        /* Fitting and Trajectory Data */
        buffer.readFloats(o, pref1.data(), 3);  // Reference points for the WIC cluster
        o += 12;
        buffer.readFloats(o, pfit.data(), 4);   // Trajectory fit parameters within the WIC
        o += 16;
        buffer.readFloats(o, dpfit.data(), 10);  // Error matrix for WIC-specific track fit
        o += 40;        
  
        chi2      = buffer.readFloat(o); o += 4; // Chi-squared of the WIC internal track fit
        ndf       = buffer.readShort(o); o += 2; // Degrees of freedom for internal WIC fit
        punfit    = buffer.readShort(o); o += 2; // Status/counter for points not utilized in fit
        matchChi2 = buffer.readFloat(o); o += 4; // Chi-squared of CDC track-to-WIC cluster match
        matchNdf  = buffer.readShort(o); o += 2; // Degrees of freedom for the track-to-WIC matching
        
        return 140; // Fixed size
    }

} // namespace jazelle