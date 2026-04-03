/**
 * @file PHCHRG.cpp
 * @brief Implementation of the PHCHRG bank read method. Store for tracking information.
 * * The PHCHRG family stores all tracking information reconstructed from the Drift Chamber 
 * and Vertex Detector data. There is one PHCHRG bank per track.
 */

#include "jazelle/banks/PHCHRG.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHCHRG::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;

        /* Helix Fit Parameters  */
        // hlxpar(1) = phi, (2) = 1/pt, (3) = tan(lambda)
        // hlxpar(4) = x, (5) = y, (6) = z
        // theta = pi/2 - lambda
        buffer.readFloats(o, hlxpar.data(), 6);
        o += 24;

        /* Helix Parameter Error Matrix (15 elements)  */
        buffer.readFloats(o, dhlxpar.data(), 15);
        o += 60;

        /* Impact Parameters and Normalization */
        bnorm   = buffer.readFloat(o); o += 4; // Normalization factor
        impact  = buffer.readFloat(o); o += 4; // 2D impact parameter 
        b3norm  = buffer.readFloat(o); o += 4; // 3D normalization
        impact3 = buffer.readFloat(o); o += 4; // 3D impact parameter
        charge  = buffer.readShort(o); o += 2; // Particle charge 
        smwstat = buffer.readShort(o); o += 2; // Swim status
        status  = buffer.readInt(o);   o += 4; // Track status flags
        tkpar0  = buffer.readFloat(o); o += 4; // Initial track parameter

        /* General Track Parameters and Error Matrix */
        buffer.readFloats(o, tkpar.data(), 5);
        o += 20;
        buffer.readFloats(o, dtkpar.data(), 15);
        o += 60;

        /* Track Quality and Hit Statistics */
        length  = buffer.readFloat(o); o += 4; // Track length
        chi2dt  = buffer.readFloat(o); o += 4; // Chi-squared of Drift Chamber fit
        imc     = buffer.readShort(o); o += 2; // Monte Carlo particle association index
        ndfdt   = buffer.readShort(o); o += 2; // Degrees of freedom for Drift Chamber fit
        nhit    = buffer.readShort(o); o += 2; // Total number of hits on track 
        nhite   = buffer.readShort(o); o += 2; // Number of expected hits
        nhitp   = buffer.readShort(o); o += 2; // Number of potential hits
        nmisht  = buffer.readShort(o); o += 2; // Number of missing hits
        nwrght  = buffer.readShort(o); o += 2; // Number of wrong-way hits
        nhitv   = buffer.readShort(o); o += 2; // Number of hits in the Vertex Detector (VXD)
        chi2    = buffer.readFloat(o); o += 4; // Combined fit Chi-squared
        chi2v   = buffer.readFloat(o); o += 4; // Vertex Detector fit Chi-squared
        vxdhit  = buffer.readInt(o);   o += 4; // Bitmask of hit VXD layers
        mustat  = buffer.readShort(o); o += 2; // Muon identification status 
        estat   = buffer.readShort(o); o += 2; // Electron identification status 
        dedx    = buffer.readInt(o);   o += 4; // dE/dx ionization energy loss 

        return 236; // Fixed size
    }

} // namespace jazelle