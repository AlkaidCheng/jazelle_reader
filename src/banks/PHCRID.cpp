/**
 * @file PHCRID.cpp
 * @brief Implementation of the PHCRID bank read method. Store for CRID information.
 * * The PHCRID family contains particle identification data from the CRID, 
 * utilizing Cherenkov angle reconstruction in liquid and gas radiators.
 */

#include "jazelle/banks/PHCRID.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHCRID::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;
        
        /* Control and Normalization */
        ctlword = buffer.readInt(o);   o += 4; // Control word containing radiator presence flags
        norm    = buffer.readFloat(o); o += 4; // Normalization factor for likelihood calculations
        rc      = buffer.readShort(o); o += 2; // Return code from the CRID reconstruction processor
        geom    = buffer.readShort(o); o += 2; // Geometry version or status flag
        trkp    = buffer.readShort(o); o += 2; // Track parameter pointer or index
        nhits   = buffer.readShort(o); o += 2; // Number of Cherenkov photon hits associated with the track
        
        /* Radiator Presence Flags */
        // Extracted from the control word to determine which radiators provided data
        bool liqPresent = (ctlword & 0x10000) != 0; // Bit flag for Liquid Radiator (C6F14)
        bool gasPresent = (ctlword & 0x20000) != 0; // Bit flag for Gas Radiator (C5F12)
        
        /* Radiator Data Blocks */
        // Read specific likelihood and Cherenkov angle data for each radiator
        int32_t liq_size = liq.read(buffer, o, liqPresent);
        o += liq_size;
        
        int32_t gas_size = gas.read(buffer, o, gasPresent);
        o += gas_size;
        
        /* Combined Particle ID Vector */
        // Create combined log-likelihood vector (llik) for species hypothesis (e, mu, pi, K, p)
        llik = PIDVEC(
            liq.llik.has_value() ? &(*liq.llik) : nullptr,
            gas.llik.has_value() ? &(*gas.llik) : nullptr,
            norm
        );
        
        return (o - offset); // Returns variable size based on present radiator data
    }

} // namespace jazelle