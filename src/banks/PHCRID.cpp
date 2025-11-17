/**
 * @file PHCRID.cpp
 * @brief Implementation of the PHCRID bank read method.
 */

#include "jazelle/banks/PHCRID.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHCRID::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;
        
        ctlword = buffer.readInt(o);   o += 4;
        norm    = buffer.readFloat(o); o += 4;
        rc      = buffer.readShort(o); o += 2;
        geom    = buffer.readShort(o); o += 2;
        trkp    = buffer.readShort(o); o += 2;
        nhits   = buffer.readShort(o); o += 2;
        
        // ctlword flags
        bool liqPresent = (ctlword & 0x10000) != 0;
        bool gasPresent = (ctlword & 0x20000) != 0;
        
        int32_t liq_size = liq.read(buffer, o, liqPresent);
        o += liq_size;
        
        int32_t gas_size = gas.read(buffer, o, gasPresent);
        o += gas_size;
        
        // Create combined likelihood vector
        llik = PIDVEC(
            liq.llik.has_value() ? &(*liq.llik) : nullptr,
            gas.llik.has_value() ? &(*gas.llik) : nullptr,
            norm
        );
        
        return (o - offset); // Variable size
    }

} // namespace jazelle