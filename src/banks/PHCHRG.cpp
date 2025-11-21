/**
 * @file PHCHRG.cpp
 * @brief Implementation of the PHCHRG bank read method.
 */

#include "jazelle/banks/PHCHRG.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHCHRG::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;

        buffer.readFloats(o, hlxpar.data(), 6);
        o += 24;
        buffer.readFloats(o, dhlxpar.data(), 15);
        o += 60;

        bnorm   = buffer.readFloat(o); o += 4;
        impact  = buffer.readFloat(o); o += 4;
        b3norm  = buffer.readFloat(o); o += 4;
        impact3 = buffer.readFloat(o); o += 4;
        charge  = buffer.readShort(o); o += 2;
        smwstat = buffer.readShort(o); o += 2;
        status  = buffer.readInt(o);   o += 4;
        tkpar0  = buffer.readFloat(o); o += 4;

        buffer.readFloats(o, tkpar.data(), 5);
        o += 20;
        buffer.readFloats(o, dtkpar.data(), 15);
        o += 60;
        
        length  = buffer.readFloat(o); o += 4;
        chi2dt  = buffer.readFloat(o); o += 4;
        imc     = buffer.readShort(o); o += 2;
        ndfdt   = buffer.readShort(o); o += 2;
        nhit    = buffer.readShort(o); o += 2;
        nhite   = buffer.readShort(o); o += 2;
        nhitp   = buffer.readShort(o); o += 2;
        nmisht  = buffer.readShort(o); o += 2;
        nwrght  = buffer.readShort(o); o += 2;
        nhitv   = buffer.readShort(o); o += 2;
        chi2    = buffer.readFloat(o); o += 4;
        chi2v   = buffer.readFloat(o); o += 4;
        vxdhit  = buffer.readInt(o);   o += 4;
        mustat  = buffer.readShort(o); o += 2;
        estat   = buffer.readShort(o); o += 2;
        dedx    = buffer.readInt(o);   o += 4;
        
        return 236; // Fixed size
    }

} // namespace jazelle