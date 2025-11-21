/**
 * @file PHWIC.cpp
 * @brief Implementation of the PHWIC bank read method.
 */

#include "jazelle/banks/PHWIC.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHWIC::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;
        idstat   = buffer.readShort(o); o += 2;
        nhit     = buffer.readShort(o); o += 2;
        nhit45   = buffer.readShort(o); o += 2;
        npat     = buffer.readShort(o); o += 2;
        nhitpat  = buffer.readShort(o); o += 2;
        syshit   = buffer.readShort(o); o += 2;
        qpinit   = buffer.readFloat(o); o += 4;
        t1       = buffer.readFloat(o); o += 4;
        t2       = buffer.readFloat(o); o += 4;
        t3       = buffer.readFloat(o); o += 4;
        hitmiss  = buffer.readInt(o);   o += 4;
        itrlen   = buffer.readFloat(o); o += 4;
        nlayexp  = buffer.readShort(o); o += 2;
        nlaybey  = buffer.readShort(o); o += 2;
        missprob = buffer.readFloat(o); o += 4;
        phwicid  = buffer.readInt(o);   o += 4;
        nhitshar = buffer.readShort(o); o += 2;
        nother   = buffer.readShort(o); o += 2;
        hitsused = buffer.readInt(o);   o += 4;
    
        buffer.readFloats(o, pref1.data(), 3);
        o += 12;
        buffer.readFloats(o, pfit.data(), 4);
        o += 16;
        buffer.readFloats(o, dpfit.data(), 10);
        o += 40;        
  
        chi2      = buffer.readFloat(o); o += 4;
        ndf       = buffer.readShort(o); o += 2;
        punfit    = buffer.readShort(o); o += 2;
        matchChi2 = buffer.readFloat(o); o += 4;
        matchNdf  = buffer.readShort(o); o += 2;
        
        // o += 2; // Spare 2 bytes at the end
        
        return 140; // Fixed size
    }

} // namespace jazelle