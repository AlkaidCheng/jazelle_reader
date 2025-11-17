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
        
        for (int i = 0; i < 6; ++i) {
            hlxpar[i]  = buffer.readFloat(o); o += 4;
        }
        for (int i = 0; i < 15; ++i) {
            dhlxpar[i] = buffer.readFloat(o); o += 4;
        }
        
        bnorm   = buffer.readFloat(o); o += 4;
        impact  = buffer.readFloat(o); o += 4;
        b3norm  = buffer.readFloat(o); o += 4;
        impact3 = buffer.readFloat(o); o += 4;
        charge  = buffer.readShort(o); o += 2;
        smwstat = buffer.readShort(o); o += 2;
        status  = buffer.readInt(o);   o += 4;
        tkpar0  = buffer.readFloat(o); o += 4;
        
        for (int i = 0; i < 5; ++i)  {
            tkpar[i]  = buffer.readFloat(o); o += 4;
        }
        for (int i = 0; i < 15; ++i) {
            dtkpar[i] = buffer.readFloat(o); o += 4;
        }
        
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