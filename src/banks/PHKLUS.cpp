/**
 * @file PHKLUS.cpp
 * @brief Implementation of the PHKLUS bank read method.
 */

#include "jazelle/banks/PHKLUS.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHKLUS::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;
        status = buffer.readInt(o);   o += 4;
        eraw   = buffer.readFloat(o); o += 4;
        cth    = buffer.readFloat(o); o += 4;
        wcth   = buffer.readFloat(o); o += 4;
        phi    = buffer.readFloat(o); o += 4;
        wphi   = buffer.readFloat(o); o += 4;
        
        for (int i = 0; i < 8; ++i) {
            elayer[i] = buffer.readFloat(o); o += 4;
        }
        
        nhit2  = buffer.readInt(o);   o += 4;
        cth2   = buffer.readFloat(o); o += 4;
        wcth2  = buffer.readFloat(o); o += 4;
        phi2   = buffer.readFloat(o); o += 4;
        whphi2 = buffer.readFloat(o); o += 4;
        nhit3  = buffer.readInt(o);   o += 4;
        cth3   = buffer.readFloat(o); o += 4;
        wcth3  = buffer.readFloat(o); o += 4;
        phi3   = buffer.readFloat(o); o += 4;
        wphi3  = buffer.readFloat(o); o += 4;
        
        return 96; // Fixed size
    }

} // namespace jazelle