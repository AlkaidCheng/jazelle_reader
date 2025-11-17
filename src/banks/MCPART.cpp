/**
 * @file MCPART.cpp
 * @brief Implementation of the MCPART bank read method.
 */

#include "jazelle/banks/MCPART.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t MCPART::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        // Based on mcpart.template
        p[0]   = buffer.readFloat(offset);
        p[1]   = buffer.readFloat(offset + 4);
        p[2]   = buffer.readFloat(offset + 8);
        e      = buffer.readFloat(offset + 12);
        ptot   = buffer.readFloat(offset + 16);
        ptype  = buffer.readInt(offset + 20);
        charge = buffer.readFloat(offset + 24);
        origin = buffer.readInt(offset + 28);
        xt[0]  = buffer.readFloat(offset + 32);
        xt[1]  = buffer.readFloat(offset + 36);
        xt[2]  = buffer.readFloat(offset + 40);
        parent_id = buffer.readInt(offset + 44);
        
        return 48; // Fixed size
    }

} // namespace jazelle