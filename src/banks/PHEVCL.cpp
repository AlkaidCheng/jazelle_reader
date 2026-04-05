/**
 * @file PHEVCL.cpp
 * @brief Implementation of the PHEVCL bank read method. Store information on global event classification.
 */

#include "jazelle/banks/PHEVCL.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHEVCL::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;

        // 1x 32-bit integer (4 bytes)
        evtclass = buffer.readInt(o); 
        o += 4;

        // 3x 32-bit floats (12 bytes)
        buffer.readFloats(o, thrust.data(), 3); 
        o += 12;

        // 7x 16-bit integers (14 bytes)
        vxdstat  = buffer.readShort(o); o += 2;
        cdcstat  = buffer.readShort(o); o += 2;
        kalstat  = buffer.readShort(o); o += 2;
        crdstat  = buffer.readShort(o); o += 2;
        wicstat  = buffer.readShort(o); o += 2;
        conditi1 = buffer.readShort(o); o += 2;
        conditi2 = buffer.readShort(o); o += 2;

        return 72; 
    }

} // namespace jazelle