/**
 * @file PHPSUM.cpp
 * @brief Implementation of the PHPSUM bank read method.
 */

#include "jazelle/banks/PHPSUM.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHPSUM::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        px     = buffer.readFloat(offset);
        py     = buffer.readFloat(offset + 4);
        pz     = buffer.readFloat(offset + 8);
        x      = buffer.readFloat(offset + 12);
        y      = buffer.readFloat(offset + 16);
        z      = buffer.readFloat(offset + 20);
        charge = buffer.readFloat(offset + 24);
        status = buffer.readInt(offset + 28);
        return 32; // Fixed size
    }

} // namespace jazelle