/**
 * @file MCHEAD.cpp
 * @brief Implementation of the MCHEAD bank read method.
 */

#include "jazelle/banks/MCHEAD.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t MCHEAD::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        ntot   = buffer.readInt(offset);
        origin = buffer.readInt(offset + 4);
        ipx    = buffer.readFloat(offset + 8);
        ipy    = buffer.readFloat(offset + 12);
        ipz    = buffer.readFloat(offset + 16);
        return 20; // Fixed size
    }

} // namespace jazelle