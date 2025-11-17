/**
 * @file CRIDHYP.cpp
 * @brief Implementation of the CRIDHYP helper struct.
 */

#include "jazelle/banks/CRIDHYP.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t CRIDHYP::read(const DataBuffer& data, int32_t offset, bool full)
    {
        m_full = full;
        if (m_full)
        {
            // Full 36-byte version
            llik.emplace(data, offset); // Read PIDVEC at offset 0 (20 bytes)
            rc      = data.readShort(offset + 20);
            nhits   = data.readShort(offset + 22);
            besthyp = data.readInt(offset + 24);
            nhexp   = data.readShort(offset + 28);
            nhfnd   = data.readShort(offset + 30);
            nhbkg   = data.readShort(offset + 32);
            mskphot = data.readShort(offset + 34);
            return 36;
        }
        else
        {
            // Reduced 4-byte version
            llik.reset(); // No PIDVEC
            rc      = data.readShort(offset);
            nhits   = data.readShort(offset + 2);
            besthyp = 0;
            nhexp   = 0;
            nhfnd   = 0;
            nhbkg   = 0;
            mskphot = 0;
            return 4;
        }
    }

} // namespace jazelle