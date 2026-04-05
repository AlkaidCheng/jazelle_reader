/**
 * @file PHKCHRG.cpp
 * @brief Implementation of the PHKCHRG bank read method.
 */

#include "jazelle/banks/PHKCHRG.hpp"
#include "DataBuffer.hpp" 

namespace jazelle
{
    int32_t PHKCHRG::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;

        // 1. Consume the 32-bit header
        int32_t header = buffer.readInt(o);
        phchrg_id = header & 0xffff;
        o += 4;

        // 2. Read the angular matching kinematics
        match_distance = buffer.readFloat(o); o += 4;
        delta_phi      = buffer.readFloat(o); o += 4;
        delta_theta    = buffer.readFloat(o); o += 4;

        return 16;
    }

} // namespace jazelle