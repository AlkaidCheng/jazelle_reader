/**
 * @file PHKELID.cpp
 * @brief Implementation of the PHKELID bank read method.
 */

#include "jazelle/banks/PHKELID.hpp"
#include "DataBuffer.hpp" // Internal buffer header
#include "jazelle/JazelleEvent.hpp" // For findPHCHRG

namespace jazelle
{
    int32_t PHKELID::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;
        
        // Read the pointer ID first
        int32_t pid = buffer.readInt(o);
        m_phchrg_id = pid >> 16;
        phchrg = event.get<PHCHRG>().find(m_phchrg_id);
        o += 4;
        
        idstat  = buffer.readShort(o); o += 2;
        prob    = buffer.readShort(o); o += 2;
        phi     = buffer.readFloat(o); o += 4;
        theta   = buffer.readFloat(o); o += 4;
        qp      = buffer.readFloat(o); o += 4;
        dphi    = buffer.readFloat(o); o += 4;
        dtheta  = buffer.readFloat(o); o += 4;
        dqp     = buffer.readFloat(o); o += 4;
        tphi    = buffer.readFloat(o); o += 4;
        ttheta  = buffer.readFloat(o); o += 4;
        isolat  = buffer.readFloat(o); o += 4;
        em1     = buffer.readFloat(o); o += 4;
        em12    = buffer.readFloat(o); o += 4;
        dem12   = buffer.readFloat(o); o += 4;
        had1    = buffer.readFloat(o); o += 4;
        emphi   = buffer.readFloat(o); o += 4;
        emtheta = buffer.readFloat(o); o += 4;
        phiwid  = buffer.readFloat(o); o += 4;
        thewid  = buffer.readFloat(o); o += 4;
        em1x1   = buffer.readFloat(o); o += 4;
        em2x2a  = buffer.readFloat(o); o += 4;
        em2x2b  = buffer.readFloat(o); o += 4;
        em3x3a  = buffer.readFloat(o); o += 4;
        em3x3b  = buffer.readFloat(o); o += 4;
        
        return 96; // Fixed size
    }

} // namespace jazelle