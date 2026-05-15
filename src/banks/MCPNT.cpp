#include "jazelle/banks/MCPNT.hpp"
#include "DataBuffer.hpp"

namespace jazelle
{
    int32_t MCPNT::read(const DataBuffer& buffer, int32_t offset, JazelleEvent&)
    {
        int32_t o = offset;

        // Packed header: (MCPNT_id << 16) | MCPART_id
        const uint32_t hdr = static_cast<uint32_t>(buffer.readInt(o));
        o += 4;
        m_id       = static_cast<int32_t>((hdr >> 16) & 0xffffu);
        mcpart_id  = static_cast<int32_t>( hdr        & 0xffffu);

        reason   = buffer.readInt(o);    o += 4;
        nhits    = buffer.readInt(o);    o += 4;
        econtrib = buffer.readFloat(o);  o += 4;

        // phpoint_id is filled in by the second pass in parseMiniDst.
        phpoint_id = 0;
        phpoint    = nullptr;
        mcpart     = nullptr;
        return o - offset;   // always 16
    }
} // namespace jazelle