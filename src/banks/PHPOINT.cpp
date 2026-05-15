/**
 * @file PHPOINT.cpp
 * @brief Implementation of the PHPOINT bank read method.
 */

#include "jazelle/banks/PHPOINT.hpp"
#include "DataBuffer.hpp"

namespace jazelle
{
    int32_t PHPOINT::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;

        // 1. Consume the 32-bit header
        int32_t header = buffer.readInt(o);
        int32_t id = header & 0xffff;
        int32_t mask = (header >> 16) & 0xffff;
        o += 4;

        // Reset pointers to 0 before we map them
        phpsum_id = 0; phcrid_id = 0; phklus_id = 0; 
        phkelid_id = 0; phchrg_id = 0; phwic_id = 0;

        // 2. Implicit PHPSUM logic
        // As per the memo, if bit 0 is 0, the PHPSUM ID exactly matches this bank's ID.
        if ((mask & (1 << 0)) == 0) {
            phpsum_id = id;
        }

        // 3. Dynamic Bit-Counting Extraction
        // Loop through all 16 possible bits. If a bit is set, read the 16-bit pointer.
        for (int bit = 0; bit < 16; ++bit) 
        {
            if (mask & (1 << bit)) 
            {
                int16_t ptr = buffer.readShort(o);
                o += 2;

                // Map the known bits based on the DUCS template
                if      (bit == 0) phpsum_id = ptr; // Explicitly stored
                else if (bit == 1) phcrid_id = ptr;
                else if (bit == 2) phchrg_id = ptr;
                else if (bit == 3) phklus_id = ptr;
                else if (bit == 4) phkelid_id = ptr;
                else if (bit == 5) phwic_id = ptr;
                // If SLD used bits 6-15 for tracking other subsystems (like PHKTRK),
                // they are safely consumed here and the offset pointer advances properly.
            }
        }

        // 4. Force 32-bit (4-byte) alignment padding
        int32_t bytes_read = o - offset;
        if (bytes_read % 4 != 0) 
        {
            bytes_read += (4 - (bytes_read % 4)); 
        }

        return bytes_read; 
    }

} // namespace jazelle