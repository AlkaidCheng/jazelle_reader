/**
 * @file PHBM.cpp
 * @brief Implementation of the PHBM bank read method. Store for beam energy and position.
 * * Contains the Z pole center-of-mass energy, interaction point, and crucial
 * electron beam polarization data.
 */

#include "jazelle/banks/PHBM.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHBM::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;

        /* Center of Mass Energy */
        ecm     = buffer.readFloat(o); o += 4; // Center of mass energy (~91.2 GeV)
        decm    = buffer.readFloat(o); o += 4; // Error on CM energy

        /* Interaction Point (Vertex) */
        buffer.readFloats(o, pos.data(), 3);   // X, Y, Z of interaction point
        o += 12;
        buffer.readFloats(o, dpos.data(), 6);  // Error matrix of X, Y, Z (symmetric 3x3)
        o += 24;

        /* Polarization (8 bytes) */
        pol  = buffer.readFloat(o); o += 4; 
        dpol = buffer.readFloat(o); o += 4;
        
        
        return 60; // Fixed size
    }

} // namespace jazelle