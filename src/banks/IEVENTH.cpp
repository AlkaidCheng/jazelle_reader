/**
 * @file IEVENTH.cpp
 * @brief Implementation of the IEVENTH bank's special read method.
 */

#include "jazelle/banks/IEVENTH.hpp"
#include "JazelleStream.hpp" // Internal stream header

namespace jazelle
{
    void IEVENTH::read(JazelleStream& stream)
    {
        header  = stream.readInt();
        run     = stream.readInt();
        event   = stream.readInt();
        evttime = stream.readDate();
        weight  = stream.readFloat();
        evttype = stream.readInt();
        trigger = stream.readInt();
    }

} // namespace jazelle