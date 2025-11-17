/**
 * @file PHMTOC.cpp
 * @brief Implementation of the PHMTOC struct.
 */

#include "jazelle/PHMTOC.hpp"
#include "JazelleStream.hpp" // Internal stream header

namespace jazelle
{
    // This constructor is a direct translation of PHMTOC.java
    PHMTOC::PHMTOC(JazelleStream& stream)
    {
        m_version = stream.readFloat();
        m_nMcPart = stream.readInt();
        m_nPhPSum = stream.readInt();
        m_nPhChrg = stream.readInt();
        m_nPhKlus = stream.readInt();
        m_nPhKTrk = stream.readInt();
        m_nPhWic = stream.readInt();
        m_nPhWMC = stream.readInt();
        m_nPhCrid = stream.readInt();
        m_nPhPoint = stream.readInt();
        m_nMcPnt = stream.readInt();
        m_nPhKMC1 = stream.readInt();
        m_nPhKChrg = stream.readInt();
        m_nPhBm = stream.readInt();
        m_nPhEvCl = stream.readInt();
        m_nMcBeam = stream.readInt();
        m_nPhKElId = stream.readInt();
        m_nPhVxOv = stream.readInt();
    }

} // namespace jazelle