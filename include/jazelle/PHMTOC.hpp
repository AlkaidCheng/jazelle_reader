/**
 * @file PHMTOC.hpp
 * @brief Defines the 'Table of Contents' (TOC) for a MINIDST record.
 *
 * This struct corresponds to PHMTOC.java and holds the counts of all
 * banks present in the main data blob.
 *
 * @see hep.sld.jazelle.mdst.PHMTOC
 */

#pragma once

#include <cstdint>

namespace jazelle
{
    // Forward-declaration of the stream reader
    class JazelleStream;

    /**
     * @struct PHMTOC
     * @brief Holds the bank counts for a MINIDST data record.
     *
     * This struct is populated directly from the JazelleInputStream
     * just before the main data blob is read.
     */
    struct PHMTOC
    {
        /**
         * @brief Deserialization constructor.
         * Reads all TOC fields from the stream.
         * @param stream The Jazelle input stream.
         */
        explicit PHMTOC(JazelleStream& stream);

        /**
         * @brief Default constructor.
         */
        PHMTOC() = default;

        // Member variables are public for easy access, matching the
        // package-private access in the Java original.
        float   m_version;
        int32_t m_nMcPart;
        int32_t m_nPhPSum;
        int32_t m_nPhChrg;
        int32_t m_nPhKlus;
        int32_t m_nPhKTrk;
        int32_t m_nPhWic;
        int32_t m_nPhWMC;
        int32_t m_nPhCrid;
        int32_t m_nPhPoint;
        int32_t m_nMcPnt;
        int32_t m_nPhKMC1;
        int32_t m_nPhKChrg;
        int32_t m_nPhBm;
        int32_t m_nPhEvCl;
        int32_t m_nMcBeam;
        int32_t m_nPhKElId;
        int32_t m_nPhVxOv;
    };

} // namespace jazelle