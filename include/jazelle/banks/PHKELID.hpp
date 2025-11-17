/**
 * @file PHKELID.hpp
 * @brief Definition of the PHKELID (Calorimeter/Electron ID) bank.
 *
 * This bank has a pointer to a PHCHRG bank.
 *
 * @see hep.sld.jazelle.family.PHKELID
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>

namespace jazelle
{
    class DataBuffer;   // Forward-declaration
    class JazelleEvent; // Forward-declaration
    struct PHCHRG;      // Forward-declaration of the linked bank type

    /**
     * @struct PHKELID
     * @brief The Calorimeter/Electron ID bank.
     */
    struct PHKELID : public Bank
    {
        // --- Member Variables ---
        
        /// Pointer to the associated PHCHRG bank.
        /// This is resolved during the read() call.
        PHCHRG* phchrg = nullptr;
        
        int16_t idstat;
        int16_t prob;
        float   phi;
        float   theta;
        float   qp;
        float   dphi;
        float   dtheta;
        float   dqp;
        float   tphi;
        float   ttheta;
        float   isolat;
        float   em1;
        float   em12;
        float   dem12;
        float   had1;
        float   emphi;
        float   emtheta;
        float   phiwid;
        float   thewid;
        float   em1x1;
        float   em2x2a;
        float   em2x2b;
        float   em3x3a;
        float   em3x3b;

        // Internal: We store the ID of the linked PHCHRG for resolution.
        int32_t m_phchrg_id = 0;

        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHKELID(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         *
         * This method also resolves the pointer to the PHCHRG bank
         * by calling event.findPHCHRG().
         *
         * @return The number of bytes read (96).
         * @see hep.sld.jazelle.family.PHKELID#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };

} // namespace jazelle