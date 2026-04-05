/**
 * @file PHPOINT.hpp
 * @brief Definition of the PHPOINT (Pointers to Particle Information) bank.
 *
 * @see hep.sld.jazelle.family.PHPOINT
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>

namespace jazelle
{
    class DataBuffer;   // Forward-declaration
    class JazelleEvent; // Forward-declaration

    // Forward-declare the target banks so we can store typed pointers
    struct PHPSUM;
    struct PHCHRG;
    struct PHKLUS;
    struct PHKELID;
    struct PHWIC;
    struct PHCRID;

    /**
     * @struct PHPOINT
     * @brief Pointers to Particle Information bank.
     * Links the central physics summary to its subsystem components.
     */
    struct PHPOINT : public Bank
    {
        
        // --- Member Variables (Raw Bank IDs) ---
        int32_t phpsum_id;  // Pointer to PHPSUM bank
        int32_t phchrg_id;  // Pointer to tracking information
        int32_t phklus_id;  // Pointer to calorimetry information
        int32_t phkelid_id; // Pointer to calorimetry hypothesis
        int32_t phwic_id;   // Pointer to muon detector information
        int32_t phcrid_id;  // Pointer to CRID information

        // --- Resolved Pointers (Populated later during event building) ---
        PHPSUM* phpsum  = nullptr;
        PHCHRG* phchrg  = nullptr;
        PHKLUS* phklus  = nullptr;
        PHKELID* phkelid = nullptr;
        PHWIC* phwic   = nullptr;
        PHCRID* phcrid  = nullptr;
        

        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHPOINT(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read.
         * @see hep.sld.jazelle.family.PHPOINT#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };

} // namespace jazelle