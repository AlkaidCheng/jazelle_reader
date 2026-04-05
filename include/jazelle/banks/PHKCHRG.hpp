/**
 * @file PHKCHRG.hpp
 * @brief Definition of the PHKCHRG (Relational table PHKLUS to PHCHRG) bank.
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>

namespace jazelle
{
    class DataBuffer;   // Forward-declaration
    class JazelleEvent; // Forward-declaration
    
    struct PHCHRG; // Target declarations
    struct PHKLUS;

    struct PHKCHRG : public Bank
    {
        // --- Raw IDs ---
        int32_t phchrg_id;
        int32_t phklus_id;

        // --- Matching Kinematics ---
        float match_distance; // Overall spatial match distance or probability ?
        float delta_phi;      // Azimuthal angular residual ?
        float delta_theta;    // Polar angular residual ?

        // --- Resolved Pointers ---
        PHCHRG* phchrg = nullptr;
        PHKLUS* phklus = nullptr;

        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHKCHRG(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (16).
         * @see hep.sld.jazelle.family.PHKCHRG#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };
} // namespace jazelle