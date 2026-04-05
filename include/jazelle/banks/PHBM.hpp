/**
 * @file PHBM.hpp
 * @brief Definition of the PHBM (Beam energy and position) bank.
 *
 * @see hep.sld.jazelle.family.PHBM
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <array>

namespace jazelle
{
    class DataBuffer;   // Forward-declaration
    class JazelleEvent; // Forward-declaration
    
    /**
     * @struct PHBM
     * @brief The Beam energy and position bank.
     * Contains the Z pole center-of-mass energy, interaction point, and crucial
     * electron beam polarization data for calculating Left-Right Asymmetry (A_LR).
     */
    struct PHBM : public Bank
    {
        // --- Member Variables ---
        float                ecm;    // Center of mass energy (~91.2 GeV)
        float                decm;   // Error on CM energy
        std::array<float, 3> pos;    // X,Y,Z of interaction point (from SLC)
        std::array<float, 6> dpos;   // Error matrix of X,Y,Z of interaction point
        //int32_t              status; // Status word describing beam info origin/validity
        float                pol;    // Average polarization measured by the Compton (-1. to 1.)
        float                dpol;   // Error on polarization of e- beam
        //float                bpol;   // Polarization sign for this specific event (1, 0, -1)

        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHBM(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (60).
         * @see hep.sld.jazelle.family.PHBM#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;

    };

} // namespace jazelle