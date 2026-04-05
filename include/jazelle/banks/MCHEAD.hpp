/**
 * @file MCHEAD.hpp
 * @brief Definition of the MCHEAD (Monte Carlo Header) bank.
 *
 * Defines the initial state of the simulated event, including the primary 
 * physics process (e.g., Z -> bb) and the precise interaction vertex.
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>

namespace jazelle
{
    class DataBuffer;
    class JazelleEvent;

    /**
     * @struct MCHEAD
     * @brief The Monte Carlo Header bank.
     */
    struct MCHEAD : public Bank
    {
        // --- Origin Bitmask Definitions ---
        struct Origin {
            static constexpr int32_t UUBAR   = 0;  // Z -> u ubar
            static constexpr int32_t DDBAR   = 1;  // Z -> d dbar
            static constexpr int32_t SSBAR   = 2;  // Z -> s sbar
            static constexpr int32_t CCBAR   = 3;  // Z -> c cbar
            static constexpr int32_t BBBAR   = 4;  // Z -> b bbar
            static constexpr int32_t TTBAR   = 5;  // Z -> t tbar (Kinematically impossible at SLD, but defined)
            static constexpr int32_t EE      = 6;  // Z -> e+ e- (Bhabha)
            static constexpr int32_t MUMU    = 7;  // Z -> mu+ mu-
            static constexpr int32_t TAUTAU  = 8;  // Z -> tau+ tau-
            static constexpr int32_t TWOPHOT = 12; // Two-photon interaction (gamma gamma -> X)
        };

        int32_t ntot;   ///< Total number of final state particles generated in the event
        int32_t origin; ///< Origin physics process (maps to MCHEAD::Origin flags)
        float   ipx;    ///< Primary vertex X position (Interaction Point)
        float   ipy;    ///< Primary vertex Y position (Interaction Point)
        float   ipz;    ///< Primary vertex Z position (Interaction Point)

        explicit MCHEAD(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };
} // namespace jazelle