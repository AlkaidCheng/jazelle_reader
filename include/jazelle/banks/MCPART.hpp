/**
 * @file MCPART.hpp
 * @brief Definition of the MCPART (Monte Carlo Particle) bank.
 *
 * Contains the exact truth-level kinematics, particle IDs, and history 
 * for a single generated particle before detector simulation.
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <array>

namespace jazelle
{
    class DataBuffer;
    class JazelleEvent;

    /**
     * @struct MCPART
     * @brief Holds the Monte Carlo truth information for a single particle.
     */
    struct MCPART : public Bank
    {
        // --- Origin Bitmask Definitions ---
        struct Origin {
            static constexpr int32_t DECAYED  = (1 << 0);  // Decayed by the event generator
            static constexpr int32_t DECAYFLT = (1 << 1);  // Decayed in flight during detector tracking (swimmer)
            static constexpr int32_t BEAMPIPE = (1 << 2);  // Lost down the beam pipe (undetected)
            static constexpr int32_t NOINTER  = (1 << 3);  // Traversed the detector without interacting
            static constexpr int32_t STOPPED  = (1 << 4);  // Tracked until energy fell below simulation cutoff
            static constexpr int32_t INTERACT = (1 << 5);  // Interacted with detector material
            static constexpr int32_t INTSHDEP = (1 << 6);  // Interacted, initiating a calorimeter shower
            static constexpr int32_t PRIMARY  = (1 << 8);  // Particle originated at the primary e+e- vertex
            static constexpr int32_t ISTOP1   = (1 << 9);  // GEANT tracking flag: Stopped normally
            static constexpr int32_t ISTOP2   = (1 << 10); // GEANT tracking flag: Special termination
            static constexpr int32_t KALTOCDC = (1 << 11); // Shower albedo (backscatter) from Calorimeter into Tracking
            static constexpr int32_t SWERROR  = (1 << 12); // Tracking error occurred during simulation
            static constexpr int32_t SW2MNYST = (1 << 13); // Tracking aborted (too many GEANT steps)
            static constexpr int32_t SWOUTOFT = (1 << 14); // Tracking aborted (outside sensitive time window)
            static constexpr int32_t EMAXTERM = (1 << 15); // EM shower axis terminated (sufficient depth reached)
            static constexpr int32_t NOTTRACK = (1 << 16); // Neutrino or other invisible particle (untracked)
            static constexpr int32_t ISR      = (1 << 17); // Initial State Radiation (photon)
            static constexpr int32_t BEAM     = (1 << 18); // Initial state beam particle (e+ or e-)
            static constexpr int32_t PREFRAG  = (1 << 19); // Pre-fragmentation parton (quark/gluon)
            static constexpr int32_t SWUM     = (1 << 20); // Particle was produced dynamically by the GEANT swimmer
        };

        std::array<float, 3> p;  ///< True (X,Y,Z) momentum vector at track origin (GeV)
        float   e;               ///< True total energy of the particle (GeV)
        float   ptot;            ///< True scalar momentum magnitude (GeV)
        int32_t ptype;           ///< LUND/PDG Particle Identification Code
        float   charge;          ///< Particle electric charge
        int32_t origin;          ///< Simulation history bitmask (maps to MCPART::Origin)
        
        std::array<float, 3> xt; ///< True (X,Y,Z) spatial coordinate of particle termination/decay
        
        int32_t parent_id;       ///< Bank ID of the parent particle that produced this one
        MCPART* parent = nullptr;///< Resolved pointer to parent MCPART

        explicit MCPART(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };
} // namespace jazelle