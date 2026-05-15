#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <cmath>

namespace jazelle
{
    class DataBuffer;
    class JazelleEvent;

    struct MCPART : public Bank
    {
        // --- Origin Bitmask Definitions ---
        struct Origin {
            static constexpr int32_t DECAYED  = (1 << 0);
            static constexpr int32_t DECAYFLT = (1 << 1);
            static constexpr int32_t BEAMPIPE = (1 << 2);
            static constexpr int32_t NOINTER  = (1 << 3);
            static constexpr int32_t STOPPED  = (1 << 4);
            static constexpr int32_t INTERACT = (1 << 5);
            static constexpr int32_t INTSHDEP = (1 << 6);
            static constexpr int32_t PRIMARY  = (1 << 8);
            static constexpr int32_t ISTOP1   = (1 << 9);
            static constexpr int32_t ISTOP2   = (1 << 10);
            static constexpr int32_t KALTOCDC = (1 << 11);
            static constexpr int32_t SWERROR  = (1 << 12);
            static constexpr int32_t SW2MNYST = (1 << 13);
            static constexpr int32_t SWOUTOFT = (1 << 14);
            static constexpr int32_t EMAXTERM = (1 << 15);
            static constexpr int32_t NOTTRACK = (1 << 16);
            static constexpr int32_t ISR      = (1 << 17);
            static constexpr int32_t BEAM     = (1 << 18);
            static constexpr int32_t PREFRAG  = (1 << 19);
            static constexpr int32_t SWUM     = (1 << 20);
        };

        float   px;              ///< True X momentum at track origin (GeV)
        float   py;              ///< True Y momentum at track origin (GeV)
        float   pz;              ///< True Z momentum at track origin (GeV)
        float   e;               ///< True total energy of the particle (GeV)
        int32_t ptype;           ///< LUND/PDG Particle Identification Code
        float   charge;          ///< Particle electric charge
        int32_t origin;          ///< Simulation history bitmask (MCPART::Origin)

        float   xt_x;            ///< True X coordinate of termination/decay (cm)
        float   xt_y;            ///< True Y coordinate of termination/decay (cm)
        float   xt_z;            ///< True Z coordinate of termination/decay (cm)

        int32_t parent_id;       ///< Bank ID of the parent particle
        MCPART* parent = nullptr;///< Resolved pointer to parent MCPART

        explicit MCPART(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;

        /// True scalar momentum magnitude (GeV). Computed on demand from (px, py, pz).
        float ptot() const { return std::sqrt(px*px + py*py + pz*pz); }
    };
} // namespace jazelle