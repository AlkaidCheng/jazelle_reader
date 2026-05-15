/**
 * @file MCPART.cpp
 * @brief Implementation of the MCPART bank read method.
 *
 * Binary layout (36 or 40 bytes per row):
 *
 *   +0..+11    float[3]  P(3)        momentum (px, py, pz)
 *   +12        int32     ID-word     packed:
 *                bits 0-12  : bank_id     (1..5000)
 *                bits 13-25 : parent_id   (0 = no parent, else 1..5000)
 *                bits 26-29 : charge_code (sign bit | 3-bit magnitude idx)
 *                bit 30     : E-omitted   (0 = E follows, 1 = derive E)
 *                bit 31     : spare
 *   +16        int32     PTYPE       particle type id
 *   +20        int32     ORIGIN      template bitmask
 *   +24..+35   float[3]  XT(3)       termination point (cm)
 *   +36        float     E           energy (GeV) — only if !E-omitted
 *
 * Banks are written in PARENT-key traversal order, so parent_id may
 * reference a later or earlier bank. Parent resolution is done by
 * parseMiniDst once the whole family has been read.
 */

#include "jazelle/banks/MCPART.hpp"
#include "DataBuffer.hpp"

#include <cmath>
#include <cstdlib>

namespace jazelle
{
    namespace
    {
        /**
         * @brief Decode the 4-bit charge code to a float charge value.
         *
         * Verified against Z->uū (codes 1/9) and Z->dd̄ (codes 2/10):
         *   bit 3 of code = sign  (1 = positive, 0 = negative)
         *   bits 0-2      = magnitude index into {0, 2/3, 1/3, 1, 2, ...}
         */
        float chargeFromCode(uint32_t code)
        {
            static constexpr float kMagnitude[8] = {
                0.0f,
                2.0f / 3.0f,   // 1, 9
                1.0f / 3.0f,   // 2, 10
                1.0f,          // 3, 11
                2.0f,          // 4, 12
                0.0f, 0.0f, 0.0f
            };
            const uint32_t mag_idx  = code & 0x7u;
            const bool     positive = ((code >> 3) & 0x1u) != 0u;
            const float    m        = kMagnitude[mag_idx];
            return positive ? m : -m;
        }

        /**
         * @brief PDG '92 mass (GeV) for the "ordinary" particle list used
         *        by the compressor when E has been squeezed out.
         *
         * The exact PTYPE -> particle mapping in SLD's Partid scheme is
         * still inferred; placeholders below for codes 1-4 in particular
         * should be audited when a definitive Partid table is available.
         */
        float massForPtype(int32_t ptype)
        {
            switch (std::abs(ptype))
            {
                case  1: return 0.000511f;  // electron  (placeholder)
                case  2: return 0.0f;       // photon    (placeholder)
                case  3: return 0.0f;       // neutrino  (placeholder)
                case  4: return 0.10566f;   // muon
                case  5: return 0.13957f;   // charged pion
                case  6: return 0.13498f;   // neutral pion
                case  7: return 0.49368f;   // charged kaon
                case  8: return 0.49761f;   // neutral kaon
                case  9: return 0.93827f;   // proton
                case 10: return 0.93957f;   // neutron
                default: return 0.0f;       // unknown -> massless
            }
        }
    } // anonymous namespace

    int32_t MCPART::read(const DataBuffer& buffer, int32_t offset, JazelleEvent&)
    {
        int32_t o = offset;

        // --- P(3): momentum ---
        p[0] = buffer.readFloat(o);  o += 4;
        p[1] = buffer.readFloat(o);  o += 4;
        p[2] = buffer.readFloat(o);  o += 4;

        // --- ID-word ---
        const uint32_t idw = static_cast<uint32_t>(buffer.readInt(o));
        o += 4;

        m_id      = static_cast<int32_t>((idw      ) & 0x1fffu);   // bits 0-12
        parent_id = static_cast<int32_t>((idw >> 13) & 0x1fffu);   // bits 13-25
        const uint32_t charge_code = (idw >> 26) & 0xfu;           // bits 26-29
        const bool     e_omitted   = ((idw >> 30) & 0x1u) != 0u;   // bit 30
        // bit 31 spare

        charge = chargeFromCode(charge_code);

        // --- PTYPE ---
        ptype = buffer.readInt(o);  o += 4;

        // --- ORIGIN ---
        origin = buffer.readInt(o);  o += 4;

        // --- XT(3): termination / production point (cm) ---
        xt[0] = buffer.readFloat(o);  o += 4;
        xt[1] = buffer.readFloat(o);  o += 4;
        xt[2] = buffer.readFloat(o);  o += 4;

        // --- E: stored only if e_omitted == false ---
        const float p2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        if (!e_omitted) {
            e = buffer.readFloat(o);
            o += 4;
        } else {
            const float m = massForPtype(ptype);
            e = std::sqrt(p2 + m * m);
        }

        // --- PTOT: always derived ---
        ptot = std::sqrt(p2);

        // Parent pointer resolved by parseMiniDst after the family is built.
        parent = nullptr;

        return o - offset;   // 36 or 40
    }

} // namespace jazelle