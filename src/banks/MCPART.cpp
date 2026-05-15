#include "jazelle/banks/MCPART.hpp"
#include "DataBuffer.hpp"

#include <cmath>
#include <cstdlib>

namespace jazelle
{
    namespace
    {
        /// Returns +0.0f if x is -0.0f, otherwise x unchanged.
        /// IEEE-754: -0.0f and +0.0f compare equal but have different bit
        /// patterns; `x + 0.0f` is the canonical signed-zero-to-positive trick.
        inline float canonicalize_zero(float x) {
            return x + 0.0f;
        }

        float chargeFromCode(uint32_t code)
        {
            static constexpr float kMagnitude[8] = {
                0.0f,
                2.0f / 3.0f,
                1.0f / 3.0f,
                1.0f,
                2.0f,
                0.0f, 0.0f, 0.0f
            };
            const uint32_t mag_idx  = code & 0x7u;
            const bool     positive = ((code >> 3) & 0x1u) != 0u;
            const float    m        = kMagnitude[mag_idx];
            // Use canonicalize_zero so charge code 0 (and code 8) never yield
            // a stray -0.0f when the sign bit is clear/set on a zero magnitude.
            return canonicalize_zero(positive ? m : -m);
        }

        float massForPtype(int32_t ptype)
        {
            switch (std::abs(ptype))
            {
                case  1: return 0.000511f;
                case  2: return 0.0f;
                case  3: return 0.0f;
                case  4: return 0.10566f;
                case  5: return 0.13957f;
                case  6: return 0.13498f;
                case  7: return 0.49368f;
                case  8: return 0.49761f;
                case  9: return 0.93827f;
                case 10: return 0.93957f;
                default: return 0.0f;
            }
        }
    } // anonymous namespace

    int32_t MCPART::read(const DataBuffer& buffer, int32_t offset, JazelleEvent&)
    {
        int32_t o = offset;

        // --- P(3) ---
        px = buffer.readFloat(o);  o += 4;
        py = buffer.readFloat(o);  o += 4;
        pz = buffer.readFloat(o);  o += 4;

        // --- ID-word ---
        const uint32_t idw = static_cast<uint32_t>(buffer.readInt(o));
        o += 4;

        m_id      = static_cast<int32_t>((idw      ) & 0x1fffu);
        parent_id = static_cast<int32_t>((idw >> 13) & 0x1fffu);
        const uint32_t charge_code = (idw >> 26) & 0xfu;
        const bool     e_omitted   = ((idw >> 30) & 0x1u) != 0u;

        charge = chargeFromCode(charge_code);

        // --- PTYPE ---
        ptype = buffer.readInt(o);  o += 4;

        // --- ORIGIN ---
        origin = buffer.readInt(o);  o += 4;

        // --- XT(3) ---
        xt_x = buffer.readFloat(o);  o += 4;
        xt_y = buffer.readFloat(o);  o += 4;
        xt_z = buffer.readFloat(o);  o += 4;

        // --- E: stored if !e_omitted, else reconstruct from PTYPE mass ---
        const float p2 = px*px + py*py + pz*pz;
        if (!e_omitted) {
            e = buffer.readFloat(o);
            o += 4;
        } else {
            const float m = massForPtype(ptype);
            e = std::sqrt(p2 + m * m);
        }

        parent = nullptr;
        return o - offset;
    }

} // namespace jazelle