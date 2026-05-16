#pragma once

#include "../Bank.hpp"
#include <cstdint>

namespace jazelle
{
    class DataBuffer;
    class JazelleEvent;
    struct MCPART;
    struct PHPOINT;

    /**
     * @struct MCPNT
     * @brief Relational table linking an MCPART to a PHPOINT.
     *
     * Two keys: MCPART (first) and PHPOINT (second). The bank is written
     * as two passes back-to-back in the buffer:
     *   - Pass 1 (16 B per entry): data in MCPART-key traversal order
     *   - Pass 2 (4 B per entry):  PHPOINT_id list in PHPOINT-key order
     */
    struct MCPNT : public Bank
    {
        // Bits in REASON (per the MCPNT template)
        struct Reason {
            static constexpr int32_t CDC     = (1 << 0);
            static constexpr int32_t VTX     = (1 << 1);
            static constexpr int32_t EDCI    = (1 << 2);
            static constexpr int32_t EDCO    = (1 << 3);
            static constexpr int32_t WIC     = (1 << 4);
            static constexpr int32_t KAL     = (1 << 8);
            static constexpr int32_t VIRTUAL = (1 << 24);
            static constexpr int32_t PARTIAL = (1 << 25);
        };

        /// Sentinel value stored in NHITS when no tracking hits are recorded.
        static constexpr int32_t NHITS_NONE = -999;

        int32_t  mcpart_id;     ///< First-key DATA pointer (set by read)
        MCPART*  mcpart = nullptr;
        int32_t  phpoint_id;    ///< Second-key DATA pointer (set by Pass 2)
        PHPOINT* phpoint = nullptr;
        int32_t  reason;
        int32_t  nhits;
        float    econtrib;

        explicit MCPNT(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset,
                     JazelleEvent& event) override;
    };
} // namespace jazelle