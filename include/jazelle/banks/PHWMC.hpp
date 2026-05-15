#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <vector>

namespace jazelle
{
    class DataBuffer;
    class JazelleEvent;

    /**
     * @struct PHWMCPair
     * @brief Per-entry pair record in the variable suffix of a PHWMC bank.
     *
     * The semantic meaning of these fields is not fully understood — the
     * PHWMC template has not been sourced yet. Field interpretation:
     *   - count: probably a hit count or weight (sums to PHWMC::total_count)
     *   - id   : probably an MCPART or PHWIC bank id
     *   - value: probably an MCPART id, energy code, or sub-count
     */
    struct PHWMCPair {
        uint16_t count;
        uint16_t id;
        int32_t  value;
    };

    /**
     * @struct PHWMC
     * @brief PHWIC <-> MC relation bank (template not yet sourced).
     *
     * Variable size: 20 bytes of fixed prefix + N × 8 bytes of pair entries,
     * where N == n_pairs is the int32 stored at byte offset +16 within
     * the bank. Total bytes = 20 + 8 * n_pairs.
     *
     * This decoder accounts for the byte layout exactly (so downstream
     * banks parse cleanly), but the semantic interpretation of the
     * prefix fields and the pair entries is partial pending template.
     */
    struct PHWMC : public Bank
    {
        int32_t word1;        ///< Byte +4  (mirror of n_pairs)
        int32_t word2;        ///< Byte +8  (purpose unclear)
        int32_t total_count;  ///< Byte +12 (sum of pair.count across pairs)
        int32_t n_pairs;      ///< Byte +16 (drives variable size)

        std::vector<PHWMCPair> pairs;

        explicit PHWMC(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset,
                     JazelleEvent& event) override;
    };
} // namespace jazelle