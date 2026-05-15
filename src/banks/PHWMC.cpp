#include "jazelle/banks/PHWMC.hpp"
#include "DataBuffer.hpp"

namespace jazelle
{
    int32_t PHWMC::read(const DataBuffer& buffer, int32_t offset, JazelleEvent&)
    {
        int32_t o = offset;

        m_id        = buffer.readInt(o);  o += 4;
        word1       = buffer.readInt(o);  o += 4;
        word2       = buffer.readInt(o);  o += 4;
        total_count = buffer.readInt(o);  o += 4;
        n_pairs     = buffer.readInt(o);  o += 4;

        pairs.clear();
        if (n_pairs > 0) {
            pairs.reserve(static_cast<size_t>(n_pairs));
            for (int32_t i = 0; i < n_pairs; ++i) {
                const uint32_t packed =
                    static_cast<uint32_t>(buffer.readInt(o));
                o += 4;
                const int32_t value = buffer.readInt(o);
                o += 4;

                PHWMCPair pr;
                pr.count = static_cast<uint16_t>((packed >> 16) & 0xffffu);
                pr.id    = static_cast<uint16_t>( packed        & 0xffffu);
                pr.value = value;
                pairs.push_back(pr);
            }
        }

        return o - offset;   // 20 + 8 * n_pairs
    }
} // namespace jazelle