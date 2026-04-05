/**
 * @file PHKTRK.cpp
 * @brief Implementation of the PHKTRK bank read method.
 */

#include "jazelle/banks/PHKTRK.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    int32_t PHKTRK::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        return 0; // Fixed size
    }

} // namespace jazelle