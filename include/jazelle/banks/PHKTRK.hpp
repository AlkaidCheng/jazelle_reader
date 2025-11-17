/**
 * @file PHKTRK.hpp
 * @brief Definition of the PHKTRK bank (STUB).
 *
 * The Java file PHKTRK.java is a stub and reads 0 bytes.
 * We replicate that behavior.
 *
 * @see hep.sld.jazelle.family.PHKTRK
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>

namespace jazelle
{
    class DataBuffer; // Forward-declaration
    class JazelleEvent; // Forward-declaration

    /**
     * @struct PHKTRK
     * @brief A stub bank, as defined in the Java source.
     */
    struct PHKTRK : public Bank
    {
        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHKTRK(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (0).
         * @see hep.sld.jazelle.family.PHKTRK#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override
        {
            // This is a stub bank in the original Java code.
            return 0;
        }
    };

} // namespace jazelle