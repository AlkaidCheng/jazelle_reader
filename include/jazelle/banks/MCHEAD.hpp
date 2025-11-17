/**
 * @file MCHEAD.hpp
 * @brief Definition of the MCHEAD (Monte Carlo Header) bank.
 *
 * @see hep.sld.jazelle.family.MCHEAD
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>

namespace jazelle
{
    class DataBuffer; // Forward-declaration
    class JazelleEvent; // Forward-declaration

    /**
     * @struct MCHEAD
     * @brief The Monte Carlo Header bank.
     */
    struct MCHEAD : public Bank
    {
        // --- Member Variables ---
        int32_t ntot;   ///< Total number of final state particles
        int32_t origin; ///< Origin of this event (bitmask for UUBAR, DDBAR, etc.)
        float   ipx;    ///< Primary vertex X momentum
        float   ipy;    ///< Primary vertex Y momentum
        float   ipz;    ///< Primary vertex Z momentum

        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit MCHEAD(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (20).
         * @see hep.sld.jazelle.family.MCHEAD#read(DataBuffer, int)
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };

} // namespace jazelle