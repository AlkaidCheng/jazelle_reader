/**
 * @file IEVENTH.hpp
 * @brief Definition of the IEVENTH (Event Header) bank.
 *
 * This bank is special: it's read directly from the JazelleInputStream
 * as part of the logical record header, not from the main DataBuffer.
 *
 * @see hep.sld.jazelle.family.IEVENTH
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <chrono>

namespace jazelle
{
    class JazelleStream; // Forward-declaration

    /**
     * @struct IEVENTH
     * @brief The Event Header bank.
     */
    struct IEVENTH : public Bank
    {
        // --- Member Variables ---
        int32_t header;   ///< "Pointer to header bank in this segment"
        int32_t run;      ///< "Run number"
        int32_t event;    ///< "Event number"
        
        /// "Time when event was created" (Java Date -> C++ time_point)
        std::chrono::system_clock::time_point evttime;
        
        float   weight;   ///< "Event weight (1.0 for real data)"
        int32_t evttype;  ///< "Event type" (0=PHYSICS, 1=TRUTH, 2=FASTMC, etc.)
        int32_t trigger;  ///< "Trigger mask for this event"
        
        /**
         * @brief Constructor. IEVENTH is always ID 1.
         */
        IEVENTH() : Bank(1) {} // ID is hardcoded to 1 in JazelleFile.java

        /**
         * @brief Special read method for IEVENTH.
         * Reads data directly from the input stream, not the DataBuffer.
         * @param stream The Jazelle input stream.\
         */
        void read(JazelleStream& stream);

        /**
         * @brief Default implementation for the virtual Bank::read.
         * This bank type is not read from the DataBuffer, so this
         * method should never be called.
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override
        {
            // This bank is read specially from the stream
            return 0;
        }
    };

} // namespace jazelle