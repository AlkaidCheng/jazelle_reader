/**
 * @file IEVENTH.hpp
 * @brief Definition of the IEVENTH (Event Header) bank.
 *
 * This bank serves as the primary metadata block for the event, tracking
 * run numbers, timestamps, and high-level trigger states.
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <chrono>

namespace jazelle
{
    class JazelleStream;

    /**
     * @struct IEVENTH
     * @brief The Event Header bank.
     */
    struct IEVENTH : public Bank
    {
        int32_t header;   ///< Internal Jazelle pointer to the header bank in this segment
        int32_t run;      ///< Run number
        int32_t event;    ///< Event number within the run
        
        /// UTC Timestamp of when the event was recorded
        std::chrono::system_clock::time_point evttime;
        
        float   weight;   ///< Event weight (Always 1.0 for real data; varies for MC)
        int32_t evttype;  ///< Event generation type (0=PHYSICS, 1=TRUTH, 2=FASTMC, etc.)
        int32_t trigger;  ///< Hardware trigger mask for this event
        
        /**
         * @brief Constructor. IEVENTH is always ID 1.
         */
        IEVENTH() : Bank(1) {} 

        void read(JazelleStream& stream);

        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override
        {
            return 0; // Handled directly by stream
        }
    };
} // namespace jazelle