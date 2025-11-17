/**
 * @file Bank.hpp
 * @brief Abstract base class for all Jazelle data banks.
 *
 * This file defines the 'Bank' interface that all specific data banks
 * (like MCHEAD, PHCHRG, etc.) must implement. It's a translation of
 * the abstract Bank.java class.
 *
 * @see hep.sld.jazelle.Bank
 */

#pragma once

#include <cstdint>
#include <string>

namespace jazelle
{
    // Forward-declarations for internal reader classes
    class DataBuffer;
    class JazelleInputStream;
    
    // Forward-declaration for the main event class
    class JazelleEvent;

    /**
     * @class Bank
     * @brief Abstract base class for a Jazelle data bank.
     *
     * In the original Java, Banks were stored in a generic Family object.
     * In C++, this abstract class provides the common interface for
     * ID management and data deserialization.
     */
    class Bank
    {
    public:
        /**
         * @brief Constructs a Bank with a specific ID.
         * @param id The bank's unique identifier within its family.
         */
        explicit Bank(int32_t id) : m_id(id) {}

        /**
         * @brief Virtual destructor to ensure correct cleanup of derived classes.
         */
        virtual ~Bank() = default;

        // --- Core Deserialization Interface ---

        /**
         * @brief Reads the bank's data from the main DataBuffer.
         *
         * This is the primary method for all fixed-size banks. It populates
         * the derived class's members from the buffer at the given offset.
         *
         * We pass the JazelleEvent to allow for cross-bank pointer resolution,
         * as seen in PHKELID.java.
         *
         * @param buffer The raw data buffer for the current record.
         * @param offset The starting byte offset for this bank's data.
         * @param event The parent event, used for resolving bank-to-bank links.
         * @return The total number of bytes read by this bank.
         * @see hep.sld.jazelle.Bank#read(DataBuffer, int)
         */
        virtual int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) = 0;

        // --- Accessors ---

        /**
         * @brief Gets the bank's unique ID.
         * @return The bank's ID.
         * @see hep.sld.jazelle.Bank#getID()
         */
        int32_t getId() const { return m_id; }

    protected:
        int32_t m_id; ///< The bank's unique ID.
    };

} // namespace jazelle