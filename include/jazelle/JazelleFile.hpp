/**
 * @file JazelleFile.hpp
 * @brief The main user-facing API for reading a Jazelle file.
 *
 * This class corresponds to JazelleFile.java. It handles file opening,
 * record-by-record reading, and parsing.
 *
 * This modified version adds support for building an event index
 * for efficient random access.
 *
 * @see hep.sld.jazelle.JazelleFile
 */

#pragma once

#include "JazelleEvent.hpp" // Includes all bank headers
#include "PHMTOC.hpp"
#include <string>
#include <memory>
#include <chrono>

namespace jazelle
{
    /**
     * @class JazelleFile
     * @brief Main class for reading a Jazelle file.
     *
     * This class opens a .jazelle or .jazelle.gz file and provides
     * a nextRecord() method to iterate through the events.
     */
    class JazelleFile
    {
    public:
        /**
         * @brief Opens a Jazelle file for reading.
         *
         * @param filepath Path to the .jazelle file.
         * @throws std::runtime_error if the file cannot be opened or is
         * not a valid Jazelle file.
         * @see hep.sld.jazelle.JazelleFile#JazelleFile(File)
         */
        explicit JazelleFile(const std::string& filepath);

        /**
         * @brief Destructor.
         * Cleans up file handles and internal resources.
         */
        ~JazelleFile();

        // --- Core API ---

        /**
         * @brief Reads the next logical record from the file sequentially.
         *
         * @param event A JazelleEvent object to be populated.
         * @return true if a record was successfully read, false if
         * the end of the file is reached.
         * @throws std::runtime_error on a parsing or I/O error.
         * @see hep.sld.jazelle.JazelleFile#nextRecord()
         */
        bool nextRecord(JazelleEvent& event);

        /**
         * @brief Scans the entire file to build an event offset index.
         * This is called automatically by getTotalEvents() or readEvent()
         * on their first use.
         */
        void buildIndex();

        /**
         * @brief Gets the total number of events in the file.
         * @return The total event count.
         */
        int32_t getTotalEvents();

        /**
         * @brief Reads a specific event by its index.
         *
         * @param index The 0-based event index to read.
         * @param event A JazelleEvent object to be populated.
         * @return true if the event was read, false if index is out of bounds.
         */
        bool readEvent(int32_t index, JazelleEvent& event);


        // --- File Metadata Accessors ---

        /**
         * @brief Gets the file name from the Jazelle header.
         * @return The internal file name string.
         * @see hep.sld.jazelle.JazelleInputStream#getName()
         */
        std::string getFileName() const;

        /**
         * @brief Gets the creation timestamp from the Jazelle header.
         * @return The file creation time.
         * @see hep.sld.jazelle.JazelleInputStream#getCreated()
         */
        std::chrono::system_clock::time_point getCreationDate() const;

        /**
         * @brief Gets the modification timestamp from the Jazelle header.
         * @return The file modification time.
         * @see hep.sld.jazelle.JazelleInputStream#getModified()
         */
        std::chrono::system_clock::time_point getModifiedDate() const;

        /**
         * @brief Gets the record type of the *last read* record.
         * @return The record type string (e.g., "DATA", "EOF").
         * @see hep.sld.jazelle.JazelleFile#getRecType()
         */
        std::string getLastRecordType() const;

        /**
         * @brief Rewinds the file pointer to the first event.
         */
        void rewind();

    private:
        /**
         * @struct Impl
         * @brief Private implementation (PIMPL) idiom.
         */
        struct Impl;
        
        /// @brief Unique pointer to the private implementation.
        std::unique_ptr<Impl> m_impl;
    };

} // namespace jazelle