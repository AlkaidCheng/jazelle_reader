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
#include <functional>
#include <unordered_map>

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

        /**
         * @brief Load the next record's data buffer without parsing the
         *        MINIDST payload.
         *
         * Advances the stream to the next logical record, reads the logical
         * record header, IEVENTH, PHMTOC, and the raw data block, but stops
         * before decoding the data block into typed banks. Useful when
         * MINIDST parsing fails on a particular event and you still want to
         * inspect the raw bytes.
         *
         * After this call, the data buffer accessible via dumpBinary() /
         * dumpBinaryText() / printBinary() corresponds to this record, and
         * getBankFamilyOffset(s)() can be used to locate bank families
         * within it.
         *
         * The stream is left in the same position it would be in after a
         * successful nextRecord(), so subsequent sequential reads are not
         * affected.
         *
         * @return true on success, false on EOF.
         */
        bool loadEventBuffer();

        /**
         * @brief Same as loadEventBuffer(), but for the event at a specific
         *        index (uses the event index, building it if needed).
         *
         * @param index 0-based event index.
         * @return true on success, false if index is out of range.
         */
        bool loadEventBuffer(int32_t index);

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
         * @brief Gets the format of the *last read* record.
         * @return The format string (e.g., "MINIDST").
         */
        std::string getLastFormat() const;

        /**
         * @brief Rewinds the file pointer to the first event.
         */
        void rewind();

        /**
         * @brief Parallel event reader with callback
         * @param start_idx Starting event index
         * @param count Number of events to read
         * @param callback Function called for each event (must be thread-safe)
         * @param num_threads Number of reader threads (0 = auto)
         */
        void readEventsParallel(
            int32_t start_idx,
            int32_t count,
            std::function<void(int32_t idx, JazelleEvent&)> callback,
            size_t num_threads = 0
        );
        
        /**
         * @brief Read multiple events into a vector (parallel)
         * @return Vector of events in order
         */
        std::vector<JazelleEvent> readEventsBatch(
            int32_t start_idx,
            int32_t count,
            size_t num_threads = 0
        );

        /**
         * @brief Context for event parsing (used internally and by parallel readers)
         */
        struct ParseContext {
            JazelleStream* stream;
            DataBuffer* dataBufferView;
            std::vector<uint8_t>* rawDataBuffer;
            
            ParseContext(JazelleStream* s, DataBuffer* db, std::vector<uint8_t>* rb)
                : stream(s), dataBufferView(db), rawDataBuffer(rb) {}
        };
        
        /**
         * @brief Core parsing logic shared by sequential and parallel reads
         * @param ctx Parsing context with stream and buffers
         * @param event Event object to populate
         * @return true if successfully parsed, false on EOF or error
         */
        static bool parseEvent(ParseContext& ctx, JazelleEvent& event);
        
        /**
         * @brief Parse MINIDST data buffer into event.
         * @param toc Table of contents (bank counts).
         * @param event Event to populate.
         * @param buffer Data buffer to read from.
         * @param family_offsets If non-null, will be populated with a map of
         *        family name (e.g. "PHCHRG") to its starting byte offset
         *        within the buffer. This allows offset discovery without
         *        keeping the parsed event.
         */
        static void parseMiniDst(const PHMTOC& toc, JazelleEvent& event,
                                 const DataBuffer& buffer,
                                 std::unordered_map<std::string, int32_t>*
                                     family_offsets = nullptr);

        /**
         * @brief Returns the raw binary data of the current event's data buffer.
         *
         * The buffer corresponds to the data portion of the most recently read
         * event (via nextRecord() or readEvent()). If no event has been read
         * yet, the returned vector is empty.
         *
         * @param start_offset Byte offset to start from (default: 0).
         * @param end_offset   Byte offset to stop at. If negative, defaults to
         *                     the end of the buffer.
         * @return Vector of raw bytes in the range [start_offset, end_offset).
         */
        std::vector<uint8_t> dumpBinary(int32_t start_offset = 0,
                                        int32_t end_offset = -1) const;

        /**
         * @brief Returns a formatted text representation of the current event's
         *        data buffer (offset, hex, int32, upper/lower 16-bit halves,
         *        float). This is the original dumpBinary() formatting, returned
         *        as a string instead of printed.
         *
         * @param start_offset Byte offset to start from (default: 0).
         * @param end_offset   Byte offset to stop at. If negative, defaults to
         *                     the end of the buffer.
         * @return Formatted dump string.
         */
        std::string dumpBinaryText(int32_t start_offset = 0,
                                   int32_t end_offset = -1) const;

        /**
         * @brief Prints the formatted text dump of the current event's data
         *        buffer to stdout (equivalent to the original dumpBinary).
         *
         * @param start_offset Byte offset to start from (default: 0).
         * @param end_offset   Byte offset to stop at. If negative, defaults to
         *                     the end of the buffer.
         */
        void printBinary(int32_t start_offset = 0,
                         int32_t end_offset = -1) const;

        /**
         * @brief Compute the starting byte offset of every bank family in
         *        the currently loaded data buffer.
         *
         * Requires that a buffer has been loaded via nextRecord(),
         * readEvent(), or loadEventBuffer(). Returns a map of family name
         * (e.g. "MCHEAD", "PHCHRG", "PHBM") to its starting byte offset.
         *
         * If the buffer is malformed and walking fails partway, any
         * families reached before the failure are still returned (missing
         * keys indicate the walk could not reach them).
         *
         * @return Map of family name -> byte offset.
         * @throws std::runtime_error if no MINIDST buffer is currently
         *         loaded.
         */
        std::unordered_map<std::string, int32_t> getBankFamilyOffsets() const;

        /**
         * @brief Get the starting byte offset of a specific bank family.
         *
         * @param familyName Family name, e.g. "PHCHRG", "MCPART".
         * @return Starting byte offset within the data buffer.
         * @throws std::runtime_error if no buffer is loaded, if the family
         *         name is not a known family, or if walking could not reach
         *         it due to a malformed buffer.
         */
        int32_t getBankFamilyOffset(const std::string& familyName) const;

        // --- MINIDST TOC accessor ---

        /**
         * @brief Check whether a valid MINIDST TOC is currently cached.
         *
         * Returns true after a successful nextRecord(), readEvent(), or
         * loadEventBuffer() on a MINIDST record. Returns false before any
         * load, or when the most recent record was not MINIDST.
         */
        bool hasToc() const;

        /**
         * @brief Get the PHMTOC (Table of Contents) of the most recently
         *        loaded MINIDST record.
         *
         * @return The cached PHMTOC (by value).
         * @throws std::runtime_error if no MINIDST buffer is currently
         *         loaded (use hasToc() to test first).
         */
        PHMTOC getToc() const;

    private:
        /**
         * @struct Impl
         * @brief Private implementation (PIMPL) idiom.
         */
        struct Impl;
        std::unique_ptr<Impl> m_impl;

    };

} // namespace jazelle