/**
 * @file JazelleStream.hpp
 * @brief Internal class for reading Jazelle physical/logical records.
 *
 * This class handles the complex I/O logic, physical record skipping,
 * logical record assembly, and stream-based data type conversion
 * (VAX float, little-endian).
 */

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>

// (JazelleEOF exception class remains the same)
class JazelleEOF : public std::runtime_error {
public:
    JazelleEOF()
        : std::runtime_error("unexpected EOF while reading file") {}

    explicit JazelleEOF(const std::string& dtype)
        : std::runtime_error("unexpected EOF while reading " + dtype) {}
};

namespace jazelle
{
    class JazelleStream
    {
    public:
        /**
         * @brief Constructor.
         * Opens the file and reads the file header.
         * @param filepath Path to the file.
         */
        JazelleStream(const std::string& filepath);

        /**
         * @brief Destructor. Closes file handles.
         */
        ~JazelleStream();

        // --- Record Navigation ---

        /**
         * @brief Skips to the start of the next logical record.
         * @return true if successful, false on EOF.
         */
        bool nextLogicalRecord();

        /**
         * @brief Seeks to a specific byte offset and re-syncs the stream.
         * @param offset The absolute byte offset in the file.
         */
        void seekTo(int64_t offset);

        /**
         * @brief Gets the current absolute byte offset in the file.
         * @return The file position.
         */
        int64_t tellg();

        /**
         * @brief Seeks back to the start of the first logical record.
         */
        void rewind();

        /**
         * @brief Gets the file offset of the most recently read record header.
         * @return The absolute byte offset of the record start.
         */
        int64_t getCurrentRecordOffset() const { return m_current_record_offset; }

        // --- Stream Data Readers ---

        /**
         * @brief Reads a 2-byte little-endian short from the stream.
         */
        int16_t readShort();
        
        /**
         * @brief Reads a 4-byte little-endian integer from the stream.
         */
        int32_t readInt();
        
        /**
         * @brief Reads an 8-byte little-endian long from the stream.
         */
        int64_t readLong();
        
        /**
         * @brief Reads a 4-byte VAX F_FLOAT from the stream.
         */
        float readFloat();
        
        /**
         * @brief Reads a fixed-length string from the stream.
         * @param len The number of bytes to read.
         * @return The read string, trimmed of trailing spaces.
         */
        std::string readString(int32_t len);
        
        /**
         * @brief Reads an 8-byte VAX date and converts it.
         * @return A C++ time_point.
         */
        std::chrono::system_clock::time_point readDate();
        
        /**
         * @brief Reads a block of data fully.
         * @param buffer Buffer to fill.
         * @param length Number of bytes to read.
         */
        void readFully(uint8_t* buffer, int32_t length);
        
        /**
         * @brief Gets the number of bytes read in the current physical record.
         */
        int32_t getNBytes() const { return m_nBytes; }
        
        /**
         * @brief Fills a vector with the specified number of bytes.
         * @param buffer The vector to fill.
         * @param length Number of bytes to read.
         */
        void readToVector(std::vector<uint8_t>& buffer, int32_t length);

        // --- Header Data Accessors ---
        std::string getFileName() const { return m_name; }
        std::chrono::system_clock::time_point getCreationDate() const { return m_created; }
        std::chrono::system_clock::time_point getModifiedDate() const { return m_modified; }

    private:
        // --- Low-Level I/O ---
        void openFile(const std::string& filepath);
        void closeFile();
        
        // --- Physical Record Logic ---
        int16_t readShortHeader();
        void readPhysicalHeader();
        void readLogicalHeader();
        void nextPhysicalRecord();
        void skip(int64_t n);
        
        /**
         * @brief Reads one byte, advancing physical record if needed.
         */
        int32_t read();

        std::ifstream m_fileStream;
        
        std::vector<uint8_t> m_string_buffer; // Re-usable buffer

        // --- Record State ---
        int32_t m_reclen = 0; // Length of current physical record
        int32_t m_nBytes = 0; // Bytes read in current physical record
        bool m_toBeContinued = false; // Logical record flag
        bool m_eof = false; // End-of-file flag
        
        // --- New indexing member ---
        int64_t m_first_record_offset = 0; // Offset after file header

        /**
         * @brief The start offset of the current physical record.
         * This is updated by readPhysicalHeader().
         */
        int64_t m_current_record_offset = 0;

        // --- File Header Data ---
        std::string m_name;
        std::chrono::system_clock::time_point m_created;
        std::chrono::system_clock::time_point m_modified;
        int32_t m_nmod;
    };

} // namespace jazelle