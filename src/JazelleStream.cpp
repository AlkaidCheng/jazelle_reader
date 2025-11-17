/**
 * @file JazelleStream.cpp
 * @brief Implementation of the JazelleStream class.
 */

#include "JazelleStream.hpp"
#include "utils/BinaryIO.hpp" // Use the central utility
#include <stdexcept>
#include <cstring> // For std::memcpy
#include <algorithm> // For std::find_if_not

namespace jazelle
{

// --- Constructor / Destructor ---

JazelleStream::JazelleStream(const std::string& filepath)
{
    openFile(filepath);
    try
    {
        readLogicalHeader();
        
        std::string check = readString(8);
        if (check != "JAZELLE")
        {
            throw std::runtime_error("Not a JAZELLE format file.");
        }
        
        [[maybe_unused]] int32_t ibmvax = readInt(); // "ibm/vax" flag, unused
        m_created = readDate();
        m_modified = readDate();
        m_nmod = readInt();
        m_name = readString(80);

        if (!nextLogicalRecord())
        {
            // This means the file has a header but NO data.
            throw std::runtime_error("File contains header but no data records.");
        }

        m_first_record_offset = getCurrentRecordOffset();
    }
    catch(const JazelleEOF& e)
    {
        throw std::runtime_error("File is truncated or empty.");
    }
}

JazelleStream::~JazelleStream()
{
    closeFile();
}

// --- Low-Level I/O ---

void JazelleStream::openFile(const std::string& filepath)
{
    m_fileStream.open(filepath, std::ios::binary);
    if (!m_fileStream.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    m_string_buffer.resize(80); // Default size
}

void JazelleStream::closeFile()
{
    if (m_fileStream.is_open()) m_fileStream.close();
}

// --- Record Navigation ---

int16_t JazelleStream::readShortHeader()
{
    m_nBytes += 2;
    
    uint8_t b[2];
    m_fileStream.read(reinterpret_cast<char*>(b), 2);
    if (m_fileStream.gcount() != 2) {
        throw JazelleEOF("short header");
    }

    return utils::readLeShort(b);
}

void JazelleStream::readPhysicalHeader()
{
    // Store the offset *before* reading the header.
    m_fileStream.clear(); // Ensure tellg() is accurate
    m_current_record_offset = m_fileStream.tellg();
    
    m_nBytes = 0;
    m_reclen = readShortHeader();
    if (m_reclen < 0)
    {
        throw JazelleEOF();
    }
    [[maybe_unused]] int16_t prres = readShortHeader(); // "reserved"
}

void JazelleStream::readLogicalHeader()
{
    readPhysicalHeader();
    [[maybe_unused]] int16_t lrlen = readShortHeader();
    int16_t lrcnt = readShortHeader();

    if ((lrcnt & 0xfffffffc) != 0) throw std::runtime_error("IOSYNCH1");
    bool continued = (lrcnt & 2) != 0;
    
    // On the first call (from constructor), m_toBeContinued is false,
    // and the first record's 'continued' flag should also be false.
    // When seeking, this check is still valid.
    if (continued != m_toBeContinued)
    {
        throw std::runtime_error("IOSYNCH2");
    }
    m_toBeContinued = (lrcnt & 1) != 0;
}

void JazelleStream::nextPhysicalRecord()
{
    // Skip remaining bytes in this record
    skip(static_cast<int64_t>(m_reclen) - m_nBytes);
    readLogicalHeader();
}

void JazelleStream::skip(int64_t n)
{
    if (n <= 0) return;
    
    m_fileStream.ignore(n);
    if (m_fileStream.eof())
    {
        m_eof = true;
        throw JazelleEOF("next physical record");
    }
}

bool JazelleStream::nextLogicalRecord()
try
{
    while (m_toBeContinued)
    {
        nextPhysicalRecord();
    }
    nextPhysicalRecord();

    return true;
}
catch (const JazelleEOF& e)
{
    // This is a clean end-of-file
    return false;
}

int64_t JazelleStream::tellg()
{
    m_fileStream.clear();
    return m_fileStream.tellg();
}

void JazelleStream::seekTo(int64_t offset)
try
{
    // Clear any fail/eof bits before seeking
    m_fileStream.clear();
    m_fileStream.seekg(offset);

    if (m_fileStream.fail())
    {
        throw std::runtime_error("Seek failed");
    }

    // Reset stream state and re-sync by reading the headers
    // at the new location.
    m_eof = false;
    m_toBeContinued = false; // We must be at the start of a record
    readLogicalHeader(); // Re-sync the stream state
}
catch (const JazelleEOF& e)
{
    throw std::runtime_error("Seek failed due to EOF");
}

void JazelleStream::rewind()
{
    seekTo(m_first_record_offset);
}


// --- Stream Data Readers ---

// note: no eof check
int32_t JazelleStream::read()
{
    if (m_nBytes++ < m_reclen)
        return m_fileStream.get();
    
    readLogicalHeader();
    m_nBytes++;
    return m_fileStream.get();
}

void JazelleStream::readFully(uint8_t* buffer, int32_t length)
try
{
    for (int32_t i = 0; i < length; ++i)
    {
        buffer[i] = static_cast<uint8_t>(read());
    }
}
catch(const JazelleEOF& e)
{
    throw std::runtime_error("Unexpected EOF in readFully.");
}

void JazelleStream::readToVector(std::vector<uint8_t>& buffer, int32_t length)
{
    if (buffer.size() < static_cast<size_t>(length))
    {
        buffer.resize(length);
    }
    readFully(buffer.data(), length);
}

int16_t JazelleStream::readShort()
{
    uint8_t b[2];
    readFully(b, 2);
    return utils::readLeShort(b);
}

int32_t JazelleStream::readInt()
{
    uint8_t b[4];
    readFully(b, 4);
    return utils::readLeInt(b);
}

int64_t JazelleStream::readLong()
{
    uint8_t b[8];
    readFully(b, 8);
    return utils::readLeLong(b);
}

float JazelleStream::readFloat()
{
    int32_t fbits = readInt();
    return utils::vaxToIeeeFloat(fbits);
}

std::string JazelleStream::readString(int32_t len)
{
    if (m_string_buffer.size() < static_cast<size_t>(len))
    {
        m_string_buffer.resize(len);
    }
    
    readFully(m_string_buffer.data(), len);
    
    // Find last non-space character
    auto it = std::find_if_not(
        m_string_buffer.rbegin() + (m_string_buffer.size() - len), 
        m_string_buffer.rend(), 
        [](unsigned char c){ return c == ' '; }
    );
    
    int32_t len_without_spaces = 0;
    if (it != m_string_buffer.rend())
    {
        len_without_spaces = &(*it) - m_string_buffer.data() + 1;
    }

    return std::string(m_string_buffer.begin(), m_string_buffer.begin() + len_without_spaces);
}

std::chrono::system_clock::time_point JazelleStream::readDate()
{
    int64_t vax_time = readLong();
    if (vax_time == 0) return std::chrono::system_clock::time_point{};
    
    int64_t vax_millis = vax_time / 10000;
    int64_t java_millis = vax_millis - 3506716800730L;
    
    return std::chrono::system_clock::from_time_t(java_millis / 1000) +
           std::chrono::milliseconds(java_millis % 1000);
}

} // namespace jazelle