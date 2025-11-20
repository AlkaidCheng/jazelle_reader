/**
 * @file JazelleFile.cpp
 * @brief Implementation of the JazelleFile class.
 */

#include "jazelle/JazelleFile.hpp"
#include "JazelleStream.hpp"
#include "DataBuffer.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>

namespace jazelle
{
    /**
     * @struct JazelleFile::Impl
     * @brief Private implementation (PIMPL) struct for JazelleFile.
     */
    struct JazelleFile::Impl
    {
        std::unique_ptr<JazelleStream> stream;
        DataBuffer dataBufferView; // The non-owning buffer view
        std::vector<uint8_t> rawDataBuffer; // The owning data buffer
        
        std::string lastRecordType;
        std::string lastFormat;
        std::vector<int64_t> m_event_offsets;
        bool m_index_built = false;

        explicit Impl(const std::string& filepath)
        {
            stream = std::make_unique<JazelleStream>(filepath);
            // Pre-allocate a reasonable default size
            rawDataBuffer.reserve(512 * 1024); // 512kb
        }

        /**
         * @brief Parses the event data from the stream's *current* position.
         * Assumes the stream is already positioned at the start of a
         * logical record's header.
         */
        bool readCurrentRecord(JazelleEvent& event)
        {
            try
            {
                event.clear();

                // Read logical record header
                [[maybe_unused]] int32_t recno = stream->readInt();
                [[maybe_unused]] int32_t t1 = stream->readInt();
                [[maybe_unused]] int32_t t2 = stream->readInt();
                [[maybe_unused]] int32_t target = stream->readInt();
                
                lastRecordType = stream->readString(8);
                if (lastRecordType == "EOF")
                {
                    return false; // Clean EOF
                }
                
                [[maybe_unused]] int32_t p1 = stream->readInt();
                [[maybe_unused]] int32_t p2 = stream->readInt();
                
                lastFormat = stream->readString(8);
                [[maybe_unused]] std::string context = stream->readString(8);
                    
                [[maybe_unused]] int32_t tocrec = stream->readInt();
                int32_t datrec = stream->readInt();
                [[maybe_unused]] int32_t tocsiz = stream->readInt();
                int32_t datsiz = stream->readInt();
                
                [[maybe_unused]] int32_t tocoff1 = stream->readInt();
                [[maybe_unused]] int32_t tocoff2 = stream->readInt();
                [[maybe_unused]] int32_t tocoff3 = stream->readInt();
                [[maybe_unused]] int32_t datoff = stream->readInt();
                    
                [[maybe_unused]] std::string segname = stream->readString(8);
                std::string usrnam = stream->readString(8);
                [[maybe_unused]] int32_t usroff = stream->readInt();
                    
                [[maybe_unused]] int32_t lrecflgs = stream->readInt();
                [[maybe_unused]] int32_t spare1 = stream->readInt();
                [[maybe_unused]] int32_t spare2 = stream->readInt();
                    
                // Read the event header
                if (usrnam == "IJEVHD")
                {
                    event.ieventh.read(*stream);
                }

                if (lastFormat == "MINIDST")
                {
                    PHMTOC toc(*stream);
                    
                    if (datrec > 0)
                    {
                        // This is tricky: datrec > 0 means the data is in the
                        // *next* logical record.
                        if (!stream->nextLogicalRecord())
                        {
                             throw std::runtime_error("Unexpected EOF looking for MINIDST data record.");
                        }
                    }

                    // Read the entire data record
                    stream->readToVector(rawDataBuffer, datsiz);
                    // Set the non-owning view
                    dataBufferView.setData(
                        {rawDataBuffer.data(), static_cast<size_t>(datsiz)}
                    );
                    // Parse the data buffer
                    parseMiniDst(toc, event);
                }
                
                return true;
            }
            catch (const JazelleEOF& e)
            {
                // This is a clean EOF detected mid-read.
                return false;
            }
            catch (const std::exception& e)
            {
                // This is a real error
                throw std::runtime_error(std::string("Error during readCurrentRecord(): ") + e.what());
            }
        }


        /**
         * @brief Implements the bank-parsing loop from MINIDST.java.
         */
        void parseMiniDst(const PHMTOC& toc, JazelleEvent& event)
        {
            int32_t offset = 0;
            // This parsing order is a direct translation of MINIDST.java

            // Read MCHead (ID 1, not read from stream)
            Bank* mchead = event.add("MCHEAD", 1);
            offset += mchead->read(dataBufferView, offset, event);
            
            // Read MCPart
            for (int32_t i = 0; i < toc.m_nMcPart; i++)
            {
                int32_t id = dataBufferView.readInt(offset);
                offset += 4;
                Bank* mcpart = event.add("MCPART", id);
                offset += mcpart->read(dataBufferView, offset, event);
            }
            // Read PHPSUM
            for (int32_t i = 0; i < toc.m_nPhPSum; i++)
            {
                int32_t id = dataBufferView.readInt(offset);
                offset += 4;
                Bank* phpsum = event.add("PHPSUM", id);
                offset += phpsum->read(dataBufferView, offset, event);
            }
            // Read PHCHRG
            for (int32_t i = 0; i < toc.m_nPhChrg; i++)
            {
                int32_t id = dataBufferView.readInt(offset);
                offset += 4;
                Bank* phchrg = event.add("PHCHRG", id);
                offset += phchrg->read(dataBufferView, offset, event);
            }
            // Read PHKLUS
            for (int32_t i = 0; i < toc.m_nPhKlus; i++)
            {
                int32_t id = dataBufferView.readInt(offset);
                offset += 4;
                Bank* phklus = event.add("PHKLUS", id);
                offset += phklus->read(dataBufferView, offset, event);
            }
            // Read PHWIC
            for (int32_t i = 0; i < toc.m_nPhWic; i++)
            {
                int32_t id = dataBufferView.readInt(offset);
                offset += 4;
                Bank* phwic = event.add("PHWIC", id);
                offset += phwic->read(dataBufferView, offset, event);
            }
            // Read PHCRID
            for (int32_t i = 0; i < toc.m_nPhCrid; i++)
            {
                int32_t id = dataBufferView.readInt(offset) & 0xffff;
                Bank* phcrid = event.add("PHCRID", id);
                offset += phcrid->read(dataBufferView, offset, event);
            }
            // Read PHKTRK
            for (int32_t i = 0; i < toc.m_nPhKTrk; i++)
            {
                int32_t id = dataBufferView.readInt(offset) & 0xffff;
                Bank* phktrk = event.add("PHKTRK", id);
                offset += phktrk->read(dataBufferView, offset, event);
            }
            // Read PHKELID
            for (int32_t i = 0; i < toc.m_nPhKElId; i++)
            {
                int32_t id = dataBufferView.readInt(offset) & 0xffff;
                Bank* phkelid = event.add("PHKELID", id);
                offset += phkelid->read(dataBufferView, offset, event);
            }
            // Resolve MCPART parent pointers
            size_t mcpartCount = event.mcpartFamily.count();
            for (size_t i = 0; i < mcpartCount; ++i)
            {
                MCPART* part = event.mcpartFamily.at(i);
                
                if (part && part->parent_id > 0)
                {
                    // Look up the parent by ID using the event's finder
                    part->parent = event.findMCPART(part->parent_id);
                }
                else if (part)
                {
                    part->parent = nullptr;
                }
            }
        }
    };

    // --- Public JazelleFile Methods ---

    JazelleFile::JazelleFile(const std::string& filepath)
    try : m_impl(std::make_unique<Impl>(filepath))
    {
        // Constructor body is intentionally empty.
        // The Impl constructor does all the work.
    }
    catch (const std::exception& e)
    {
        // Add context to exceptions thrown during opening
        throw std::runtime_error("Failed to open JazelleFile '" + filepath + "': " + e.what());
    }

    JazelleFile::~JazelleFile()
    {
        // m_impl is automatically destroyed, closing files
    }

    bool JazelleFile::nextRecord(JazelleEvent& event)
    {
        // 1. Advance the stream to the next logical record
        if (!m_impl->stream->nextLogicalRecord())
        {
            return false; // Clean EOF
        }
        
        // 2. Read the record at the new position
        return m_impl->readCurrentRecord(event);
    }

    // --- New Indexing Methods ---

    void JazelleFile::buildIndex()
    {
        if (m_impl->m_index_built) return;

        // Rewind to the start of the *first data record*
        m_impl->stream->rewind();
        m_impl->m_event_offsets.clear();

        while (true)
        {
            // Get the offset of the header we just rewound to (or
            // advanced to in the last loop iteration).
            m_impl->m_event_offsets.push_back(m_impl->stream->getCurrentRecordOffset());

            // Advance to the *next* record's header
            if (!m_impl->stream->nextLogicalRecord())
            {
                // We hit EOF, so the last record we added was the final one.
                break;
            }
        }

        m_impl->m_index_built = true;
        // Rewind again so the file is ready for reading
        m_impl->stream->rewind();
    }

    int32_t JazelleFile::getTotalEvents()
    {
        if (!m_impl->m_index_built)
        {
            buildIndex();
        }
        return m_impl->m_event_offsets.size();
    }

    bool JazelleFile::readEvent(int32_t index, JazelleEvent& event)
    {
        if (!m_impl->m_index_built)
        {
            buildIndex();
        }

        if (index < 0 || index >= m_impl->m_event_offsets.size())
        {
            return false; // Index out of bounds
        }

        // 1. Seek the stream to the pre-calculated offset
        int64_t offset = m_impl->m_event_offsets[index];
        m_impl->stream->seekTo(offset);

        // 2. Read the record at that exact position
        return m_impl->readCurrentRecord(event);
    }


    // --- Accessors ---

    std::string JazelleFile::getFileName() const
    {
        return m_impl->stream->getFileName();
    }

    std::chrono::system_clock::time_point JazelleFile::getCreationDate() const
    {
        return m_impl->stream->getCreationDate();
    }

    std::chrono::system_clock::time_point JazelleFile::getModifiedDate() const
    {
        return m_impl->stream->getModifiedDate();
    }

    std::string JazelleFile::getLastRecordType() const
    {
        return m_impl->lastRecordType;
    }

} // namespace jazelle