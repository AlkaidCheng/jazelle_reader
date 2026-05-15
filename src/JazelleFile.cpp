/**
 * @file JazelleFile.cpp
 * @brief Implementation of the JazelleFile class.
 */

#include "jazelle/JazelleFile.hpp"
#include "utils/LockFreeQueue.hpp"
#include "utils/ObjectPool.hpp"
#include "JazelleStream.hpp"
#include "DataBuffer.hpp"
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <set>

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

        PHMTOC lastToc;
        bool   lastTocValid = false;

        explicit Impl(const std::string& filepath)
        {
            stream = std::make_unique<JazelleStream>(filepath);
            // Pre-allocate a reasonable default size
            rawDataBuffer.reserve(512 * 1024); // 512kb
        }

        /**
         * @brief Create a parse context from this implementation
         */
        ParseContext getParseContext() {
            return ParseContext(stream.get(), &dataBufferView, &rawDataBuffer);
        }
        
        /**
         * @brief Reads event at current stream position
         */
        bool readCurrentRecord(JazelleEvent& event)
        {
            auto ctx = getParseContext();
            std::string recordType;
            std::string format;
            PHMTOC toc;

            bool success = parseEventWithMetadata(
                ctx, event, recordType, format,
                /*out_toc=*/&toc,
                /*parse_minidst=*/true);

            if (success) {
                lastRecordType = recordType;
                lastFormat     = format;
                lastToc        = toc;
                lastTocValid   = (format == "MINIDST");
            }
            return success;
        }

        /**
         * @brief Reads event buffer without parsing MINIDST 
         */
        bool readCurrentBufferOnly()
        {
            auto ctx = getParseContext();
            std::string recordType;
            std::string format;
            PHMTOC toc;

            JazelleEvent scratch;

            bool success = parseEventWithMetadata(
                ctx, scratch, recordType, format,
                /*out_toc=*/&toc,
                /*parse_minidst=*/false);

            if (success) {
                lastRecordType = recordType;
                lastFormat     = format;
                lastToc        = toc;
                lastTocValid   = (format == "MINIDST");
            }
            return success;
        }

        /**
         * @brief Helper that captures metadata during parsing
         * DECLARATION - must be static to be called from JazelleFile::parseEvent
         */
        static bool parseEventWithMetadata(
            JazelleFile::ParseContext& ctx,
            JazelleEvent& event,
            std::string& recordType,
            std::string& format,
            PHMTOC* out_toc = nullptr,
            bool parse_minidst = true);
    };

    // Static method: Core parsing logic
    /**
     * @brief Helper that captures metadata during parsing
     */
    bool JazelleFile::Impl::parseEventWithMetadata(
        JazelleFile::ParseContext& ctx,
        JazelleEvent& event,
        std::string& recordType,
        std::string& format,
        PHMTOC* out_toc,
        bool parse_minidst)
    {
        try
        {
            event.clear();

            // Read logical record header
            [[maybe_unused]] int32_t recno = ctx.stream->readInt();
            [[maybe_unused]] int32_t t1 = ctx.stream->readInt();
            [[maybe_unused]] int32_t t2 = ctx.stream->readInt();
            [[maybe_unused]] int32_t target = ctx.stream->readInt();
            
            recordType = ctx.stream->readString(8);
            if (recordType == "EOF") {
                return false;
            }
            
            [[maybe_unused]] int32_t p1 = ctx.stream->readInt();
            [[maybe_unused]] int32_t p2 = ctx.stream->readInt();
            
            format = ctx.stream->readString(8);
            [[maybe_unused]] std::string context = ctx.stream->readString(8);
                
            [[maybe_unused]] int32_t tocrec = ctx.stream->readInt();
            int32_t datrec = ctx.stream->readInt();
            [[maybe_unused]] int32_t tocsiz = ctx.stream->readInt();
            int32_t datsiz = ctx.stream->readInt();
            
            int32_t tocoff1 = ctx.stream->readInt();
            int32_t tocoff2 = ctx.stream->readInt();
            int32_t tocoff3 = ctx.stream->readInt();
            int32_t datoff = ctx.stream->readInt();
                
            [[maybe_unused]] std::string segname = ctx.stream->readString(8);
            std::string usrnam = ctx.stream->readString(8);
            int32_t usroff = ctx.stream->readInt();
                
            [[maybe_unused]] int32_t lrecflgs = ctx.stream->readInt();
            [[maybe_unused]] int32_t spare1 = ctx.stream->readInt();
            [[maybe_unused]] int32_t spare2 = ctx.stream->readInt();
            
            // Read the event header
            if (usrnam == "IJEVHD")
            {
                if (ctx.stream->getNBytes() != usroff) {
                    throw std::runtime_error(
                        "Consistency Check Failed: 'usrOff' mismatch. Expected: " + 
                        std::to_string(usroff) + ", Actual: " + std::to_string(ctx.stream->getNBytes())
                    );
                }
                event.ieventh.read(*ctx.stream);
            }

            if (format == "MINIDST")
            {
                if (ctx.stream->getNBytes() != tocoff1) {
                    throw std::runtime_error(
                        "Consistency Check Failed: 'tocOff' mismatch. Expected: " +
                        std::to_string(tocoff1) + ", Actual: " +
                        std::to_string(ctx.stream->getNBytes())
                    );
                }

                PHMTOC toc(*ctx.stream);

                if (datrec > 0) {
                    ctx.stream->nextPhysicalRecord();
                }

                if (ctx.stream->getNBytes() != datoff) {
                    throw std::runtime_error(
                        "Consistency Check Failed: 'datOff' mismatch. Expected: " +
                        std::to_string(datoff) + ", Actual: " +
                        std::to_string(ctx.stream->getNBytes())
                    );
                }

                // Always load the buffer
                ctx.stream->readToVector(*ctx.rawDataBuffer, datsiz);
                ctx.dataBufferView->setData(
                    {ctx.rawDataBuffer->data(), static_cast<size_t>(datsiz)}
                );

                // Hand TOC back to caller if requested
                if (out_toc) *out_toc = toc;

                // Optionally skip MINIDST decoding
                if (parse_minidst) {
                    JazelleFile::parseMiniDst(toc, event, *ctx.dataBufferView);
                }
            }
            
            return true;
        }
        catch (const JazelleEOF& e)
        {
            return false;
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(
                std::string("Error during parseEvent(): ") + e.what()
            );
        }
    }

    bool JazelleFile::parseEvent(ParseContext& ctx, JazelleEvent& event)
    {
        std::string recordType;  // Ignored
        std::string format;      // Ignored
        return Impl::parseEventWithMetadata(ctx, event, recordType, format);
    }

    // Static method: MINIDST parsing logic
    void JazelleFile::parseMiniDst(const PHMTOC& toc, JazelleEvent& event,
                                   const DataBuffer& buffer,
                                   std::unordered_map<std::string, int32_t>*
                                       family_offsets)
    {
        int32_t offset = 0;

        auto& mcHeadFam  = event.get<MCHEAD>();
        auto& mcPartFam  = event.get<MCPART>();
        auto& phSumFam   = event.get<PHPSUM>();
        auto& phChrgFam  = event.get<PHCHRG>();
        auto& phKlusFam  = event.get<PHKLUS>();
        auto& phWicFam   = event.get<PHWIC>();
        auto& phCridFam  = event.get<PHCRID>();
        auto& phKtrkFam  = event.get<PHKTRK>();
        auto& phKelidFam = event.get<PHKELID>();
        auto& phPointFam = event.get<PHPOINT>();
        auto& phKChrgFam = event.get<PHKCHRG>();
        auto& phBmFam    = event.get<PHBM>();
    
        // Pre-allocate memory
        mcHeadFam.reserve(1); 
        mcPartFam.reserve(toc.m_nMcPart);
        phSumFam.reserve(toc.m_nPhPSum);
        phChrgFam.reserve(toc.m_nPhChrg);
        phKlusFam.reserve(toc.m_nPhKlus);
        phWicFam.reserve(toc.m_nPhWic);
        phCridFam.reserve(toc.m_nPhCrid);
        phKtrkFam.reserve(toc.m_nPhKTrk);
        phKelidFam.reserve(toc.m_nPhKElId);
        phPointFam.reserve(toc.m_nPhPoint);
        phKChrgFam.reserve(toc.m_nPhKChrg);
        phBmFam.reserve(toc.m_nPhBm);

        // Helper to record offsets concisely
        #define RECORD_FAMILY_OFFSET(name) \
            do { if (family_offsets) (*family_offsets)[name] = offset; } while (0)

        // Read MCHEAD
        RECORD_FAMILY_OFFSET("MCHEAD");
        MCHEAD* mchead = mcHeadFam.add(1);
        offset += mchead->read(buffer, offset, event);

        // Read MCPART
        RECORD_FAMILY_OFFSET("MCPART");
        for (int32_t i = 0; i < toc.m_nMcPart; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            MCPART* mcpart = mcPartFam.add(id);
            offset += mcpart->read(buffer, offset, event);
        }
        
        // Read PHPSUM
        RECORD_FAMILY_OFFSET("PHPSUM");
        for (int32_t i = 0; i < toc.m_nPhPSum; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            PHPSUM* phpsum = phSumFam.add(id);
            offset += phpsum->read(buffer, offset, event);
        }
        
        // Read PHCHRG
        RECORD_FAMILY_OFFSET("PHCHRG");
        for (int32_t i = 0; i < toc.m_nPhChrg; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            PHCHRG* phchrg = phChrgFam.add(id);
            offset += phchrg->read(buffer, offset, event);
        }

        // Read PHKLUS
        RECORD_FAMILY_OFFSET("PHKLUS");
        for (int32_t i = 0; i < toc.m_nPhKlus; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            PHKLUS* phklus = phKlusFam.add(id);
            offset += phklus->read(buffer, offset, event);
        }

        // Read PHWIC
        RECORD_FAMILY_OFFSET("PHWIC");
        for (int32_t i = 0; i < toc.m_nPhWic; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            PHWIC* phwic = phWicFam.add(id);
            offset += phwic->read(buffer, offset, event);
        }

        // Read PHCRID
        RECORD_FAMILY_OFFSET("PHCRID");
        for (int32_t i = 0; i < toc.m_nPhCrid; i++)
        {
            int32_t id = buffer.readInt(offset) & 0xffff;
            PHCRID* phcrid = phCridFam.add(id);
            offset += phcrid->read(buffer, offset, event);
        }

        // Read PHKTRK
        RECORD_FAMILY_OFFSET("PHKTRK");
        for (int32_t i = 0; i < toc.m_nPhKTrk; i++)
        {
            int32_t id = buffer.readInt(offset) & 0xffff;
            PHKTRK* phktrk = phKtrkFam.add(id);
            offset += phktrk->read(buffer, offset, event);
        }

        // Read PHKELID
        RECORD_FAMILY_OFFSET("PHKELID");
        for (int32_t i = 0; i < toc.m_nPhKElId; i++)
        {
            int32_t id = buffer.readInt(offset) & 0xffff;
            PHKELID* phkelid = phKelidFam.add(id);
            offset += phkelid->read(buffer, offset, event);
        }

        // Read PHPOINT
        RECORD_FAMILY_OFFSET("PHPOINT");
        for (int32_t i = 0; i < toc.m_nPhPoint; i++)
        {
            int32_t id = buffer.readInt(offset) & 0xffff;
            PHPOINT* phpoint = phPointFam.add(id);
            offset += phpoint->read(buffer, offset, event);
        }

        // ==========================================
        // Read PHKCHRG (Two-Pass Relational Table)
        RECORD_FAMILY_OFFSET("PHKCHRG");
        // Pass 1: Main Data & Track Key
        for (int32_t i = 0; i < toc.m_nPhKChrg; i++)
        {
            int32_t word = buffer.readInt(offset);
            int32_t id = (word >> 16) & 0xffff;
            PHKCHRG* phkchrg = phKChrgFam.add(id);
            offset += phkchrg->read(buffer, offset, event);
        }

        // Pass 2: Cluster Key
        for (int32_t i = 0; i < toc.m_nPhKChrg; i++)
        {
            int32_t word = buffer.readInt(offset);
            int32_t id = (word >> 16) & 0xffff;
            int32_t phklus_id = word & 0xffff;
            offset += 4;

            PHKCHRG* phkchrg = phKChrgFam.find(id);
            if (phkchrg) {
                phkchrg->phklus_id = phklus_id;
            }
        }

        // Read PHBM
        RECORD_FAMILY_OFFSET("PHBM");
        for (int32_t i = 0; i < toc.m_nPhBm; i++)
        {
            PHBM* phbm = phBmFam.add(1);
            offset += phbm->read(buffer, offset, event);
        }

        #undef RECORD_FAMILY_OFFSET
        
        // Resolve MCPART parent pointers
        size_t mcpartSize = mcPartFam.size();
        for (size_t i = 0; i < mcpartSize; ++i)
        {
            MCPART* part = mcPartFam.at(i);
            if (part && part->parent_id > 0)
            {
                part->parent = mcPartFam.find(part->parent_id);
            }
            else if (part)
            {
                part->parent = nullptr;
            }
        }
    }

    std::vector<uint8_t> JazelleFile::dumpBinary(int32_t start_offset,
                                                 int32_t end_offset) const
    {
        const int32_t buffer_size =
            static_cast<int32_t>(m_impl->dataBufferView.size());

        // Sentinel -1 (or any negative) => to end of buffer
        if (end_offset < 0) {
            end_offset = buffer_size;
        }

        // Clamp to valid range
        if (start_offset < 0)          start_offset = 0;
        if (start_offset > buffer_size) start_offset = buffer_size;
        if (end_offset   > buffer_size) end_offset   = buffer_size;
        if (end_offset <= start_offset) {
            return {};
        }

        // rawDataBuffer is the owning storage; dataBufferView is the view over
        // its first dataBufferView.size() bytes for the current event.
        const uint8_t* data = m_impl->rawDataBuffer.data();
        return std::vector<uint8_t>(data + start_offset, data + end_offset);
    }

    std::string JazelleFile::dumpBinaryText(int32_t start_offset,
                                            int32_t end_offset) const
    {
        const DataBuffer& buffer = m_impl->dataBufferView;
        const int32_t buffer_size = static_cast<int32_t>(buffer.size());

        if (end_offset < 0) {
            end_offset = buffer_size;
        }

        std::ostringstream oss;
        oss << "\n========== UNPARSED BINARY DUMP ==========\n"
            << "Buffer Total Size: " << buffer_size  << " bytes\n"
            << "Start Offset:      " << start_offset << " bytes\n"
            << "End Offset:        " << end_offset   << " bytes\n";

        const int32_t remaining_bytes = end_offset - start_offset;
        if (remaining_bytes <= 0) {
            oss << "No bytes to dump (end_offset <= start_offset).\n"
                << "==========================================\n\n";
            return oss.str();
        }

        const int32_t words_to_dump = remaining_bytes / 4; // 4-byte words

        oss << "Dumping " << remaining_bytes << " bytes ("
            << words_to_dump << " words)...\n";

        // Header (same column layout as the original)
        oss << std::left  << std::setw(10) << "Offset"
                          << std::setw(15) << "Hex (32-bit)"
                          << std::setw(15) << "Int (32-bit)"
                          << std::setw(12) << "Upper 16"
                          << std::setw(12) << "Lower 16"
                          << "Float" << '\n';
        oss << "------------------------------------------------------------------------\n";

        for (int32_t i = 0; i < words_to_dump; ++i)
        {
            const int32_t current_offset = start_offset + (i * 4);

            const int32_t raw_int   = buffer.readInt(current_offset);
            const float   raw_float = buffer.readFloat(current_offset);

            const int16_t upper_16 =
                static_cast<int16_t>((raw_int >> 16) & 0xffff);
            const int16_t lower_16 =
                static_cast<int16_t>(raw_int & 0xffff);

            // Offset
            oss << "+" << std::left << std::setfill(' ') << std::setw(8)
                << (i * 4);

            // Hex
            oss << "0x" << std::right << std::setfill('0') << std::setw(8)
                << std::hex << static_cast<uint32_t>(raw_int) << "        ";

            // Int32, upper16, lower16, float
            oss << std::left << std::setfill(' ') << std::dec
                << std::setw(15) << raw_int
                << std::setw(12) << upper_16
                << std::setw(12) << lower_16
                << raw_float << '\n';
        }
        oss << "==========================================\n\n";
        return oss.str();
    }

    void JazelleFile::printBinary(int32_t start_offset, int32_t end_offset) const
    {
        std::cout << dumpBinaryText(start_offset, end_offset);
    }


    // --- Public JazelleFile Methods ---

    JazelleFile::JazelleFile(const std::string& filepath)
    try : m_impl(std::make_unique<Impl>(filepath))
    {
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Failed to open JazelleFile '" + filepath + "': " + e.what());
    }

    JazelleFile::~JazelleFile()
    {
    }

    bool JazelleFile::nextRecord(JazelleEvent& event)
    {
        if (!m_impl->stream->nextLogicalRecord())
        {
            return false; // Clean EOF
        }
        return m_impl->readCurrentRecord(event);
    }

    // --- New Indexing Methods ---

    void JazelleFile::buildIndex()
    {
        if (m_impl->m_index_built) return;

        m_impl->stream->rewind();
        m_impl->m_event_offsets.clear();

        while (true)
        {
            if (!m_impl->stream->nextLogicalRecord())
            {
                break;
            }
            
            m_impl->m_event_offsets.push_back(m_impl->stream->getCurrentRecordOffset());
        }

        m_impl->m_index_built = true;
        m_impl->stream->rewind(); // Rewind to the start after building the index
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
            return false;
        }

        int64_t offset = m_impl->m_event_offsets[index];
        m_impl->stream->seekTo(offset);

        return m_impl->readCurrentRecord(event);
    }

    bool JazelleFile::loadEventBuffer()
    {
        if (!m_impl->stream->nextLogicalRecord())
        {
            return false;
        }
        return m_impl->readCurrentBufferOnly();
    }

    bool JazelleFile::loadEventBuffer(int32_t index)
    {
        if (!m_impl->m_index_built)
        {
            buildIndex();
        }

        if (index < 0 ||
            static_cast<size_t>(index) >= m_impl->m_event_offsets.size())
        {
            return false;
        }

        int64_t offset = m_impl->m_event_offsets[index];
        m_impl->stream->seekTo(offset);

        return m_impl->readCurrentBufferOnly();
    }

    // --- Bank family offsets ---

    std::unordered_map<std::string, int32_t>
    JazelleFile::getBankFamilyOffsets() const
    {
        if (!m_impl->lastTocValid)
        {
            throw std::runtime_error(
                "No MINIDST buffer is currently loaded. Call nextRecord(), "
                "readEvent(), or loadEventBuffer() first."
            );
        }

        std::unordered_map<std::string, int32_t> offsets;
        JazelleEvent scratch; // Discarded; only needed for read() signatures.

        try {
            JazelleFile::parseMiniDst(
                m_impl->lastToc, scratch, m_impl->dataBufferView, &offsets);
        }
        catch (...) {
            // Swallow: a malformed buffer leaves us with the offsets we
            // managed to record. Missing keys signal where the walk
            // stopped.
        }
        return offsets;
    }

    int32_t
    JazelleFile::getBankFamilyOffset(const std::string& familyName) const
    {
        // Known set of families that parseMiniDst walks.
        static const std::set<std::string> kKnownFamilies = {
            "MCHEAD", "MCPART", "MCPNT", "PHPSUM", "PHCHRG", "PHKLUS", "PHWIC",
            "PHCRID", "PHKTRK", "PHKELID", "PHPOINT", "PHKCHRG", "PHBM", "PHWMC"
        };

        if (!kKnownFamilies.count(familyName)) {
            throw std::runtime_error(
                "Unknown bank family: '" + familyName + "'."
            );
        }

        const auto offsets = getBankFamilyOffsets();
        auto it = offsets.find(familyName);
        if (it == offsets.end()) {
            throw std::runtime_error(
                "Bank family '" + familyName + "' could not be located in "
                "the current buffer (the walker stopped before reaching it, "
                "likely due to a malformed earlier family)."
            );
        }
        return it->second;
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

    std::string JazelleFile::getLastFormat() const
    {
        return m_impl->lastFormat;
    }

    void JazelleFile::rewind()
    {
        m_impl->stream->rewind();
    }

    /**
     * @brief Thread-local context for parallel reading
     * Each thread gets its own file handle and buffers
     */
    struct ThreadContext {
        std::unique_ptr<JazelleStream> stream;
        DataBuffer dataBufferView;
        std::vector<uint8_t> rawDataBuffer;
        
        explicit ThreadContext(const std::string& filepath)
        {
            stream = std::make_unique<JazelleStream>(filepath);
            rawDataBuffer.reserve(512 * 1024);
        }
        
        JazelleFile::ParseContext getParseContext() {
            return JazelleFile::ParseContext(
                stream.get(), 
                &dataBufferView, 
                &rawDataBuffer
            );
        }
        
        bool readEventAt(int64_t offset, JazelleEvent& event) {
            stream->seekTo(offset);
            auto ctx = getParseContext();
            return JazelleFile::parseEvent(ctx, event);
        }
    };
    
    void JazelleFile::readEventsParallel(
        int32_t start_idx,
        int32_t count,
        std::function<void(int32_t idx, JazelleEvent&)> callback,
        size_t num_threads
    )
    {
        if (!m_impl->m_index_built) {
            buildIndex();
        }
        
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }
        
        // Clamp to reasonable values
        if (num_threads > 32) num_threads = 32;
        
        int32_t end_idx = std::min(start_idx + count, getTotalEvents());
        if (start_idx >= end_idx) return;
        
        // Get filepath once
        const std::string filepath = m_impl->stream->getFilePath();
        
        // Validate filepath is accessible
        std::ifstream test_stream(filepath, std::ios::binary);
        if (!test_stream.is_open()) {
            throw std::runtime_error("Cannot open file for parallel reading: " + filepath);
        }
        test_stream.close();
        
        // Result queue
        using EventResult = std::pair<int32_t, std::unique_ptr<JazelleEvent>>;
        utils::MPSCQueue<EventResult> result_queue(num_threads, 16);
        
        // Object pool
        utils::ObjectPool<JazelleEvent> event_pool(num_threads * 2);
        
        // Work distribution
        std::atomic<int32_t> next_event_idx{start_idx};
        std::atomic<bool> reading_done{false};
        std::atomic<bool> error_occurred{false};
        std::exception_ptr last_exception;
        std::mutex exception_mutex;
        
        // Consumer thread
        std::thread consumer([&]() {
            try {
                while (!reading_done.load(std::memory_order_acquire) || 
                       !result_queue.allEmpty()) 
                {
                    auto result_opt = result_queue.tryPop();
                    if (result_opt.has_value()) {
                        auto& [idx, event_ptr] = result_opt.value();
                        
                        // Validate event before callback
                        if (event_ptr) {
                            callback(idx, *event_ptr);
                            event_pool.release(std::move(event_ptr));
                        }
                    } else {
                        std::this_thread::yield();
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                last_exception = std::current_exception();
                error_occurred.store(true);
            }
        });
        
        // Producer threads
        std::vector<std::thread> producers;
        producers.reserve(num_threads);
        
        for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
            producers.emplace_back([&, thread_id, filepath]() {
                try {
                    // Thread-local context
                    ThreadContext ctx(filepath);
                    
                    while (!error_occurred.load() && true) {
                        int32_t idx = next_event_idx.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= end_idx) break;
                        
                        auto event_ptr = event_pool.acquire();
                        
                        int64_t offset = m_impl->m_event_offsets[idx];
                        if (ctx.readEventAt(offset, *event_ptr)) {
                            while (!result_queue.tryPush(
                                std::make_pair(idx, std::move(event_ptr)), thread_id))
                            {
                                if (error_occurred.load()) break;
                                std::this_thread::yield();
                            }
                        }
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!last_exception) {
                        last_exception = std::current_exception();
                    }
                    error_occurred.store(true);
                }
            });
        }
        
        for (auto& producer : producers) {
            producer.join();
        }
        
        reading_done.store(true, std::memory_order_release);
        consumer.join();
        
        // Rethrow any exception
        if (last_exception) {
            std::rethrow_exception(last_exception);
        }
    }
    
    std::vector<JazelleEvent> JazelleFile::readEventsBatch(
        int32_t start_idx,
        int32_t count,
        size_t num_threads
    )
    {
        // Pre-allocate the entire vector (important!)
        std::vector<JazelleEvent> events(count);
        
        // Use atomic counter for thread safety
        std::atomic<int32_t> events_written{0};
        std::vector<std::exception_ptr> exceptions(num_threads);
        
        readEventsParallel(start_idx, count, 
            [&](int32_t idx, JazelleEvent& event) {
                try {
                    int32_t local_idx = idx - start_idx;
                    if (local_idx >= 0 && local_idx < count) {
                        // Direct assignment without mutex (pre-allocated slots)
                        events[local_idx] = std::move(event);
                        events_written.fetch_add(1, std::memory_order_relaxed);
                    }
                } catch (...) {
                    // Capture exception (thread-safe with atomic index)
                    size_t thread_idx = events_written.fetch_add(0, std::memory_order_relaxed) % num_threads;
                    exceptions[thread_idx] = std::current_exception();
                }
            },
            num_threads
        );
        
        // Rethrow any captured exceptions
        for (const auto& ex : exceptions) {
            if (ex) {
                std::rethrow_exception(ex);
            }
        }
        
        return events;
    }

} // namespace jazelle