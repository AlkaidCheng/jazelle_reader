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
#include <iostream>
#include <optional>
#include <vector>
#include <thread>

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
            
            // We need to capture metadata during parsing
            std::string recordType;
            std::string format;
            
            bool success = parseEventWithMetadata(ctx, event, recordType, format);
            
            if (success) {
                lastRecordType = recordType;
                lastFormat = format;
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
            std::string& format);

    };

    // Static method: Core parsing logic
    /**
     * @brief Helper that captures metadata during parsing
     */
    bool JazelleFile::Impl::parseEventWithMetadata(
        JazelleFile::ParseContext& ctx, 
        JazelleEvent& event,
        std::string& recordType,
        std::string& format)
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
            
            [[maybe_unused]] int32_t tocoff1 = ctx.stream->readInt();
            [[maybe_unused]] int32_t tocoff2 = ctx.stream->readInt();
            [[maybe_unused]] int32_t tocoff3 = ctx.stream->readInt();
            [[maybe_unused]] int32_t datoff = ctx.stream->readInt();
                
            [[maybe_unused]] std::string segname = ctx.stream->readString(8);
            std::string usrnam = ctx.stream->readString(8);
            [[maybe_unused]] int32_t usroff = ctx.stream->readInt();
                
            [[maybe_unused]] int32_t lrecflgs = ctx.stream->readInt();
            [[maybe_unused]] int32_t spare1 = ctx.stream->readInt();
            [[maybe_unused]] int32_t spare2 = ctx.stream->readInt();
                
            // Read the event header
            if (usrnam == "IJEVHD")
            {
                event.ieventh.read(*ctx.stream);
            }

            if (format == "MINIDST")
            {
                PHMTOC toc(*ctx.stream);
                
                if (datrec > 0)
                {
                    ctx.stream->nextPhysicalRecord();
                }

                // Read the entire data record
                ctx.stream->readToVector(*ctx.rawDataBuffer, datsiz);
                ctx.dataBufferView->setData(
                    {ctx.rawDataBuffer->data(), static_cast<size_t>(datsiz)}
                );
                
                // Parse the data buffer
                JazelleFile::parseMiniDst(toc, event, *ctx.dataBufferView);
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
                                   const DataBuffer& buffer)
    {
        int32_t offset = 0;
    
        auto& mcHeadFam = event.get<MCHEAD>();
        auto& mcPartFam = event.get<MCPART>();
        auto& phSumFam  = event.get<PHPSUM>();
        auto& phChrgFam = event.get<PHCHRG>();
        auto& phKlusFam = event.get<PHKLUS>();
        auto& phWicFam  = event.get<PHWIC>();
        auto& phCridFam = event.get<PHCRID>();
        auto& phKtrkFam = event.get<PHKTRK>();
        auto& phKelidFam= event.get<PHKELID>();
    
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
    
        // Read MCHead
        MCHEAD* mchead = mcHeadFam.add(1);
        offset += mchead->read(buffer, offset, event);
        
        // Read MCPart
        for (int32_t i = 0; i < toc.m_nMcPart; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            MCPART* mcpart = mcPartFam.add(id);
            offset += mcpart->read(buffer, offset, event);
        }
    
        // Read PHPSUM
        for (int32_t i = 0; i < toc.m_nPhPSum; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            PHPSUM* phpsum = phSumFam.add(id);
            offset += phpsum->read(buffer, offset, event);
        }
    
        // Read PHCHRG
        for (int32_t i = 0; i < toc.m_nPhChrg; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            PHCHRG* phchrg = phChrgFam.add(id);
            offset += phchrg->read(buffer, offset, event);
        }
    
        // Read PHKLUS
        for (int32_t i = 0; i < toc.m_nPhKlus; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            PHKLUS* phklus = phKlusFam.add(id);
            offset += phklus->read(buffer, offset, event);
        }
    
        // Read PHWIC
        for (int32_t i = 0; i < toc.m_nPhWic; i++)
        {
            int32_t id = buffer.readInt(offset);
            offset += 4;
            PHWIC* phwic = phWicFam.add(id);
            offset += phwic->read(buffer, offset, event);
        }
    
        // Read PHCRID
        for (int32_t i = 0; i < toc.m_nPhCrid; i++)
        {
            int32_t id = buffer.readInt(offset) & 0xffff;
            PHCRID* phcrid = phCridFam.add(id);
            offset += phcrid->read(buffer, offset, event);
        }
    
        // Read PHKTRK
        for (int32_t i = 0; i < toc.m_nPhKTrk; i++)
        {
            int32_t id = buffer.readInt(offset) & 0xffff;
            PHKTRK* phktrk = phKtrkFam.add(id);
            offset += phktrk->read(buffer, offset, event);
        }
    
        // Read PHKELID
        for (int32_t i = 0; i < toc.m_nPhKElId; i++)
        {
            int32_t id = buffer.readInt(offset) & 0xffff;
            PHKELID* phkelid = phKelidFam.add(id);
            offset += phkelid->read(buffer, offset, event);
        }
    
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
        
        // Physical total
        int32_t physical_total = m_impl->m_event_offsets.size();
        int32_t end_idx = std::min(start_idx + count, physical_total);
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