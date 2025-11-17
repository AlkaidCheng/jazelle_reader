/**
 * @file DataBuffer.hpp
 * @brief Internal helper class to read from a byte buffer.
 *
 * This class wraps a std::span (C++20) and provides methods to read
 * data from specific offsets. It is an internal implementation detail.
 * This replaces the previous vector-based DataBuffer and the BufferView.
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <span> // Use C++20's span for a non-owning view
#include "utils/BinaryIO.hpp" // Use the central utility

namespace jazelle
{
    // Forward-declaration
    class JazelleInputStream;

    class DataBuffer
    {
    public:
        DataBuffer() = default;

        /**
         * @brief Sets the internal view to a block of memory.
         * @param data A span of bytes to view.
         */
        void setData(std::span<const uint8_t> data)
        {
            m_span = data;
        }

        /**
         * @brief Reads a 2-byte little-endian short.
         * @param offset Byte offset to read from.
         * @return The int16_t value.
         */
        int16_t readShort(int32_t offset) const
        {
            checkBounds(offset, sizeof(int16_t));
            return utils::readLeShort(m_span.data() + offset);
        }

        /**
         * @brief Reads a 4-byte little-endian integer.
         * @param offset Byte offset to read from.
         * @return The int32_t value.
         */
        int32_t readInt(int32_t offset) const
        {
            checkBounds(offset, sizeof(int32_t));
            return utils::readLeInt(m_span.data() + offset);
        }

        /**
         * @brief Reads a 4-byte VAX F_FLOAT and converts it to IEEE float.
         * @param offset Byte offset to read from.
         * @return The float value.
         */
        float readFloat(int32_t offset) const
        {
            // readInt handles bounds checking and LE conversion
            int32_t fbits = readInt(offset);
            return utils::vaxToIeeeFloat(fbits);
        }

    private:
        /**
         * @brief Checks if a read is within the buffer bounds.
         * @throws std::out_of_range if the read is invalid.
         */
        void checkBounds(int32_t offset, size_t readSize) const
        {
            if (offset < 0 || static_cast<size_t>(offset) + readSize > m_span.size())
            {
                throw std::out_of_range(
                    "Read offset " + std::to_string(offset) +
                    " with size " + std::to_string(readSize) +
                    " is out of bounds for buffer of size " +
                    std::to_string(m_span.size())
                );
            }
        }

        /// @brief A non-owning view of the main data buffer.
        std::span<const uint8_t> m_span;
    };

} // namespace jazelle