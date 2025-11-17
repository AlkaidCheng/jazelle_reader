/**
 * @file BinaryIO.hpp
 * @brief Central utility functions for binary I/O.
 *
 * This file provides a single source of truth for all low-level
 * binary conversion logic, including platform-independent
 * little-endian reading and VAX-to-IEEE float conversion.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring> // For std::memcpy

namespace jazelle
{
namespace utils
{
    /**
     * @brief Reads a 16-bit little-endian short from a byte buffer.
     * @param b A pointer to at least 2 bytes of data.
     * @return The platform-native int16_t.
     */
    inline int16_t readLeShort(const uint8_t* b)
    {
        return static_cast<int16_t>(
            static_cast<uint16_t>(b[0]) |
            (static_cast<uint16_t>(b[1]) << 8)
        );
    }

    /**
     * @brief Reads a 32-bit little-endian integer from a byte buffer.
     * @param b A pointer to at least 4 bytes of data.
     * @return The platform-native int32_t.
     */
    inline int32_t readLeInt(const uint8_t* b)
    {
        return static_cast<int32_t>(
            static_cast<uint32_t>(b[0]) |
            (static_cast<uint32_t>(b[1]) << 8) |
            (static_cast<uint32_t>(b[2]) << 16) |
            (static_cast<uint32_t>(b[3]) << 24)
        );
    }

    /**
     * @brief Reads a 64-bit little-endian long from a byte buffer.
     * @param b A pointer to at least 8 bytes of data.
     * @return The platform-native int64_t.
     */
    inline int64_t readLeLong(const uint8_t* b)
    {
        return static_cast<int64_t>(
            static_cast<uint64_t>(b[0]) |
            (static_cast<uint64_t>(b[1]) << 8) |
            (static_cast<uint64_t>(b[2]) << 16) |
            (static_cast<uint64_t>(b[3]) << 24) |
            (static_cast<uint64_t>(b[4]) << 32) |
            (static_cast<uint64_t>(b[5]) << 40) |
            (static_cast<uint64_t>(b[6]) << 48) |
            (static_cast<uint64_t>(b[7]) << 56)
        );
    }

    /**
     * @brief Converts a 32-bit VAX F_FLOAT (as an int) to an IEEE float.
     *
     * This is a faithful translation of the original Java code's
     * bitwise operations. This specific conversion was found to be
     * correct for VAX D-float format, which may be what was
     * actually used despite the F-float name.
     *
     * @param fbits The 32-bit integer representing the VAX float.
     * @return The converted IEEE float.
     */
    inline float vaxToIeeeFloat(int32_t fbits)
    {
        if (fbits == 0) return 0.0f;

        // This is a direct, literal translation of the bit logic
        // from the original DataBuffer.java.
        int32_t sign     = fbits & 0x8000;
        int32_t exp      = fbits & 0x7f80;
        exp -= 256; // 2 << 7
        int32_t mantissa_hi = (fbits & 0x7f) << 16;
        // Java '>>' is arithmetic, so we cast to signed first.
        int32_t mantissa_lo = (fbits & 0xffff0000) >> 16; 
        int32_t mantissa = mantissa_hi + mantissa_lo; // Java used '+'
        
        int32_t bits = (sign << 16) | (exp << 16) | mantissa;
        
        float result;
        static_assert(sizeof(bits) == sizeof(result));
        std::memcpy(&result, &bits, sizeof(bits));
        return result;
    }

} // namespace utils
} // namespace jazelle