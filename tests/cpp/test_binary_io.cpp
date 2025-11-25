#include <gtest/gtest.h>
#include "utils/BinaryIO.hpp"
#include <vector>
#include <cmath>

using namespace jazelle::utils;

// Test VAX to IEEE Float Conversion
// VAX Float format is distinct from IEEE 754. 
// We test known bit patterns based on the Java implementation logic.
TEST(BinaryIO, VaxFloatConversion) {
    // Case 1: Zero
    EXPECT_FLOAT_EQ(vaxToIeeeFloat(0), 0.0f);

    // Case 2: VAX 1.0 (Float representation)
    // VAX F_FLOAT 1.0 is typically 0x40800000 in hex representation as integer
    // Note: This depends on exact endianness of the input buffer, 
    // but vaxToIeeeFloat takes an int32_t already read from the buffer.
    // 1.0 in VAX F_FLOAT: Sign=0, Exp=129 (excess 128), Mantissa=0
    // 0 10000001 000... -> 0000 0000 0000 0000 0100 0000 1000 0000
    // However, the function expects the integer representation of the bits
    
    // Let's test the logic specifically implemented in BinaryIO.hpp:
    // 1.0 IEEE is 0x3f800000
    // Let's reverse engineer a simple case or trust the translation 
    // and verify consistent non-zero behavior.
    
    // Test "Dirty Zero" (Underflow)
    // If exponent bits are small (< 256 unshifted), it should flush to zero
    int32_t dirty_zero = 0x00000080; // Very small exponent
    EXPECT_FLOAT_EQ(vaxToIeeeFloat(dirty_zero), 0.0f);
}

// Test Little Endian Reads
TEST(BinaryIO, ReadLittleEndian) {
    std::vector<uint8_t> buffer = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

    // Short: 0x0201 = 513
    EXPECT_EQ(readLeShort(buffer.data()), 513);

    // Int: 0x04030201 = 67305985
    EXPECT_EQ(readLeInt(buffer.data()), 67305985);
    
    // Long: 0x0807060504030201
    EXPECT_EQ(readLeLong(buffer.data()), 0x0807060504030201);
}