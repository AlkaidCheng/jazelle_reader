cmake_minimum_required(VERSION 3.15)
project(jazelle_reader CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and Cython
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Cython REQUIRED)

# ------------------------------------------------------------------
# 1. Build your C++ static library
# ------------------------------------------------------------------
add_library(jazelle_reader_core STATIC
    src/JazelleFile.cpp
    src/JazelleStream.cpp
    src/PHMTOC.cpp
    src/banks/CRIDHYP.cpp
    src/banks/IEVENTH.cpp
    src/banks/MCHEAD.cpp
    src/banks/MCPART.cpp
    src/banks/PHCHRG.cpp
    src/banks/PHCRID.cpp
    src/banks/PHKELID.cpp
    src/banks/PHKLUS.cpp
    src/banks/PHPSUM.cpp
    src/banks/PHWIC.cpp
    src/banks/PIDVEC.cpp
)
target_include_directories(jazelle_reader_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ------------------------------------------------------------------
# 2. Build the Cython extension module
# ------------------------------------------------------------------

# Set the path to your .pyx file
set(CYTHON_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/jazelle/jazelle_cython.pyx)

# Generate the C++ file from .pyx
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/jazelle_cython.cpp
    COMMAND Cython::Cython
        -3  # Python 3
        --cplus  # Generate C++
        -o ${CMAKE_CURRENT_BINARY_DIR}/jazelle_cython.cpp
        ${CYTHON_SOURCE}
    DEPENDS ${CYTHON_SOURCE}
    COMMENT "Cythonizing jazelle_cython.pyx"
)

# Create the Python extension module
Python_add_library(jazelle_cython MODULE
    ${CMAKE_CURRENT_BINARY_DIR}/jazelle_cython.cpp
)

# Link to your C++ library
target_link_libraries(jazelle_cython PRIVATE
    jazelle_reader_core
)

# Include directories for the Cython module
target_include_directories(jazelle_cython PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jazelle  # For .pxd files
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set the output name (without 'lib' prefix)
set_target_properties(jazelle_cython PROPERTIES
    OUTPUT_NAME "jazelle_cython"
    PREFIX ""
)

# Install the extension module
install(TARGETS jazelle_cython DESTINATION jazelle)