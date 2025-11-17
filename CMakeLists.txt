# CMakeLists.txt (in jazelle-reader/ root)

cmake_minimum_required(VERSION 3.15)
project(jazelle_reader CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
#  Find Python - required for building Python extensions
# ------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# ------------------------------------------------------------------
#  1. Build your C++ static library (as before)
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
target_include_directories(jazelle_reader_core PUBLIC include src)

# ------------------------------------------------------------------
#  2. Create the Python extension module
# ------------------------------------------------------------------
Python_add_library(_core MODULE
    src/jazelle/jazelle_cython.cpp
)

# Set the output name to match what Python expects
set_target_properties(_core PROPERTIES
    OUTPUT_NAME "_core"
    PREFIX ""  # No 'lib' prefix on Unix
)

# ------------------------------------------------------------------
#  3. Link the Cython module to your C++ static library
# ------------------------------------------------------------------
target_link_libraries(_core PRIVATE
    jazelle_reader_core
)

# Tell CMake where to find your .pxd and .hpp files
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jazelle
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ------------------------------------------------------------------
#  4. Install the extension module to the jazelle package
# ------------------------------------------------------------------
install(TARGETS _core
        LIBRARY DESTINATION jazelle
        RUNTIME DESTINATION jazelle)