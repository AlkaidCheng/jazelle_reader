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
#  Cythonize the .pyx file
# ------------------------------------------------------------------
set(CYTHON_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/jazelle/jazelle_cython.pyx)
set(CYTHON_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/jazelle_cython.cpp)

# Use Python to run Cython
add_custom_command(
    OUTPUT ${CYTHON_OUTPUT}
    COMMAND ${Python_EXECUTABLE} -m cython 
            --cplus 
            -3
            -I ${CMAKE_CURRENT_SOURCE_DIR}/src/jazelle
            ${CYTHON_SOURCE}
            -o ${CYTHON_OUTPUT}
    DEPENDS ${CYTHON_SOURCE}
    COMMENT "Cythonizing ${CYTHON_SOURCE}"
    VERBATIM
)

# ------------------------------------------------------------------
#  1. Build your C++ static library
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

set_target_properties(jazelle_reader_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------
#  2. Create the Python extension module
# ------------------------------------------------------------------
Python_add_library(jazelle_cython MODULE
    ${CYTHON_OUTPUT}
)

# Set the output name to match what Python expects
set_target_properties(jazelle_cython PROPERTIES
    OUTPUT_NAME "jazelle_cython"
    PREFIX ""  # No 'lib' prefix on Unix
)

# ------------------------------------------------------------------
#  3. Link the Cython module to your C++ static library
# ------------------------------------------------------------------
target_link_libraries(jazelle_cython PRIVATE
    jazelle_reader_core
)

# Tell CMake where to find your .pxd and .hpp files
target_include_directories(jazelle_cython PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jazelle
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Python_INCLUDE_DIRS}
)

# ------------------------------------------------------------------
#  4. Install the extension module to the jazelle package
# ------------------------------------------------------------------
install(TARGETS jazelle_cython
        LIBRARY DESTINATION jazelle
        RUNTIME DESTINATION jazelle)