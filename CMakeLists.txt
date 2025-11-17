# CMakeLists.txt (in jazelle-reader/ root)

cmake_minimum_required(VERSION 3.15)
project(jazelle_reader CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
#  1. Build your C++ static library (as before)
# ------------------------------------------------------------------
# I recommend renaming it to avoid conflicts with the Python module
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

find_package(Cython REQUIRED)

# ------------------------------------------------------------------
#  3. Build the Cython Extension
# ------------------------------------------------------------------
# This tells scikit-build-core to create the Python extension module
# It will be named "jazelle_reader"
cython_add_module(_core
    SRC
    src/jazelle/jazelle_cython.pyx
)

# Link the Cython module to your C++ static library
target_link_libraries(_core PRIVATE
    Python::Module
    jazelle_reader_core
)

# Tell CMake where to find your .pxd and .hpp files
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/jazelle # For the .pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ------------------------------------------------------------------
#  4. Install the C++ headers (Optional, but good practice)
# ------------------------------------------------------------------
# This isn't needed for pip install, but is good for packaging
install(
    DIRECTORY include/jazelle
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)