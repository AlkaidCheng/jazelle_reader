/**
 * @file AllBanks.hpp
 * @brief A convenience header to include all concrete bank types.
 *
 * This is used by JazelleEvent.hpp to define its Family members.
 */

#pragma once

// Include all helper structs first
#include "PIDVEC.hpp"
#include "CRIDHYP.hpp"

// Include all concrete bank structs
#include "IEVENTH.hpp"
#include "MCHEAD.hpp"
#include "MCPART.hpp"
#include "PHPSUM.hpp"
#include "PHCHRG.hpp"
#include "PHKLUS.hpp"
#include "PHWIC.hpp"
#include "PHCRID.hpp"
#include "PHKTRK.hpp"
#include "PHKELID.hpp"

// ... Add any other banks referenced in PHMTOC.hpp here ...
// (We have covered all banks referenced in MINIDST.java)