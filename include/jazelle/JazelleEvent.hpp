/**
 * @file JazelleEvent.hpp
 * @brief The main event data store, replacing the Jazelle.java singleton.
 *
 * This class, per your suggestion, is named JazelleEvent. It acts as
 * the container for all bank families for a single event.
 *
 * @see hep.sld.jazelle.Jazelle
 */

#pragma once

#include "Family.hpp"
#include "banks/AllBanks.hpp"
#include <string>
#include <stdexcept>

namespace jazelle
{
    /**
     * @class JazelleEvent
     * @brief Holds all bank data for a single processed event.
     *
     * This class replaces the singleton `Jazelle.java`. An instance of
     * this class is created for each record and passed to
     * `JazelleFile::nextRecord()` to be populated.
     */
    class JazelleEvent
    {
    public:
        // --- Member Variables ---
        // Public for easy access by user code and Cython.

        /// The event header, read directly from the stream.
        IEVENTH ieventh;

        // A Family manager for each bank type defined in MINIDST.java
        Family<MCHEAD>   mcheadFamily;
        Family<MCPART>   mcpartFamily;
        Family<PHPSUM>   phpsumFamily;
        Family<PHCHRG>   phchrgFamily;
        Family<PHKLUS>   phklusFamily;
        Family<PHWIC>    phwicFamily;
        Family<PHCRID>   phcridFamily;
        Family<PHKTRK>   phktrkFamily;
        Family<PHKELID>  phkelidFamily;
        // ... Add other families as needed ...

        /**
         * @brief Default constructor.
         */
        JazelleEvent() = default;

        /**
         * @brief Clears all data from all bank families.
         * Called by JazelleFile before reading the next record.
         * @see hep.sld.jazelle.Jazelle#clear()
         */
        void clear()
        {
            // Note: ieventh is overwritten, not cleared
            mcheadFamily.clear();
            mcpartFamily.clear();
            phpsumFamily.clear();
            phchrgFamily.clear();
            phklusFamily.clear();
            phwicFamily.clear();
            phcridFamily.clear();
            phktrkFamily.clear();
            phkelidFamily.clear();
        }

        /**
         * @brief String-based factory to add a bank by name.
         *
         * This replicates the logic from `Jazelle.java::family(String)`
         * and is used by the MINIDST parser.
         *
         * @param familyName The name of the family (e.g., "PHCHRG").
         * @param id The ID for the new bank.
         * @return A base pointer to the newly created bank.
         * @see hep.sld.jazelle.Jazelle#add(String, int)
         */
        Bank* add(const std::string& familyName, int32_t id)
        {
            if (familyName == "MCHEAD")   return mcheadFamily.add(id);
            if (familyName == "MCPART")   return mcpartFamily.add(id);
            if (familyName == "PHPSUM")   return phpsumFamily.add(id);
            if (familyName == "PHCHRG")   return phchrgFamily.add(id);
            if (familyName == "PHKLUS")   return phklusFamily.add(id);
            if (familyName == "PHWIC")    return phwicFamily.add(id);
            if (familyName == "PHCRID")   return phcridFamily.add(id);
            if (familyName == "PHKTRK")   return phktrkFamily.add(id);
            if (familyName == "PHKELID")  return phkelidFamily.add(id);
            
            // This translates the ClassNotFoundException
            throw std::runtime_error("Unknown bank family name: " + familyName);
        }

        // --- Convenience Finders ---
        // Provides a type-safe, user-friendly API.

        MCHEAD* findMCHEAD  (int32_t id) { return mcheadFamily.find(id);   }
        MCPART* findMCPART  (int32_t id) { return mcpartFamily.find(id);   }
        PHPSUM* findPHPSUM  (int32_t id) { return phpsumFamily.find(id);   }
        PHCHRG* findPHCHRG  (int32_t id) { return phchrgFamily.find(id);   }
        PHKLUS* findPHKLUS  (int32_t id) { return phklusFamily.find(id);   }
        PHWIC* findPHWIC   (int32_t id) { return phwicFamily.find(id);    }
        PHCRID* findPHCRID  (int32_t id) { return phcridFamily.find(id);   }
        PHKTRK* findPHKTRK  (int32_t id) { return phktrkFamily.find(id);   }
        PHKELID* findPHKELID (int32_t id) { return phkelidFamily.find(id);  }
    };

} // namespace jazelle