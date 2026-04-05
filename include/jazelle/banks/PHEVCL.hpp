/**
 * @file PHEVCL.hpp
 * @brief Definition of the PHEVCL (Event Classification Summary) bank.
 *
 * @see hep.sld.jazelle.family.PHEVCL
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <array>

namespace jazelle
{
    class DataBuffer;   // Forward-declaration
    class JazelleEvent; // Forward-declaration

    /**
     * @struct PHEVCL
     * @brief Event Classification Summary bank.
     * Contains global physics triggers, the event thrust axis, and 
     * hardware status flags for the various detector subsystems.
     */
    struct PHEVCL : public Bank
    {
        // --- Bitmask Definitions ---

        struct EvtClass {
            static constexpr uint32_t BACK     = (1 << 0);  // Background (unspecified)
            static constexpr uint32_t BEAMGAS  = (1 << 1);  // Beam gas interaction
            static constexpr uint32_t COSMIC   = (1 << 2);  // Cosmic
            static constexpr uint32_t GAMGAM   = (1 << 4);  // n-Gamma candidate
            static constexpr uint32_t NUNUGAM  = (1 << 5);  // Nu-Nu-Gamma candidate
            static constexpr uint32_t ECTAUS   = (1 << 6);  // Endcap tau filter - PASS1
            static constexpr uint32_t EE       = (1 << 8);  // e+e- candidate
            static constexpr uint32_t MUMU     = (1 << 9);  // mu+mu- candidate
            static constexpr uint32_t TAUTAU   = (1 << 10); // Tau+Tau- candidate
            static constexpr uint32_t EEGG     = (1 << 12); // e+e- gamma gamma candidate
            static constexpr uint32_t TAKAL2   = (1 << 14); // Alr tracking filter - PASS2
            static constexpr uint32_t TRKHAD2  = (1 << 15); // Passed tracking hadron filter - PASS2
            static constexpr uint32_t QQ       = (1 << 16); // qqbar (unspecified quarks)
            static constexpr uint32_t LUMBHA   = (1 << 17); // Luminosity bhabha
            static constexpr uint32_t RNDMTRG  = (1 << 18); // Random trigger
            static constexpr uint32_t LASER    = (1 << 19); // CDC laser record
            static constexpr uint32_t QOMPTON  = (1 << 20); // Compton-only data
            static constexpr uint32_t TRK      = (1 << 21); // Track trigger
            static constexpr uint32_t KZ0F     = (1 << 22); // KAL Z filter - PASS1
            static constexpr uint32_t EIT      = (1 << 23); // LAC energy trigger - PASS1
            static constexpr uint32_t MUPAIR   = (1 << 24); // Passed mu-pair filter - PASS1
            static constexpr uint32_t TAUPAIR  = (1 << 25); // Passed tau-pair filter - PASS1
            static constexpr uint32_t KZ0F2    = (1 << 26); // KAL Z filter - PASS2
            static constexpr uint32_t EIT2     = (1 << 27); // LAC energy trigger - PASS2
            static constexpr uint32_t MUPAIR2  = (1 << 28); // Passed mu-pair filter - PASS2
            static constexpr uint32_t TAUPAIR2 = (1 << 29); // Passed tau-pair filter - PASS2
            static constexpr uint32_t TRGVETO  = (1 << 30); // Trigger veto bit
            static constexpr uint32_t NOINFO   = (1 << 31); // No info stored yet
        };

        struct CdcStat {
            static constexpr uint16_t CDCHV0   = (1 << 0);  // High voltage for SL #0 is off
            static constexpr uint16_t CDCHV1   = (1 << 1);  // High voltage for SL #1 is off
            static constexpr uint16_t CDCHV2   = (1 << 2);  // High voltage for SL #2 is off
            static constexpr uint16_t CDCHV3   = (1 << 3);  // High voltage for SL #3 is off
            static constexpr uint16_t CDCHV4   = (1 << 4);  // High voltage for SL #4 is off
            static constexpr uint16_t CDCHV5   = (1 << 5);  // High voltage for SL #5 is off
            static constexpr uint16_t CDCHV6   = (1 << 6);  // High voltage for SL #6 is off
            static constexpr uint16_t CDCHV7   = (1 << 7);  // High voltage for SL #7 is off
            static constexpr uint16_t CDCHV8   = (1 << 8);  // High voltage for SL #8 is off
            static constexpr uint16_t CDCHV9   = (1 << 9);  // High voltage for SL #9 is off
            static constexpr uint16_t VETO     = (1 << 15); // CDC would not have been read out
        };

        struct CrdStat {
            static constexpr uint16_t BRLOFF   = (1 << 0);  // Barrel HV is low
            static constexpr uint16_t NOVCALB  = (1 << 1);  // No barrel velocity calibration
            static constexpr uint16_t XVCALB   = (1 << 2);  // Poor barrel velocity calibration
            static constexpr uint16_t ECSOFF   = (1 << 4);  // South EC HV is low
            static constexpr uint16_t NOVCALES = (1 << 5);  // No EC S velocity calibration
            static constexpr uint16_t XVCALES  = (1 << 6);  // Poor EC S velocity calibration
            static constexpr uint16_t ECNOFF   = (1 << 8);  // North EC HV is low
            static constexpr uint16_t NOVCALEN = (1 << 9);  // No EC N velocity calibration
            static constexpr uint16_t XVCALEN  = (1 << 10); // Poor EC N velocity calibration
            static constexpr uint16_t MANGLED  = (1 << 12); // CRIDWSM data mangled or truncated
            static constexpr uint16_t LOCRvsDC = (1 << 13); // CRID/DC data size suspiciously low
            static constexpr uint16_t NOCINFO  = (1 << 15); // No information available
        };

        struct Conditi1 {
            static constexpr uint16_t BEVNECEV = (1 << 0);  // BELHDR.EVALUATR != CRIDWSM.EVALUATR
            static constexpr uint16_t BSLNECSL = (1 << 1);  // BELHDR.SLOTS != CRIDWSM.SLOTS
            static constexpr uint16_t BEVNEKEV = (1 << 2);  // BELHDR.EVALUATR != KTAG.EVALUATR
            static constexpr uint16_t BSLNEKSL = (1 << 3);  // BELHDR.SLOTS != KTAG.SLOTS
            static constexpr uint16_t VLENEGAT = (1 << 4);  // VXDRAW.LENGREQ != VXDRAW.LENGREAD
            static constexpr uint16_t DLENEGAT = (1 << 5);  // DCWSMHIT.LENGTH != DCWSMHIT.GATHERED
            static constexpr uint16_t CLENEGAT = (1 << 6);  // CRIDWSM.LENGTH != CRIDWSM.GATHERED
            static constexpr uint16_t KLENEGAT = (1 << 7);  // KTAG.LENGTH != KTAG.GATHERED
            static constexpr uint16_t BELNEVXD = (1 << 8);  // BELHDR.TAG != VXDRAW.CROSSING
            static constexpr uint16_t BELNEDC  = (1 << 9);  // BELHDR.TAG != DCWSMHIT.TAG
            static constexpr uint16_t BELNECRD = (1 << 10); // BELHDR.TAG != CRDWSM.FBHEAD.TAG
            static constexpr uint16_t BELNEKTG = (1 << 11); // BELHDR.TAG != KTAG.HEADER.TAG
            static constexpr uint16_t EVREPEAT = (1 << 12); // Event repeat - same event twice in a row
            static constexpr uint16_t EVREPSEQ = (1 << 13); // Event repeat - separated by >=1 events
            static constexpr uint16_t BCNOTINC = (1 << 14); // BCNUMS not monotonically increasing
        };

        struct Conditi2 {
            static constexpr uint16_t NOBEL    = (1 << 0);  // No BELHDR
            static constexpr uint16_t BINCONST = (1 << 1);  // BELHDR.CONTRACT != CONTRIBU
            static constexpr uint16_t BELERROR = (1 << 2);  // BELHDR.ERR != 0
            static constexpr uint16_t BMISSVXD = (1 << 4);  // BELHDR.CONTRACT=VXDRAW & no VXDRAW
            static constexpr uint16_t BMISSDC  = (1 << 5);  // BELHDR.CONTRACT=DCWSMHIT & NO DCWSMHIT
            static constexpr uint16_t BMISSCRD = (1 << 6);  // BELHDR.CONTRACT=CRIDWSM & NO CRIDWSM
            static constexpr uint16_t BMISSKAL = (1 << 7);  // BELHDR.CONTRACT=KTAG & NO KTAG
            static constexpr uint16_t BMISSWIC = (1 << 8);  // BELHDR.CONTRACT=WCHS & no WCHS
            static constexpr uint16_t BELSLOT0 = (1 << 9);  // BELHDR.SLOTS = 0
            static constexpr uint16_t BELEVAL0 = (1 << 10); // BELHDR.EVALUATR = 0
            static constexpr uint16_t BSLINCEV = (1 << 11); // SLOTS inconsistent with EVALUATOR
            static constexpr uint16_t BEVNEVEV = (1 << 12); // BELHDR.EVALUATR != VXDRAW.EVALUATR
            static constexpr uint16_t BSLNEVSL = (1 << 13); // BELHDR.SLOTS != VXDRAW.SLOTS
            static constexpr uint16_t BEVNEDEV = (1 << 14); // BELHDR.EVALUATR != DCWSMHIT.EVALUATR
            static constexpr uint16_t BSLNEDSL = (1 << 15); // BELHDR.SLOTS != DCWSMHIT.SLOTS
        };

        // --- Member Variables ---
        uint32_t             evtclass; // Global event classification bitmask
        std::array<float, 3> thrust;   // 3-Vector Thrust axis used by ZXFIND
        uint16_t             vxdstat;  // Vertex Detector status
        uint16_t             cdcstat;  // Central Drift Chamber status
        uint16_t             kalstat;  // Liquid Argon Calorimeter status
        uint16_t             crdstat;  // CRID status
        uint16_t             wicstat;  // Warm Iron Calorimeter status
        uint16_t             conditi1; // CONDITION output from STRGCHK (Part 1)
        uint16_t             conditi2; // CONDITION output from STRGCHK (Part 2)

        /**
         * @brief Constructor.
         * @param id The bank's unique ID.
         */
        explicit PHEVCL(int32_t id) : Bank(id) {}

        /**
         * @brief Reads the bank's data from the DataBuffer.
         * @return The number of bytes read (30).
         */
        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;

        /**
         * @brief Helper method to check if this is a Hadronic (Z -> qq) event.
         */
        bool isHadronic() const
        {
            return (evtclass & EvtClass::QQ) != 0;
        }
    };

} // namespace jazelle