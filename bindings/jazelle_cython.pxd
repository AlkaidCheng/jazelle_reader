# python/jazelle_cython.pxd
#
# This is the "header" file for Cython. It declares the C++
# API that we want to access.

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libc.stdint cimport int16_t, int32_t, int64_t
from libcpp.chrono cimport system_clock, time_point

cdef extern from "jazelle/Family.hpp" namespace "jazelle":
    
    cdef cppclass CppIFamily "jazelle::IFamily":
        size_t size()
        string name()
        CppBank* at(size_t index)
        CppBank* find(int32_t id)

    cdef cppclass CppFamily "jazelle::Family"[T](CppIFamily):
        T* at(size_t index)
        T* find(int32_t id)
        T* add(int32_t id)
        void clear()

# --- Utility Structs (from banks/*.hpp) ---

cdef extern from "jazelle/banks/PIDVEC.hpp" namespace "jazelle":
    cdef cppclass CppPIDVEC "jazelle::PIDVEC":
        CppPIDVEC()
        float e, mu, pi, k, p

cdef extern from "jazelle/banks/CRIDHYP.hpp" namespace "jazelle":
    cdef cppclass CppCRIDHYP "jazelle::CRIDHYP":
        CppCRIDHYP()
        bint m_full
        int16_t rc, nhits
        int32_t besthyp
        int16_t nhexp, nhfnd, nhbkg, mskphot
        optional[CppPIDVEC] llik

# --- Bank Structs (from banks/*.hpp) ---
# We must declare all bank structs we want to wrap.

cdef extern from "jazelle/Bank.hpp" namespace "jazelle":
    cdef cppclass CppBank "jazelle::Bank":
        CppBank(int32_t)
        int32_t getId()

cdef extern from "jazelle/banks/IEVENTH.hpp" namespace "jazelle":
    cdef cppclass CppIEVENTH "jazelle::IEVENTH"(CppBank):
        CppIEVENTH()
        int32_t header, run, event, evttype, trigger
        float weight
        system_clock.time_point evttime

cdef extern from "jazelle/banks/MCHEAD.hpp" namespace "jazelle":
    cdef cppclass CppMCHEAD "jazelle::MCHEAD"(CppBank):
        CppMCHEAD(int32_t)
        int32_t ntot, origin
        float ipx, ipy, ipz

cdef extern from "jazelle/banks/MCPART.hpp" namespace "jazelle":
    cdef cppclass CppMCPART "jazelle::MCPART"(CppBank):
        CppMCPART(int32_t)
        float e, ptot, charge
        int32_t ptype, origin, parent_id
        float p[3]
        float xt[3]

cdef extern from "jazelle/banks/PHPSUM.hpp" namespace "jazelle":
    cdef cppclass CppPHPSUM "jazelle::PHPSUM"(CppBank):
        CppPHPSUM(int32_t)
        float px, py, pz, x, y, z, charge
        int32_t status
        double getPTot()

cdef extern from "jazelle/banks/PHCHRG.hpp" namespace "jazelle":
    cdef cppclass CppPHCHRG "jazelle::PHCHRG"(CppBank):
        CppPHCHRG(int32_t)
        float hlxpar[6]
        float dhlxpar[15]
        float bnorm, impact, b3norm, impact3
        int16_t charge, smwstat
        int32_t status
        float tkpar0
        float tkpar[5]
        float dtkpar[15]
        float length, chi2dt
        int16_t imc, ndfdt, nhit, nhite, nhitp, nmisht, nwrght, nhitv
        float chi2, chi2v
        int32_t vxdhit
        int16_t mustat, estat
        int32_t dedx

cdef extern from "jazelle/banks/PHKLUS.hpp" namespace "jazelle":
    cdef cppclass CppPHKLUS "jazelle::PHKLUS"(CppBank):
        CppPHKLUS(int32_t)
        int32_t status, nhit2, nhit3
        float eraw, cth, wcth, phi, wphi
        float elayer[8]
        float cth2, wcth2, phi2, whphi2
        float cth3, wcth3, phi3, wphi3

cdef extern from "jazelle/banks/PHWIC.hpp" namespace "jazelle":
    cdef cppclass CppPHWIC "jazelle::PHWIC"(CppBank):
        CppPHWIC(int32_t)
        int16_t idstat, nhit, nhit45, npat, nhitpat, syshit
        float qpinit, t1, t2, t3
        int32_t hitmiss
        float itrlen
        int16_t nlayexp, nlaybey
        float missprob
        int32_t phwicid, hitsused
        int16_t nhitshar, nother
        float pref1[3]
        float pfit[4]
        float dpfit[10]
        float chi2
        int16_t ndf, punfit
        float matchChi2
        int16_t matchNdf

cdef extern from "jazelle/banks/PHCRID.hpp" namespace "jazelle":
    cdef cppclass CppPHCRID "jazelle::PHCRID"(CppBank):
        CppPHCRID(int32_t)
        int32_t ctlword
        float norm
        int16_t rc, geom, trkp, nhits
        CppCRIDHYP liq, gas
        CppPIDVEC llik

cdef extern from "jazelle/banks/PHKTRK.hpp" namespace "jazelle":
    cdef cppclass CppPHKTRK "jazelle::PHKTRK"(CppBank):
        CppPHKTRK(int32_t)
        pass # Stub bank

cdef extern from "jazelle/banks/PHKELID.hpp" namespace "jazelle":
    cdef cppclass CppPHKELID "jazelle::PHKELID"(CppBank):
        CppPHKELID(int32_t)
        CppPHCHRG* phchrg # The linked bank
        int16_t idstat, prob
        float phi, theta, qp, dphi, dtheta, dqp
        float tphi, ttheta, isolat, em1, em12, dem12, had1
        float emphi, emtheta, phiwid, thewid
        float em1x1, em2x2a, em2x2b, em3x3a, em3x3b

# --- Main Event Container ---

cdef extern from "jazelle/JazelleEvent.hpp" namespace "jazelle":
    cdef cppclass CppJazelleEvent "jazelle::JazelleEvent":
        CppJazelleEvent() except +
        void clear()
        CppIEVENTH ieventh

        CppFamily[T]& get[T]()      

        vector[pair[string, CppIFamily*]] getFamilies()

        CppBank* add(string, int32_t) except +

        CppIFamily* getFamily(string) except +

        static vector[string] getKnownBankNames()        

# --- Main File Reader ---

cdef extern from "jazelle/JazelleFile.hpp" namespace "jazelle":
    cdef cppclass CppJazelleFile "jazelle::JazelleFile":  # Note the quoted name
        CppJazelleFile(string& filepath) except +
        bint nextRecord(CppJazelleEvent& event) except +
        bint readEvent(int32_t index, CppJazelleEvent& event) except +
        int32_t getTotalEvents() except +
        string getFileName()
        system_clock.time_point getCreationDate()
        system_clock.time_point getModifiedDate()        
        string getLastRecordType()