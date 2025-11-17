# python/jazelle_cython.pyx
#
# This is the "implementation" file. It defines the Python-facing
# 'cdef class' wrappers.

import cython
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair
import datetime

from libc.stdint cimport int16_t, int32_t
from libcpp.chrono cimport system_clock, to_time_t

cimport jazelle_cython as pxd

# Then your function becomes:
cdef object cpp_to_py_time(system_clock.time_point tp):
    cdef long time_t_val
    
    try:
        time_t_val = to_time_t(tp)  # Call it directly, not as system_clock.to_time_t
        return datetime.datetime.fromtimestamp(float(time_t_val))
    except:
        return None

#
# --- Helper: Bank Wrapper Base Class ---
#
cdef class PyBank:
    cdef pxd.Bank* _ptr
    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.id}>"
    
    @property
    def id(self):
        return self._ptr.getId()

#
# --- Helper: Struct Wrappers (for PIDVEC, CRIDHYP) ---
#
cdef class PyPIDVEC:
    cdef pxd.PIDVEC* _ptr
    def __init__(self):
        raise TypeError("Cannot instantiate PyPIDVEC directly.")
    def __repr__(self):
        return (f"<PIDVEC e={self.e:.3f}, mu={self.mu:.3f}, pi={self.pi:.3f}, "
                f"k={self.k:.3f}, p={self.p:.3f}>")
    
    @property
    def e(self): return self._ptr.e
    @property
    def mu(self): return self._ptr.mu
    @property
    def pi(self): return self._ptr.pi
    @property
    def k(self): return self._ptr.k
    @property
    def p(self): return self._ptr.p

cdef class PyCRIDHYP:
    cdef pxd.CRIDHYP* _ptr
    def __init__(self):
        raise TypeError("Cannot instantiate PyCRIDHYP directly.")
        
    @property
    def is_full(self): return self._ptr.m_full
    @property
    def rc(self): return self._ptr.rc
    @property
    def nhits(self): return self._ptr.nhits
    @property
    def besthyp(self): return self._ptr.besthyp
    @property
    def nhexp(self): return self._ptr.nhexp
    @property
    def nhfnd(self): return self._ptr.nhfnd
    @property
    def nhbkg(self): return self._ptr.nhbkg
    @property
    def mskphot(self): return self._ptr.mskphot
    
    @property
    def llik(self):
        if not self._ptr.llik.has_value():
            return None
        return wrap_pidvec(cython.address(self._ptr.llik.value()))

#
# --- Bank Wrapper Factory Functions ---
#
cdef object wrap_pidvec(pxd.PIDVEC* ptr):
    cdef PyPIDVEC py_obj = <PyPIDVEC>PyPIDVEC.__new__(PyPIDVEC)
    py_obj._ptr = ptr
    return py_obj

cdef object wrap_cridhyp(pxd.CRIDHYP* ptr):
    cdef PyCRIDHYP py_obj = <PyCRIDHYP>PyCRIDHYP.__new__(PyCRIDHYP)
    py_obj._ptr = ptr
    return py_obj

#
# --- Bank Wrappers (The "Tedious" Part) ---
#

# --- IEVENTH ---
cdef class PyIEVENTH(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyIEVENTH directly.")
    
    @property
    def run(self): return (<pxd.IEVENTH*>self._ptr).run
    @property
    def event(self): return (<pxd.IEVENTH*>self._ptr).event
    @property
    def evttype(self): return (<pxd.IEVENTH*>self._ptr).evttype
    @property
    def trigger(self): return (<pxd.IEVENTH*>self._ptr).trigger
    @property
    def weight(self): return (<pxd.IEVENTH*>self._ptr).weight
    @property
    def evttime(self):
        return cpp_to_py_time((<pxd.IEVENTH*>self._ptr).evttime)

cdef object wrap_ieventh(pxd.IEVENTH* ptr):
    cdef PyIEVENTH py_obj = <PyIEVENTH>PyIEVENTH.__new__(PyIEVENTH)
    py_obj._ptr = ptr
    return py_obj

# --- MCHEAD ---
cdef class PyMCHEAD(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyMCHEAD directly.")
    
    @property
    def ntot(self): return (<pxd.MCHEAD*>self._ptr).ntot
    @property
    def origin(self): return (<pxd.MCHEAD*>self._ptr).origin
    @property
    def ipx(self): return (<pxd.MCHEAD*>self._ptr).ipx
    @property
    def ipy(self): return (<pxd.MCHEAD*>self._ptr).ipy
    @property
    def ipz(self): return (<pxd.MCHEAD*>self._ptr).ipz

cdef object wrap_mchead(pxd.MCHEAD* ptr):
    cdef PyMCHEAD py_obj = <PyMCHEAD>PyMCHEAD.__new__(PyMCHEAD)
    py_obj._ptr = ptr
    return py_obj

# --- MCPART ---
cdef class PyMCPART(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyMCPART directly.")
    
    @property
    def e(self): return (<pxd.MCPART*>self._ptr).e
    @property
    def ptot(self): return (<pxd.MCPART*>self._ptr).ptot
    @property
    def ptype(self): return (<pxd.MCPART*>self._ptr).ptype
    @property
    def charge(self): return (<pxd.MCPART*>self._ptr).charge
    @property
    def origin(self): return (<pxd.MCPART*>self._ptr).origin
    @property
    def parent_id(self): return (<pxd.MCPART*>self._ptr).parent_id
    @property
    def p(self):
        cdef pxd.MCPART* p_obj = <pxd.MCPART*>self._ptr
        return [p_obj.p[i] for i in range(3)]
    @property
    def xt(self):
        cdef pxd.MCPART* p_obj = <pxd.MCPART*>self._ptr
        return [p_obj.xt[i] for i in range(3)]

cdef object wrap_mcpart(pxd.MCPART* ptr):
    cdef PyMCPART py_obj = <PyMCPART>PyMCPART.__new__(PyMCPART)
    py_obj._ptr = ptr
    return py_obj

# --- PHPSUM ---
cdef class PyPHPSUM(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyPHPSUM directly.")

    @property
    def px(self): return (<pxd.PHPSUM*>self._ptr).px
    @property
    def py(self): return (<pxd.PHPSUM*>self._ptr).py
    @property
    def pz(self): return (<pxd.PHPSUM*>self._ptr).pz
    @property
    def x(self): return (<pxd.PHPSUM*>self._ptr).x
    @property
    def y(self): return (<pxd.PHPSUM*>self._ptr).y
    @property
    def z(self): return (<pxd.PHPSUM*>self._ptr).z
    @property
    def charge(self): return (<pxd.PHPSUM*>self._ptr).charge
    @property
    def status(self): return (<pxd.PHPSUM*>self._ptr).status
    
    def getPTot(self):
        return (<pxd.PHPSUM*>self._ptr).getPTot()

cdef object wrap_phpsum(pxd.PHPSUM* ptr):
    cdef PyPHPSUM py_obj = <PyPHPSUM>PyPHPSUM.__new__(PyPHPSUM)
    py_obj._ptr = ptr
    return py_obj

# --- PHCHRG ---
cdef class PyPHCHRG(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyPHCHRG directly.")

    @property
    def bnorm(self): return (<pxd.PHCHRG*>self._ptr).bnorm
    @property
    def impact(self): return (<pxd.PHCHRG*>self._ptr).impact
    @property
    def b3norm(self): return (<pxd.PHCHRG*>self._ptr).b3norm
    @property
    def impact3(self): return (<pxd.PHCHRG*>self._ptr).impact3
    @property
    def charge(self): return (<pxd.PHCHRG*>self._ptr).charge
    @property
    def smwstat(self): return (<pxd.PHCHRG*>self._ptr).smwstat
    @property
    def status(self): return (<pxd.PHCHRG*>self._ptr).status
    @property
    def tkpar0(self): return (<pxd.PHCHRG*>self._ptr).tkpar0
    @property
    def length(self): return (<pxd.PHCHRG*>self._ptr).length
    @property
    def chi2dt(self): return (<pxd.PHCHRG*>self._ptr).chi2dt
    @property
    def imc(self): return (<pxd.PHCHRG*>self._ptr).imc
    @property
    def ndfdt(self): return (<pxd.PHCHRG*>self._ptr).ndfdt
    @property
    def nhit(self): return (<pxd.PHCHRG*>self._ptr).nhit
    @property
    def nhite(self): return (<pxd.PHCHRG*>self._ptr).nhite
    @property
    def nhitp(self): return (<pxd.PHCHRG*>self._ptr).nhitp
    @property
    def nmisht(self): return (<pxd.PHCHRG*>self._ptr).nmisht
    @property
    def nwrght(self): return (<pxd.PHCHRG*>self._ptr).nwrght
    @property
    def nhitv(self): return (<pxd.PHCHRG*>self._ptr).nhitv
    @property
    def chi2(self): return (<pxd.PHCHRG*>self._ptr).chi2
    @property
    def chi2v(self): return (<pxd.PHCHRG*>self._ptr).chi2v
    @property
    def vxdhit(self): return (<pxd.PHCHRG*>self._ptr).vxdhit
    @property
    def mustat(self): return (<pxd.PHCHRG*>self._ptr).mustat
    @property
    def estat(self): return (<pxd.PHCHRG*>self._ptr).estat
    @property
    def dedx(self): return (<pxd.PHCHRG*>self._ptr).dedx
        
    @property
    def hlxpar(self):
        cdef pxd.PHCHRG* p_obj = <pxd.PHCHRG*>self._ptr
        return [p_obj.hlxpar[i] for i in range(6)]
    @property
    def dhlxpar(self):
        cdef pxd.PHCHRG* p_obj = <pxd.PHCHRG*>self._ptr
        return [p_obj.dhlxpar[i] for i in range(15)]
    @property
    def tkpar(self):
        cdef pxd.PHCHRG* p_obj = <pxd.PHCHRG*>self._ptr
        return [p_obj.tkpar[i] for i in range(5)]
    @property
    def dtkpar(self):
        cdef pxd.PHCHRG* p_obj = <pxd.PHCHRG*>self._ptr
        return [p_obj.dtkpar[i] for i in range(15)]

cdef object wrap_phchrg(pxd.PHCHRG* ptr):
    cdef PyPHCHRG py_obj = <PyPHCHRG>PyPHCHRG.__new__(PyPHCHRG)
    py_obj._ptr = ptr
    return py_obj

# --- PHKLUS ---
cdef class PyPHKLUS(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyPHKLUS directly.")

    @property
    def status(self): return (<pxd.PHKLUS*>self._ptr).status
    @property
    def eraw(self): return (<pxd.PHKLUS*>self._ptr).eraw
    @property
    def cth(self): return (<pxd.PHKLUS*>self._ptr).cth
    @property
    def wcth(self): return (<pxd.PHKLUS*>self._ptr).wcth
    @property
    def phi(self): return (<pxd.PHKLUS*>self._ptr).phi
    @property
    def wphi(self): return (<pxd.PHKLUS*>self._ptr).wphi
    @property
    def nhit2(self): return (<pxd.PHKLUS*>self._ptr).nhit2
    @property
    def cth2(self): return (<pxd.PHKLUS*>self._ptr).cth2
    @property
    def wcth2(self): return (<pxd.PHKLUS*>self._ptr).wcth2
    @property
    def phi2(self): return (<pxd.PHKLUS*>self._ptr).phi2
    @property
    def whphi2(self): return (<pxd.PHKLUS*>self._ptr).whphi2
    @property
    def nhit3(self): return (<pxd.PHKLUS*>self._ptr).nhit3
    @property
    def cth3(self): return (<pxd.PHKLUS*>self._ptr).cth3
    @property
    def wcth3(self): return (<pxd.PHKLUS*>self._ptr).wcth3
    @property
    def phi3(self): return (<pxd.PHKLUS*>self._ptr).phi3
    @property
    def wphi3(self): return (<pxd.PHKLUS*>self._ptr).wphi3

    @property
    def elayer(self):
        cdef pxd.PHKLUS* p_obj = <pxd.PHKLUS*>self._ptr
        return [p_obj.elayer[i] for i in range(8)]

cdef object wrap_phklus(pxd.PHKLUS* ptr):
    cdef PyPHKLUS py_obj = <PyPHKLUS>PyPHKLUS.__new__(PyPHKLUS)
    py_obj._ptr = ptr
    return py_obj

# --- PHWIC ---
cdef class PyPHWIC(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyPHWIC directly.")
    
    @property
    def idstat(self): return (<pxd.PHWIC*>self._ptr).idstat
    @property
    def nhit(self): return (<pxd.PHWIC*>self._ptr).nhit
    @property
    def nhit45(self): return (<pxd.PHWIC*>self._ptr).nhit45
    @property
    def npat(self): return (<pxd.PHWIC*>self._ptr).npat
    @property
    def nhitpat(self): return (<pxd.PHWIC*>self._ptr).nhitpat
    @property
    def syshit(self): return (<pxd.PHWIC*>self._ptr).syshit
    @property
    def qpinit(self): return (<pxd.PHWIC*>self._ptr).qpinit
    @property
    def t1(self): return (<pxd.PHWIC*>self._ptr).t1
    @property
    def t2(self): return (<pxd.PHWIC*>self._ptr).t2
    @property
    def t3(self): return (<pxd.PHWIC*>self._ptr).t3
    @property
    def hitmiss(self): return (<pxd.PHWIC*>self._ptr).hitmiss
    @property
    def itrlen(self): return (<pxd.PHWIC*>self._ptr).itrlen
    @property
    def nlayexp(self): return (<pxd.PHWIC*>self._ptr).nlayexp
    @property
    def nlaybey(self): return (<pxd.PHWIC*>self._ptr).nlaybey
    @property
    def missprob(self): return (<pxd.PHWIC*>self._ptr).missprob
    @property
    def phwicid(self): return (<pxd.PHWIC*>self._ptr).phwicid
    @property
    def nhitshar(self): return (<pxd.PHWIC*>self._ptr).nhitshar
    @property
    def nother(self): return (<pxd.PHWIC*>self._ptr).nother
    @property
    def hitsused(self): return (<pxd.PHWIC*>self._ptr).hitsused
    @property
    def chi2(self): return (<pxd.PHWIC*>self._ptr).chi2
    @property
    def ndf(self): return (<pxd.PHWIC*>self._ptr).ndf
    @property
    def punfit(self): return (<pxd.PHWIC*>self._ptr).punfit
    @property
    def matchChi2(self): return (<pxd.PHWIC*>self._ptr).matchChi2
    @property
    def matchNdf(self): return (<pxd.PHWIC*>self._ptr).matchNdf

    @property
    def pref1(self):
        cdef pxd.PHWIC* p_obj = <pxd.PHWIC*>self._ptr
        return [p_obj.pref1[i] for i in range(3)]
    @property
    def pfit(self):
        cdef pxd.PHWIC* p_obj = <pxd.PHWIC*>self._ptr
        return [p_obj.pfit[i] for i in range(4)]
    @property
    def dpfit(self):
        cdef pxd.PHWIC* p_obj = <pxd.PHWIC*>self._ptr
        return [p_obj.dpfit[i] for i in range(10)]

cdef object wrap_phwic(pxd.PHWIC* ptr):
    cdef PyPHWIC py_obj = <PyPHWIC>PyPHWIC.__new__(PyPHWIC)
    py_obj._ptr = ptr
    return py_obj

# --- PHCRID ---
cdef class PyPHCRID(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyPHCRID directly.")
        
    @property
    def ctlword(self): return (<pxd.PHCRID*>self._ptr).ctlword
    @property
    def norm(self): return (<pxd.PHCRID*>self._ptr).norm
    @property
    def rc(self): return (<pxd.PHCRID*>self._ptr).rc
    @property
    def geom(self): return (<pxd.PHCRID*>self._ptr).geom
    @property
    def trkp(self): return (<pxd.PHCRID*>self._ptr).trkp
    @property
    def nhits(self): return (<pxd.PHCRID*>self._ptr).nhits
        
    @property
    def liq(self):
        return wrap_cridhyp(cython.address((<pxd.PHCRID*>self._ptr).liq))
    @property
    def gas(self):
        return wrap_cridhyp(cython.address((<pxd.PHCRID*>self._ptr).gas))
    @property
    def llik(self):
        return wrap_pidvec(cython.address((<pxd.PHCRID*>self._ptr).llik))

cdef object wrap_phcrid(pxd.PHCRID* ptr):
    cdef PyPHCRID py_obj = <PyPHCRID>PyPHCRID.__new__(PyPHCRID)
    py_obj._ptr = ptr
    return py_obj

# --- PHKTRK ---
cdef class PyPHKTRK(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyPHKTRK directly.")
    # This is a stub bank, no properties

cdef object wrap_phktrk(pxd.PHKTRK* ptr):
    cdef PyPHKTRK py_obj = <PyPHKTRK>PyPHKTRK.__new__(PyPHKTRK)
    py_obj._ptr = ptr
    return py_obj

# --- PHKELID ---
cdef class PyPHKELID(PyBank):
    def __init__(self):
        raise TypeError("Cannot instantiate PyPHKELID directly.")

    @property
    def phchrg(self):
        cdef pxd.PHCHRG* p = (<pxd.PHKELID*>self._ptr).phchrg
        if p == NULL:
            return None
        return wrap_phchrg(p)
    
    @property
    def idstat(self): return (<pxd.PHKELID*>self._ptr).idstat
    @property
    def prob(self): return (<pxd.PHKELID*>self._ptr).prob
    @property
    def phi(self): return (<pxd.PHKELID*>self._ptr).phi
    @property
    def theta(self): return (<pxd.PHKELID*>self._ptr).theta
    @property
    def qp(self): return (<pxd.PHKELID*>self._ptr).qp
    @property
    def dphi(self): return (<pxd.PHKELID*>self._ptr).dphi
    @property
    def dtheta(self): return (<pxd.PHKELID*>self._ptr).dtheta
    @property
    def dqp(self): return (<pxd.PHKELID*>self._ptr).dqp
    @property
    def tphi(self): return (<pxd.PHKELID*>self._ptr).tphi
    @property
    def ttheta(self): return (<pxd.PHKELID*>self._ptr).ttheta
    @property
    def isolat(self): return (<pxd.PHKELID*>self._ptr).isolat
    @property
    def em1(self): return (<pxd.PHKELID*>self._ptr).em1
    @property
    def em12(self): return (<pxd.PHKELID*>self._ptr).em12
    @property
    def dem12(self): return (<pxd.PHKELID*>self._ptr).dem12
    @property
    def had1(self): return (<pxd.PHKELID*>self._ptr).had1
    @property
    def emphi(self): return (<pxd.PHKELID*>self._ptr).emphi
    @property
    def emtheta(self): return (<pxd.PHKELID*>self._ptr).emtheta
    @property
    def phiwid(self): return (<pxd.PHKELID*>self._ptr).phiwid
    @property
    def thewid(self): return (<pxd.PHKELID*>self._ptr).thewid
    @property
    def em1x1(self): return (<pxd.PHKELID*>self._ptr).em1x1
    @property
    def em2x2a(self): return (<pxd.PHKELID*>self._ptr).em2x2a
    @property
    def em2x2b(self): return (<pxd.PHKELID*>self._ptr).em2x2b
    @property
    def em3x3a(self): return (<pxd.PHKELID*>self._ptr).em3x3a
    @property
    def em3x3b(self): return (<pxd.PHKELID*>self._ptr).em3x3b

cdef object wrap_phkelid(pxd.PHKELID* ptr):
    cdef PyPHKELID py_obj = <PyPHKELID>PyPHKELID.__new__(PyPHKELID)
    py_obj._ptr = ptr
    return py_obj


#
# --- Event Container Wrapper ---
#
cdef class JazelleEvent:
    """
    Python wrapper for the JazelleEvent C++ class.
    
    This object is a container for all the bank data for a
    single event. It is populated by JazelleFile.nextRecord()
    or JazelleFile.readEvent().
    """
    # This class *owns* the C++ object
    cdef pxd.CppJazelleEvent cpp_event  # Changed from JazelleEvent
    
    def __cinit__(self):
        # Default constructor is called
        pass
        
    def clear(self):
        """Clears all bank data from the event."""
        self.cpp_event.clear()
        
    @property
    def ieventh(self):
        """The IEVENTH (event header) bank."""
        return wrap_ieventh(cython.address(self.cpp_event.ieventh))

    # --- Convenience Finders ---
    
    def findMCHEAD(self, int id):
        cdef pxd.MCHEAD* p = self.cpp_event.findMCHEAD(id)
        if p == NULL: return None
        return wrap_mchead(p)

    def findMCPART(self, int id):
        cdef pxd.MCPART* p = self.cpp_event.findMCPART(id)
        if p == NULL: return None
        return wrap_mcpart(p)

    def findPHPSUM(self, int id):
        cdef pxd.PHPSUM* p = self.cpp_event.findPHPSUM(id)
        if p == NULL: return None
        return wrap_phpsum(p)

    def findPHCHRG(self, int id):
        cdef pxd.PHCHRG* p = self.cpp_event.findPHCHRG(id)
        if p == NULL: return None
        return wrap_phchrg(p)

    def findPHKLUS(self, int id):
        cdef pxd.PHKLUS* p = self.cpp_event.findPHKLUS(id)
        if p == NULL: return None
        return wrap_phklus(p)

    def findPHWIC(self, int id):
        cdef pxd.PHWIC* p = self.cpp_event.findPHWIC(id)
        if p == NULL: return None
        return wrap_phwic(p)

    def findPHCRID(self, int id):
        cdef pxd.PHCRID* p = self.cpp_event.findPHCRID(id)
        if p == NULL: return None
        return wrap_phcrid(p)

    def findPHKTRK(self, int id):
        cdef pxd.PHKTRK* p = self.cpp_event.findPHKTRK(id)
        if p == NULL: return None
        return wrap_phktrk(p)

    def findPHKELID(self, int id):
        cdef pxd.PHKELID* p = self.cpp_event.findPHKELID(id)
        if p == NULL: return None
        return wrap_phkelid(p)

#
# --- Main File Reader Wrapper ---
#
cdef class JazelleFile:
    """
    Python wrapper for the JazelleFile C++ class.
    
    This is the main entry point for reading a Jazelle file.
    
    Example:
        file = JazelleFile("/path/to/data.jazelle")
        print(f"Total events: {len(file)}")
        
        for event in file:
            print(f"Event: {event.ieventh.event}")
            track = event.findPHCHRG(1)
            if track:
                print(f"  Track 1 charge: {track.charge}")
    """
    # This class owns the C++ object via unique_ptr
    cdef unique_ptr[pxd.CppJazelleFile] cpp_obj  # Changed from JazelleFile
    
    def __cinit__(self, filepath):
        """
        Opens a Jazelle file for reading.
        
        Args:
            filepath (str): Path to the .jazelle file.
        """
        cdef string s_filepath = filepath.encode('UTF-8')
        try:
            self.cpp_obj.reset(new pxd.CppJazelleFile(s_filepath))  # Changed
        except Exception as e:
            raise RuntimeError(f"Error opening Jazelle file: {e}")
            
    def nextRecord(self, JazelleEvent event):
        """
        Reads the next logical record from the file sequentially.
        
        Args:
            event (JazelleEvent): A JazelleEvent object to be populated.
            
        Returns:
            bool: True if a record was read, False if at EOF.
        """
        return self.cpp_obj.get().nextRecord(event.cpp_event)

    def readEvent(self, int index, JazelleEvent event):
        """
        Reads a specific event by its 0-based index.
        
        Args:
            index (int): The event index to read.
            event (JazelleEvent): A JazelleEvent object to be populated.
            
        Returns:
            bool: True if the event was read, False if index is out of bounds.
        """
        return self.cpp_obj.get().readEvent(index, event.cpp_event)

    def getTotalEvents(self):
        """
        Gets the total number of events in the file.
        This will build an index on its first call.
        """
        return self.cpp_obj.get().getTotalEvents()
        
    @property
    def fileName(self):
        """The internal file name from the Jazelle header."""
        return self.cpp_obj.get().getFileName().decode('UTF-8')

    @property
    def creationDate(self):
        """The creation timestamp from the Jazelle header."""
        return cpp_to_py_time(self.cpp_obj.get().getCreationDate())

    @property
    def modifiedDate(self):
        """The modification timestamp from the Jazelle header."""
        return cpp_to_py_time(self.cpp_obj.get().getModifiedDate())

    @property
    def lastRecordType(self):
        """The record type of the *last read* record (e.g., "DATA")."""
        return self.cpp_obj.get().getLastRecordType().decode('UTF-8')

    def __len__(self):
        return self.getTotalEvents()

    def __iter__(self):
        """Provides an iterator for reading all events."""
        cdef int total = self.getTotalEvents()
        cdef int i
        for i in range(total):
            event = JazelleEvent()
            if self.readEvent(i, event):
                yield event
            else:
                raise IndexError(f"Failed to read event at index {i}")

    def __getitem__(self, int index):
        """
        Provides random access to events by index.
        
        Args:
            index (int): The 0-based event index.
            
        Returns:
            JazelleEvent: The event at that index.
            
        Raises:
            IndexError: If the index is out of bounds.
        """
        if index < 0:
            index += len(self)
        
        event = JazelleEvent()
        if self.readEvent(index, event):
            return event
        else:
            raise IndexError(f"Event index {index} out of range for file with {len(self)} events.")