# jazelle_reader/bindings/jazelle_cython.pyx
# distutils: language = c++

import cython
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair
import datetime

from libc.stdint cimport int16_t, int32_t
from libcpp.chrono cimport system_clock, to_time_t

from libcpp.vector cimport vector

cimport jazelle_cython as pxd

# --- Helpers ---

cdef object cpp_to_py_time(system_clock.time_point tp):
    """
    Converts a C++ system_clock::time_point to a Python datetime object.
    Returns None if the timestamp is zero/empty.
    """
    cdef long time_t_val
    try:
        if tp.time_since_epoch().count() == 0:
            return None
        time_t_val = to_time_t(tp)
        return datetime.datetime.fromtimestamp(float(time_t_val))
    except:
        return None

# --- Internal Wrapper Utilities ---

# Forward declaration
cdef class JazelleEvent:
    pass

cdef class Family:
    """
    Wraps a C++ Family template, providing list-like access to Banks in Python.
    """
    cdef pxd.CppIFamily* _ptr
    cdef JazelleEvent _event_ref
    cdef type _wrapper_class

    def __init__(self):
        raise TypeError("Cannot instantiate Family directly.")

    def __len__(self):
        return self._ptr.size()

    def __getitem__(self, int index):
        if index < 0:
            index += len(self)
        
        cdef pxd.CppBank* raw_ptr = self._ptr.at(index)
        if raw_ptr == NULL:
            raise IndexError("Family index out of range")
            
        return wrap_bank(raw_ptr, self._event_ref, self._wrapper_class)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def name(self):
        """Returns the name of the banks in this family."""
        return self._ptr.name()

    @property
    def size(self):
        """Returns the number of banks in this family."""
        return self._ptr.size()

    def find(self, int id):
        """Finds a bank by its ID. Returns None if not found."""
        cdef pxd.CppBank* raw_ptr = self._ptr.find(id)
        return wrap_bank(raw_ptr, self._event_ref, self._wrapper_class)

    cpdef object to_dict(self, str orient='records'):
        """
        Converts the family to a dictionary format.
        
        Args:
            orient (str): 
                'records' (default): Returns [{col: val}, {col: val}]
                'list': Returns {col: [val, val]} (Faster, Pandas-friendly)
        """
        if orient == 'records':
            return self._to_dict_records()
        elif orient == 'list':
            return self._to_dict_list()
        else:
            raise ValueError(f"Invalid orient '{orient}'. Must be 'records' or 'list'.")

    cdef list _to_dict_records(self):
        """
        Converts all banks in this family to a list of dictionaries.
        
        Optimization: Uses a 'Flyweight' pattern. Instead of allocating a new 
        Python wrapper object for every bank (which is slow), we allocate 
        ONE wrapper and point it to each C++ bank in sequence to extract data.
        """
        cdef list result = []
        cdef size_t size = self._ptr.size()
        
        if size == 0:
            return result
            
        cdef size_t i
        cdef pxd.CppBank* raw_ptr
        
        # 1. Allocate ONE reusable wrapper (Flyweight)
        # We use __new__ to bypass __init__ checks
        cdef Bank flyweight = self._wrapper_class.__new__(self._wrapper_class)
        flyweight._event_ref = self._event_ref
        
        for i in range(size):
            raw_ptr = self._ptr.at(i)
            if raw_ptr != NULL:
                # 2. Point the flyweight to the current C++ bank
                flyweight._ptr = raw_ptr
                
                # 3. Call the fast cpdef to_dict implementation
                # This dispatches directly to C++ logic (e.g., MCHEAD.to_dict)
                result.append(flyweight.to_dict())
                
        return result

    cdef dict _to_dict_list(self):
        """Columnar conversion optimized for HDF5/Pandas."""
        cdef size_t size = self._ptr.size()
        
        cdef Bank flyweight = self._wrapper_class.__new__(self._wrapper_class)
        flyweight._event_ref = self._event_ref
        
        cdef list keys = flyweight.get_keys()
        cdef dict columns = {k: [] for k in keys}
        
        if size == 0:
            return columns

        cdef size_t i
        cdef pxd.CppBank* raw_ptr
        
        for i in range(size):
            raw_ptr = self._ptr.at(i)
            if raw_ptr != NULL:
                flyweight._ptr = raw_ptr
                # Fast dispatch to C++ appendage
                flyweight.fill_columns(columns)
                
        return columns    


cdef object wrap_family(pxd.CppIFamily* ptr, JazelleEvent event, type py_class):
    cdef Family obj = Family.__new__(Family)
    obj._ptr = ptr
    obj._event_ref = event
    obj._wrapper_class = py_class
    return obj

# --- Bank Base Class ---

cdef dict _WRAPPER_MAP = {}

cdef class Bank:
    """
    Base wrapper for all Jazelle Bank types.
    """
    cdef pxd.CppBank* _ptr
    cdef JazelleEvent _event_ref

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.id}>"

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        name = cls.__name__
        # Ignore the base class
        if name != "Bank": 
             _WRAPPER_MAP[name] = cls
    
    @property
    def id(self):
        """The unique ID of the bank."""
        return self._ptr.getId()

    cpdef list get_keys(self):
        """Returns a list of keys (column names) for this bank type."""
        raise NotImplementedError()

    cpdef void fill_columns(self, dict columns):
        """
        Appends the current bank's data to the provided dictionary of lists.
        Used for to_dict(orient='list').
        """
        raise NotImplementedError()    

    cpdef dict to_dict(self):
        """
        Converts the bank content to a Python dictionary.
        Must be overridden by subclasses.
        """
        raise NotImplementedError(f"The wrapper class {self.__class__.__name__} has not implemented to_dict()")

cdef object wrap_bank(pxd.CppBank* ptr, JazelleEvent event, type py_class):
    if ptr == NULL:
        return None
    cdef Bank obj = py_class.__new__(py_class)
    obj._ptr = ptr
    obj._event_ref = event
    return obj

# --- Helper Struct Wrappers (PIDVEC, CRIDHYP) ---

cdef class PIDVEC:
    """
    Wrapper for PIDVEC (Particle ID Likelihood Vector).
    """
    cdef pxd.CppPIDVEC* _ptr
    cdef JazelleEvent _event_ref
    
    def __init__(self):
        raise TypeError("Cannot instantiate PIDVEC directly.")

    def __repr__(self):
        return f"<PIDVEC e={self.e:.3f}, mu={self.mu:.3f}, pi={self.pi:.3f}, k={self.k:.3f}, p={self.p:.3f}>"
    
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

    cpdef dict to_dict(self):
        return {
            'e': self._ptr.e, 'mu': self._ptr.mu,
            'pi': self._ptr.pi, 'k': self._ptr.k, 'p': self._ptr.p
        }

cdef class CRIDHYP:
    """
    Wrapper for CRIDHYP (CRID Hypothesis Data).
    """
    cdef pxd.CppCRIDHYP* _ptr
    cdef JazelleEvent _event_ref
    
    def __init__(self):
        raise TypeError("Cannot instantiate CRIDHYP directly.")
    
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
        return wrap_pidvec(cython.address(self._ptr.llik.value()), self._event_ref)

    cpdef dict to_dict(self):
        cdef dict data = {
            'is_full': self._ptr.m_full,
            'rc': self._ptr.rc,
            'nhits': self._ptr.nhits,
            'besthyp': self._ptr.besthyp,
            'nhexp': self._ptr.nhexp,
            'nhfnd': self._ptr.nhfnd,
            'nhbkg': self._ptr.nhbkg,
            'mskphot': self._ptr.mskphot
        }
        if self._ptr.llik.has_value():
            v = self._ptr.llik.value()
            data['llik'] = {'e': v.e, 'mu': v.mu, 'pi': v.pi, 'k': v.k, 'p': v.p}
        else:
            data['llik'] = None
        return data

cdef object wrap_pidvec(pxd.CppPIDVEC* ptr, JazelleEvent event):
    cdef PIDVEC py_obj = <PIDVEC>PIDVEC.__new__(PIDVEC)
    py_obj._ptr = ptr
    py_obj._event_ref = event
    return py_obj

cdef object wrap_cridhyp(pxd.CppCRIDHYP* ptr, JazelleEvent event):
    cdef CRIDHYP py_obj = <CRIDHYP>CRIDHYP.__new__(CRIDHYP)
    py_obj._ptr = ptr
    py_obj._event_ref = event
    return py_obj


# --- Concrete Banks ---

cdef class IEVENTH(Bank):
    """Wrapper for IEVENTH (Event Header) Bank."""
    def __init__(self):
        raise TypeError("Cannot instantiate IEVENTH directly.")
    
    @property
    def run(self): return (<pxd.CppIEVENTH*>self._ptr).run
    @property
    def event(self): return (<pxd.CppIEVENTH*>self._ptr).event
    @property
    def evttype(self): return (<pxd.CppIEVENTH*>self._ptr).evttype
    @property
    def trigger(self): return (<pxd.CppIEVENTH*>self._ptr).trigger
    @property
    def weight(self): return (<pxd.CppIEVENTH*>self._ptr).weight
    @property
    def evttime(self):
        return cpp_to_py_time((<pxd.CppIEVENTH*>self._ptr).evttime)
    @property
    def header(self): return (<pxd.CppIEVENTH*>self._ptr).header

    cpdef dict to_dict(self):
        cdef pxd.CppIEVENTH* ptr = <pxd.CppIEVENTH*>self._ptr
        return {
            'id': ptr.getId(),
            'header': ptr.header,
            'run': ptr.run,
            'event': ptr.event,
            'evttype': ptr.evttype,
            'trigger': ptr.trigger,
            'weight': ptr.weight,
            'evttime': cpp_to_py_time(ptr.evttime)
        }

    cpdef list get_keys(self):
        return ['id', 'header', 'run', 'event', 'evttype', 'trigger', 'weight', 'evttime']

    cpdef void fill_columns(self, dict c):
        cdef pxd.CppIEVENTH* ptr = <pxd.CppIEVENTH*>self._ptr
        c['id'].append(ptr.getId())
        c['header'].append(ptr.header)
        c['run'].append(ptr.run)
        c['event'].append(ptr.event)
        c['evttype'].append(ptr.evttype)
        c['trigger'].append(ptr.trigger)
        c['weight'].append(ptr.weight)
        c['evttime'].append(cpp_to_py_time(ptr.evttime))        

cdef class MCHEAD(Bank):
    """Wrapper for MCHEAD (Monte Carlo Header) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")
    @property
    def ntot(self): return (<pxd.CppMCHEAD*>self._ptr).ntot
    @property
    def origin(self): return (<pxd.CppMCHEAD*>self._ptr).origin
    @property
    def ipx(self): return (<pxd.CppMCHEAD*>self._ptr).ipx
    @property
    def ipy(self): return (<pxd.CppMCHEAD*>self._ptr).ipy
    @property
    def ipz(self): return (<pxd.CppMCHEAD*>self._ptr).ipz

    cpdef dict to_dict(self):
        cdef pxd.CppMCHEAD* ptr = <pxd.CppMCHEAD*>self._ptr
        return {
            'id': ptr.getId(),
            'ntot': ptr.ntot, 'origin': ptr.origin,
            'ipx': ptr.ipx, 'ipy': ptr.ipy, 'ipz': ptr.ipz
        }

    cpdef list get_keys(self):
        return ['id', 'ntot', 'origin', 'ipx', 'ipy', 'ipz']

    cpdef void fill_columns(self, dict c):
        cdef pxd.CppMCHEAD* ptr = <pxd.CppMCHEAD*>self._ptr
        c['id'].append(ptr.getId())
        c['ntot'].append(ptr.ntot)
        c['origin'].append(ptr.origin)
        c['ipx'].append(ptr.ipx)
        c['ipy'].append(ptr.ipy)
        c['ipz'].append(ptr.ipz)        

cdef class MCPART(Bank):
    """Wrapper for MCPART (Monte Carlo Particle) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")
    @property
    def e(self): return (<pxd.CppMCPART*>self._ptr).e
    @property
    def ptot(self): return (<pxd.CppMCPART*>self._ptr).ptot
    @property
    def ptype(self): return (<pxd.CppMCPART*>self._ptr).ptype
    @property
    def charge(self): return (<pxd.CppMCPART*>self._ptr).charge
    @property
    def origin(self): return (<pxd.CppMCPART*>self._ptr).origin
    @property
    def parent_id(self): return (<pxd.CppMCPART*>self._ptr).parent_id
    @property
    def p(self):
        cdef pxd.CppMCPART* p_obj = <pxd.CppMCPART*>self._ptr
        return [p_obj.p[i] for i in range(3)]
    @property
    def xt(self):
        cdef pxd.CppMCPART* p_obj = <pxd.CppMCPART*>self._ptr
        return [p_obj.xt[i] for i in range(3)]

    cpdef dict to_dict(self):
        cdef pxd.CppMCPART* ptr = <pxd.CppMCPART*>self._ptr
        return {
            'id': ptr.getId(),
            'e': ptr.e, 'ptot': ptr.ptot,
            'ptype': ptr.ptype, 'charge': ptr.charge,
            'origin': ptr.origin, 'parent_id': ptr.parent_id,
            'p': [ptr.p[i] for i in range(3)],
            'xt': [ptr.xt[i] for i in range(3)]
        }

    cpdef list get_keys(self):
        return ['id', 'e', 'ptot', 'ptype', 'charge', 'origin', 'parent_id', 'p', 'xt']

    cpdef void fill_columns(self, dict c):
        cdef pxd.CppMCPART* ptr = <pxd.CppMCPART*>self._ptr
        c['id'].append(ptr.getId())
        c['e'].append(ptr.e)
        c['ptot'].append(ptr.ptot)
        c['ptype'].append(ptr.ptype)
        c['charge'].append(ptr.charge)
        c['origin'].append(ptr.origin)
        c['parent_id'].append(ptr.parent_id)
        c['p'].append([ptr.p[i] for i in range(3)])
        c['xt'].append([ptr.xt[i] for i in range(3)])        

cdef class PHPSUM(Bank):
    """Wrapper for PHPSUM (Physics Particle Summary) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")
    @property
    def px(self): return (<pxd.CppPHPSUM*>self._ptr).px
    @property
    def py(self): return (<pxd.CppPHPSUM*>self._ptr).py
    @property
    def pz(self): return (<pxd.CppPHPSUM*>self._ptr).pz
    @property
    def x(self): return (<pxd.CppPHPSUM*>self._ptr).x
    @property
    def y(self): return (<pxd.CppPHPSUM*>self._ptr).y
    @property
    def z(self): return (<pxd.CppPHPSUM*>self._ptr).z
    @property
    def charge(self): return (<pxd.CppPHPSUM*>self._ptr).charge
    @property
    def status(self): return (<pxd.CppPHPSUM*>self._ptr).status
    def getPTot(self): return (<pxd.CppPHPSUM*>self._ptr).getPTot()

    cpdef dict to_dict(self):
        cdef pxd.CppPHPSUM* ptr = <pxd.CppPHPSUM*>self._ptr
        return {
            'id': ptr.getId(),
            'px': ptr.px, 'py': ptr.py, 'pz': ptr.pz,
            'x': ptr.x, 'y': ptr.y, 'z': ptr.z,
            'charge': ptr.charge, 'status': ptr.status,
            'ptot': ptr.getPTot()
        }

    cpdef list get_keys(self):
        return ['id', 'px', 'py', 'pz', 'x', 'y', 'z', 'charge', 'status', 'ptot']

    cpdef void fill_columns(self, dict c):
        cdef pxd.CppPHPSUM* ptr = <pxd.CppPHPSUM*>self._ptr
        c['id'].append(ptr.getId())
        c['px'].append(ptr.px)
        c['py'].append(ptr.py)
        c['pz'].append(ptr.pz)
        c['x'].append(ptr.x)
        c['y'].append(ptr.y)
        c['z'].append(ptr.z)
        c['charge'].append(ptr.charge)
        c['status'].append(ptr.status)
        c['ptot'].append(ptr.getPTot())        

cdef class PHCHRG(Bank):
    """Wrapper for PHCHRG (Charged Track) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")
    @property
    def bnorm(self): return (<pxd.CppPHCHRG*>self._ptr).bnorm
    @property
    def impact(self): return (<pxd.CppPHCHRG*>self._ptr).impact
    @property
    def b3norm(self): return (<pxd.CppPHCHRG*>self._ptr).b3norm
    @property
    def impact3(self): return (<pxd.CppPHCHRG*>self._ptr).impact3
    @property
    def charge(self): return (<pxd.CppPHCHRG*>self._ptr).charge
    @property
    def smwstat(self): return (<pxd.CppPHCHRG*>self._ptr).smwstat
    @property
    def status(self): return (<pxd.CppPHCHRG*>self._ptr).status
    @property
    def tkpar0(self): return (<pxd.CppPHCHRG*>self._ptr).tkpar0
    @property
    def length(self): return (<pxd.CppPHCHRG*>self._ptr).length
    @property
    def chi2dt(self): return (<pxd.CppPHCHRG*>self._ptr).chi2dt
    @property
    def imc(self): return (<pxd.CppPHCHRG*>self._ptr).imc
    @property
    def ndfdt(self): return (<pxd.CppPHCHRG*>self._ptr).ndfdt
    @property
    def nhit(self): return (<pxd.CppPHCHRG*>self._ptr).nhit
    @property
    def nhite(self): return (<pxd.CppPHCHRG*>self._ptr).nhite
    @property
    def nhitp(self): return (<pxd.CppPHCHRG*>self._ptr).nhitp
    @property
    def nmisht(self): return (<pxd.CppPHCHRG*>self._ptr).nmisht
    @property
    def nwrght(self): return (<pxd.CppPHCHRG*>self._ptr).nwrght
    @property
    def nhitv(self): return (<pxd.CppPHCHRG*>self._ptr).nhitv
    @property
    def chi2(self): return (<pxd.CppPHCHRG*>self._ptr).chi2
    @property
    def chi2v(self): return (<pxd.CppPHCHRG*>self._ptr).chi2v
    @property
    def vxdhit(self): return (<pxd.CppPHCHRG*>self._ptr).vxdhit
    @property
    def mustat(self): return (<pxd.CppPHCHRG*>self._ptr).mustat
    @property
    def estat(self): return (<pxd.CppPHCHRG*>self._ptr).estat
    @property
    def dedx(self): return (<pxd.CppPHCHRG*>self._ptr).dedx
    @property
    def hlxpar(self): return [(<pxd.CppPHCHRG*>self._ptr).hlxpar[i] for i in range(6)]
    @property
    def dhlxpar(self): return [(<pxd.CppPHCHRG*>self._ptr).dhlxpar[i] for i in range(15)]
    @property
    def tkpar(self): return [(<pxd.CppPHCHRG*>self._ptr).tkpar[i] for i in range(5)]
    @property
    def dtkpar(self): return [(<pxd.CppPHCHRG*>self._ptr).dtkpar[i] for i in range(15)]

    cpdef dict to_dict(self):
        cdef pxd.CppPHCHRG* ptr = <pxd.CppPHCHRG*>self._ptr
        return {
            'id': ptr.getId(),
            'bnorm': ptr.bnorm, 'impact': ptr.impact, 'b3norm': ptr.b3norm, 'impact3': ptr.impact3,
            'charge': ptr.charge, 'smwstat': ptr.smwstat, 'status': ptr.status,
            'tkpar0': ptr.tkpar0, 'length': ptr.length, 'chi2dt': ptr.chi2dt,
            'imc': ptr.imc, 'ndfdt': ptr.ndfdt, 'nhit': ptr.nhit, 'nhite': ptr.nhite,
            'nhitp': ptr.nhitp, 'nmisht': ptr.nmisht, 'nwrght': ptr.nwrght, 'nhitv': ptr.nhitv,
            'chi2': ptr.chi2, 'chi2v': ptr.chi2v, 'vxdhit': ptr.vxdhit, 'mustat': ptr.mustat,
            'estat': ptr.estat, 'dedx': ptr.dedx,
            'hlxpar': [ptr.hlxpar[i] for i in range(6)],
            'dhlxpar': [ptr.dhlxpar[i] for i in range(15)],
            'tkpar': [ptr.tkpar[i] for i in range(5)],
            'dtkpar': [ptr.dtkpar[i] for i in range(15)]
        }

    cpdef list get_keys(self):
        return [
            'id', 'bnorm', 'impact', 'b3norm', 'impact3', 'charge', 'smwstat', 'status',
            'tkpar0', 'length', 'chi2dt', 'imc', 'ndfdt', 'nhit', 'nhite', 'nhitp',
            'nmisht', 'nwrght', 'nhitv', 'chi2', 'chi2v', 'vxdhit', 'mustat', 'estat', 'dedx',
            'hlxpar', 'dhlxpar', 'tkpar', 'dtkpar'
        ]

    cpdef void fill_columns(self, dict c):
        cdef pxd.CppPHCHRG* ptr = <pxd.CppPHCHRG*>self._ptr
        c['id'].append(ptr.getId())
        c['bnorm'].append(ptr.bnorm)
        c['impact'].append(ptr.impact)
        c['b3norm'].append(ptr.b3norm)
        c['impact3'].append(ptr.impact3)
        c['charge'].append(ptr.charge)
        c['smwstat'].append(ptr.smwstat)
        c['status'].append(ptr.status)
        c['tkpar0'].append(ptr.tkpar0)
        c['length'].append(ptr.length)
        c['chi2dt'].append(ptr.chi2dt)
        c['imc'].append(ptr.imc)
        c['ndfdt'].append(ptr.ndfdt)
        c['nhit'].append(ptr.nhit)
        c['nhite'].append(ptr.nhite)
        c['nhitp'].append(ptr.nhitp)
        c['nmisht'].append(ptr.nmisht)
        c['nwrght'].append(ptr.nwrght)
        c['nhitv'].append(ptr.nhitv)
        c['chi2'].append(ptr.chi2)
        c['chi2v'].append(ptr.chi2v)
        c['vxdhit'].append(ptr.vxdhit)
        c['mustat'].append(ptr.mustat)
        c['estat'].append(ptr.estat)
        c['dedx'].append(ptr.dedx)
        # Arrays
        c['hlxpar'].append([ptr.hlxpar[i] for i in range(6)])
        c['dhlxpar'].append([ptr.dhlxpar[i] for i in range(15)])
        c['tkpar'].append([ptr.tkpar[i] for i in range(5)])
        c['dtkpar'].append([ptr.dtkpar[i] for i in range(15)])        

cdef class PHKLUS(Bank):
    """Wrapper for PHKLUS (Calorimeter Cluster) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")
    @property
    def status(self): return (<pxd.CppPHKLUS*>self._ptr).status
    @property
    def eraw(self): return (<pxd.CppPHKLUS*>self._ptr).eraw
    @property
    def cth(self): return (<pxd.CppPHKLUS*>self._ptr).cth
    @property
    def wcth(self): return (<pxd.CppPHKLUS*>self._ptr).wcth
    @property
    def phi(self): return (<pxd.CppPHKLUS*>self._ptr).phi
    @property
    def wphi(self): return (<pxd.CppPHKLUS*>self._ptr).wphi
    @property
    def nhit2(self): return (<pxd.CppPHKLUS*>self._ptr).nhit2
    @property
    def cth2(self): return (<pxd.CppPHKLUS*>self._ptr).cth2
    @property
    def wcth2(self): return (<pxd.CppPHKLUS*>self._ptr).wcth2
    @property
    def phi2(self): return (<pxd.CppPHKLUS*>self._ptr).phi2
    @property
    def whphi2(self): return (<pxd.CppPHKLUS*>self._ptr).whphi2
    @property
    def nhit3(self): return (<pxd.CppPHKLUS*>self._ptr).nhit3
    @property
    def cth3(self): return (<pxd.CppPHKLUS*>self._ptr).cth3
    @property
    def wcth3(self): return (<pxd.CppPHKLUS*>self._ptr).wcth3
    @property
    def phi3(self): return (<pxd.CppPHKLUS*>self._ptr).phi3
    @property
    def wphi3(self): return (<pxd.CppPHKLUS*>self._ptr).wphi3
    @property
    def elayer(self): return [(<pxd.CppPHKLUS*>self._ptr).elayer[i] for i in range(8)]

    cpdef dict to_dict(self):
        cdef pxd.CppPHKLUS* ptr = <pxd.CppPHKLUS*>self._ptr
        return {
            'id': ptr.getId(),
            'status': ptr.status, 'eraw': ptr.eraw,
            'cth': ptr.cth, 'wcth': ptr.wcth, 'phi': ptr.phi, 'wphi': ptr.wphi,
            'nhit2': ptr.nhit2, 'cth2': ptr.cth2, 'wcth2': ptr.wcth2,
            'phi2': ptr.phi2, 'whphi2': ptr.whphi2,
            'nhit3': ptr.nhit3, 'cth3': ptr.cth3, 'wcth3': ptr.wcth3,
            'phi3': ptr.phi3, 'wphi3': ptr.wphi3,
            'elayer': [ptr.elayer[i] for i in range(8)]
        }

    cpdef list get_keys(self):
        return [
            'id', 'status', 'eraw', 'cth', 'wcth', 'phi', 'wphi',
            'nhit2', 'cth2', 'wcth2', 'phi2', 'whphi2',
            'nhit3', 'cth3', 'wcth3', 'phi3', 'wphi3', 'elayer'
        ]

    cpdef void fill_columns(self, dict c):
        cdef pxd.CppPHKLUS* ptr = <pxd.CppPHKLUS*>self._ptr
        c['id'].append(ptr.getId())
        c['status'].append(ptr.status)
        c['eraw'].append(ptr.eraw)
        c['cth'].append(ptr.cth)
        c['wcth'].append(ptr.wcth)
        c['phi'].append(ptr.phi)
        c['wphi'].append(ptr.wphi)
        c['nhit2'].append(ptr.nhit2)
        c['cth2'].append(ptr.cth2)
        c['wcth2'].append(ptr.wcth2)
        c['phi2'].append(ptr.phi2)
        c['whphi2'].append(ptr.whphi2)
        c['nhit3'].append(ptr.nhit3)
        c['cth3'].append(ptr.cth3)
        c['wcth3'].append(ptr.wcth3)
        c['phi3'].append(ptr.phi3)
        c['wphi3'].append(ptr.wphi3)
        c['elayer'].append([ptr.elayer[i] for i in range(8)])        

cdef class PHWIC(Bank):
    """Wrapper for PHWIC (Warm Iron Calorimeter) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")
    @property
    def idstat(self): return (<pxd.CppPHWIC*>self._ptr).idstat
    @property
    def nhit(self): return (<pxd.CppPHWIC*>self._ptr).nhit
    @property
    def nhit45(self): return (<pxd.CppPHWIC*>self._ptr).nhit45
    @property
    def npat(self): return (<pxd.CppPHWIC*>self._ptr).npat
    @property
    def nhitpat(self): return (<pxd.CppPHWIC*>self._ptr).nhitpat
    @property
    def syshit(self): return (<pxd.CppPHWIC*>self._ptr).syshit
    @property
    def qpinit(self): return (<pxd.CppPHWIC*>self._ptr).qpinit
    @property
    def t1(self): return (<pxd.CppPHWIC*>self._ptr).t1
    @property
    def t2(self): return (<pxd.CppPHWIC*>self._ptr).t2
    @property
    def t3(self): return (<pxd.CppPHWIC*>self._ptr).t3
    @property
    def hitmiss(self): return (<pxd.CppPHWIC*>self._ptr).hitmiss
    @property
    def itrlen(self): return (<pxd.CppPHWIC*>self._ptr).itrlen
    @property
    def nlayexp(self): return (<pxd.CppPHWIC*>self._ptr).nlayexp
    @property
    def nlaybey(self): return (<pxd.CppPHWIC*>self._ptr).nlaybey
    @property
    def missprob(self): return (<pxd.CppPHWIC*>self._ptr).missprob
    @property
    def phwicid(self): return (<pxd.CppPHWIC*>self._ptr).phwicid
    @property
    def nhitshar(self): return (<pxd.CppPHWIC*>self._ptr).nhitshar
    @property
    def nother(self): return (<pxd.CppPHWIC*>self._ptr).nother
    @property
    def hitsused(self): return (<pxd.CppPHWIC*>self._ptr).hitsused
    @property
    def chi2(self): return (<pxd.CppPHWIC*>self._ptr).chi2
    @property
    def ndf(self): return (<pxd.CppPHWIC*>self._ptr).ndf
    @property
    def punfit(self): return (<pxd.CppPHWIC*>self._ptr).punfit
    @property
    def matchChi2(self): return (<pxd.CppPHWIC*>self._ptr).matchChi2
    @property
    def matchNdf(self): return (<pxd.CppPHWIC*>self._ptr).matchNdf
    @property
    def pref1(self): return [(<pxd.CppPHWIC*>self._ptr).pref1[i] for i in range(3)]
    @property
    def pfit(self): return [(<pxd.CppPHWIC*>self._ptr).pfit[i] for i in range(4)]
    @property
    def dpfit(self): return [(<pxd.CppPHWIC*>self._ptr).dpfit[i] for i in range(10)]

    cpdef dict to_dict(self):
        cdef pxd.CppPHWIC* ptr = <pxd.CppPHWIC*>self._ptr
        return {
            'id': ptr.getId(),
            'idstat': ptr.idstat, 'nhit': ptr.nhit, 'nhit45': ptr.nhit45,
            'npat': ptr.npat, 'nhitpat': ptr.nhitpat, 'syshit': ptr.syshit,
            'qpinit': ptr.qpinit, 't1': ptr.t1, 't2': ptr.t2, 't3': ptr.t3,
            'hitmiss': ptr.hitmiss, 'itrlen': ptr.itrlen,
            'nlayexp': ptr.nlayexp, 'nlaybey': ptr.nlaybey, 'missprob': ptr.missprob,
            'phwicid': ptr.phwicid, 'nhitshar': ptr.nhitshar, 'nother': ptr.nother,
            'hitsused': ptr.hitsused, 'chi2': ptr.chi2, 'ndf': ptr.ndf,
            'punfit': ptr.punfit, 'matchChi2': ptr.matchChi2, 'matchNdf': ptr.matchNdf,
            'pref1': [ptr.pref1[i] for i in range(3)],
            'pfit': [ptr.pfit[i] for i in range(4)],
            'dpfit': [ptr.dpfit[i] for i in range(10)]
        }

    cpdef list get_keys(self):
        return [
            'id', 'idstat', 'nhit', 'nhit45', 'npat', 'nhitpat', 'syshit',
            'qpinit', 't1', 't2', 't3', 'hitmiss', 'itrlen',
            'nlayexp', 'nlaybey', 'missprob', 'phwicid', 'nhitshar', 'nother',
            'hitsused', 'chi2', 'ndf', 'punfit', 'matchChi2', 'matchNdf',
            'pref1', 'pfit', 'dpfit'
        ]

    cpdef void fill_columns(self, dict c):
        cdef pxd.CppPHWIC* ptr = <pxd.CppPHWIC*>self._ptr
        c['id'].append(ptr.getId())
        c['idstat'].append(ptr.idstat)
        c['nhit'].append(ptr.nhit)
        c['nhit45'].append(ptr.nhit45)
        c['npat'].append(ptr.npat)
        c['nhitpat'].append(ptr.nhitpat)
        c['syshit'].append(ptr.syshit)
        c['qpinit'].append(ptr.qpinit)
        c['t1'].append(ptr.t1)
        c['t2'].append(ptr.t2)
        c['t3'].append(ptr.t3)
        c['hitmiss'].append(ptr.hitmiss)
        c['itrlen'].append(ptr.itrlen)
        c['nlayexp'].append(ptr.nlayexp)
        c['nlaybey'].append(ptr.nlaybey)
        c['missprob'].append(ptr.missprob)
        c['phwicid'].append(ptr.phwicid)
        c['nhitshar'].append(ptr.nhitshar)
        c['nother'].append(ptr.nother)
        c['hitsused'].append(ptr.hitsused)
        c['chi2'].append(ptr.chi2)
        c['ndf'].append(ptr.ndf)
        c['punfit'].append(ptr.punfit)
        c['matchChi2'].append(ptr.matchChi2)
        c['matchNdf'].append(ptr.matchNdf)
        # Arrays
        c['pref1'].append([ptr.pref1[i] for i in range(3)])
        c['pfit'].append([ptr.pfit[i] for i in range(4)])
        c['dpfit'].append([ptr.dpfit[i] for i in range(10)])

cdef class PHCRID(Bank):
    """Wrapper for PHCRID (Cerenkov Ring Imaging) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")
    @property
    def ctlword(self): return (<pxd.CppPHCRID*>self._ptr).ctlword
    @property
    def norm(self): return (<pxd.CppPHCRID*>self._ptr).norm
    @property
    def rc(self): return (<pxd.CppPHCRID*>self._ptr).rc
    @property
    def geom(self): return (<pxd.CppPHCRID*>self._ptr).geom
    @property
    def trkp(self): return (<pxd.CppPHCRID*>self._ptr).trkp
    @property
    def nhits(self): return (<pxd.CppPHCRID*>self._ptr).nhits
    @property
    def liq(self): return wrap_cridhyp(cython.address((<pxd.CppPHCRID*>self._ptr).liq), self._event_ref)
    @property
    def gas(self): return wrap_cridhyp(cython.address((<pxd.CppPHCRID*>self._ptr).gas), self._event_ref)
    @property
    def llik(self): return wrap_pidvec(cython.address((<pxd.CppPHCRID*>self._ptr).llik), self._event_ref)

    cpdef dict to_dict(self):
        cdef pxd.CppPHCRID* ptr = <pxd.CppPHCRID*>self._ptr
        return {
            'id': ptr.getId(),
            'ctlword': ptr.ctlword, 'norm': ptr.norm,
            'rc': ptr.rc, 'geom': ptr.geom, 'trkp': ptr.trkp, 'nhits': ptr.nhits,
            'liq': self.liq.to_dict(),
            'gas': self.gas.to_dict(),
            'llik': self.llik.to_dict()
        }

    cpdef list get_keys(self):
        return ['id', 'ctlword', 'norm', 'rc', 'geom', 'trkp', 'nhits', 'liq', 'gas', 'llik']

    cpdef void fill_columns(self, dict c):
        cdef pxd.CppPHCRID* ptr = <pxd.CppPHCRID*>self._ptr
        c['id'].append(ptr.getId())
        c['ctlword'].append(ptr.ctlword)
        c['norm'].append(ptr.norm)
        c['rc'].append(ptr.rc)
        c['geom'].append(ptr.geom)
        c['trkp'].append(ptr.trkp)
        c['nhits'].append(ptr.nhits)
        
        # Use nested dicts for structs
        c['liq'].append(self.liq.to_dict())
        c['gas'].append(self.gas.to_dict())
        c['llik'].append(self.llik.to_dict())        

cdef class PHKTRK(Bank):
    """Wrapper for PHKTRK (Stub) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")
    
    cpdef dict to_dict(self):
        return {'id': self._ptr.getId()}

    cpdef list get_keys(self):
        return ['id']

    cpdef void fill_columns(self, dict c):
        c['id'].append(self._ptr.getId())

cdef class PHKELID(Bank):
    """Wrapper for PHKELID (Calorimeter/Electron ID) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")
    @property
    def phchrg(self):
        cdef pxd.CppPHCHRG* p = (<pxd.CppPHKELID*>self._ptr).phchrg
        if p == NULL: return None
        return wrap_bank(<pxd.CppBank*>p, self._event_ref, PHCHRG)
    @property
    def idstat(self): return (<pxd.CppPHKELID*>self._ptr).idstat
    @property
    def prob(self): return (<pxd.CppPHKELID*>self._ptr).prob
    @property
    def phi(self): return (<pxd.CppPHKELID*>self._ptr).phi
    @property
    def theta(self): return (<pxd.CppPHKELID*>self._ptr).theta
    @property
    def qp(self): return (<pxd.CppPHKELID*>self._ptr).qp
    @property
    def dphi(self): return (<pxd.CppPHKELID*>self._ptr).dphi
    @property
    def dtheta(self): return (<pxd.CppPHKELID*>self._ptr).dtheta
    @property
    def dqp(self): return (<pxd.CppPHKELID*>self._ptr).dqp
    @property
    def tphi(self): return (<pxd.CppPHKELID*>self._ptr).tphi
    @property
    def ttheta(self): return (<pxd.CppPHKELID*>self._ptr).ttheta
    @property
    def isolat(self): return (<pxd.CppPHKELID*>self._ptr).isolat
    @property
    def em1(self): return (<pxd.CppPHKELID*>self._ptr).em1
    @property
    def em12(self): return (<pxd.CppPHKELID*>self._ptr).em12
    @property
    def dem12(self): return (<pxd.CppPHKELID*>self._ptr).dem12
    @property
    def had1(self): return (<pxd.CppPHKELID*>self._ptr).had1
    @property
    def emphi(self): return (<pxd.CppPHKELID*>self._ptr).emphi
    @property
    def emtheta(self): return (<pxd.CppPHKELID*>self._ptr).emtheta
    @property
    def phiwid(self): return (<pxd.CppPHKELID*>self._ptr).phiwid
    @property
    def thewid(self): return (<pxd.CppPHKELID*>self._ptr).thewid
    @property
    def em1x1(self): return (<pxd.CppPHKELID*>self._ptr).em1x1
    @property
    def em2x2a(self): return (<pxd.CppPHKELID*>self._ptr).em2x2a
    @property
    def em2x2b(self): return (<pxd.CppPHKELID*>self._ptr).em2x2b
    @property
    def em3x3a(self): return (<pxd.CppPHKELID*>self._ptr).em3x3a
    @property
    def em3x3b(self): return (<pxd.CppPHKELID*>self._ptr).em3x3b

    cpdef dict to_dict(self):
        cdef pxd.CppPHKELID* ptr = <pxd.CppPHKELID*>self._ptr
        cdef object linked_id = None
        if ptr.phchrg != NULL:
            linked_id = ptr.phchrg.getId()
            
        return {
            'id': ptr.getId(),
            'phchrg_id': linked_id,
            'idstat': ptr.idstat, 'prob': ptr.prob,
            'phi': ptr.phi, 'theta': ptr.theta, 'qp': ptr.qp,
            'dphi': ptr.dphi, 'dtheta': ptr.dtheta, 'dqp': ptr.dqp,
            'tphi': ptr.tphi, 'ttheta': ptr.ttheta, 'isolat': ptr.isolat,
            'em1': ptr.em1, 'em12': ptr.em12, 'dem12': ptr.dem12, 'had1': ptr.had1,
            'emphi': ptr.emphi, 'emtheta': ptr.emtheta, 'phiwid': ptr.phiwid, 'thewid': ptr.thewid,
            'em1x1': ptr.em1x1, 'em2x2a': ptr.em2x2a, 'em2x2b': ptr.em2x2b,
            'em3x3a': ptr.em3x3a, 'em3x3b': ptr.em3x3b
        }

    cpdef list get_keys(self):
        return [
            'id', 'phchrg_id', 'idstat', 'prob', 'phi', 'theta', 'qp',
            'dphi', 'dtheta', 'dqp', 'tphi', 'ttheta', 'isolat',
            'em1', 'em12', 'dem12', 'had1', 'emphi', 'emtheta', 'phiwid', 'thewid',
            'em1x1', 'em2x2a', 'em2x2b', 'em3x3a', 'em3x3b'
        ]

    cpdef void fill_columns(self, dict c):
        cdef pxd.CppPHKELID* ptr = <pxd.CppPHKELID*>self._ptr
        
        c['id'].append(ptr.getId())
        
        # Handle linked ID
        if ptr.phchrg != NULL:
            c['phchrg_id'].append(ptr.phchrg.getId())
        else:
            c['phchrg_id'].append(None)
            
        c['idstat'].append(ptr.idstat)
        c['prob'].append(ptr.prob)
        c['phi'].append(ptr.phi)
        c['theta'].append(ptr.theta)
        c['qp'].append(ptr.qp)
        c['dphi'].append(ptr.dphi)
        c['dtheta'].append(ptr.dtheta)
        c['dqp'].append(ptr.dqp)
        c['tphi'].append(ptr.tphi)
        c['ttheta'].append(ptr.ttheta)
        c['isolat'].append(ptr.isolat)
        c['em1'].append(ptr.em1)
        c['em12'].append(ptr.em12)
        c['dem12'].append(ptr.dem12)
        c['had1'].append(ptr.had1)
        c['emphi'].append(ptr.emphi)
        c['emtheta'].append(ptr.emtheta)
        c['phiwid'].append(ptr.phiwid)
        c['thewid'].append(ptr.thewid)
        c['em1x1'].append(ptr.em1x1)
        c['em2x2a'].append(ptr.em2x2a)
        c['em2x2b'].append(ptr.em2x2b)
        c['em3x3a'].append(ptr.em3x3a)
        c['em3x3b'].append(ptr.em3x3b)        

# --- JazelleEvent Wrapper ---

cdef class JazelleEvent:
    """
    Wrapper for the C++ JazelleEvent class.
    Holds all bank data for a single event.
    """
    cdef pxd.CppJazelleEvent cpp_event
    cdef tuple _cached_families
    
    def __cinit__(self):
        self._cached_families = None
        
    def clear(self):
        """Clears all data from the event."""
        self.cpp_event.clear()
        
    @property
    def ieventh(self):
        """The IEVENTH (Event Header) Bank."""
        return wrap_bank(cython.address(self.cpp_event.ieventh), self, IEVENTH)

    # Families
    @property
    def mchead(self): 
        return wrap_family(&self.cpp_event.get[pxd.CppMCHEAD](), self, MCHEAD)

    @property
    def mcpart(self): 
        return wrap_family(&self.cpp_event.get[pxd.CppMCPART](), self, MCPART)

    @property
    def phpsum(self): 
        return wrap_family(&self.cpp_event.get[pxd.CppPHPSUM](), self, PHPSUM)

    @property
    def phchrg(self): 
        return wrap_family(&self.cpp_event.get[pxd.CppPHCHRG](), self, PHCHRG)

    @property
    def phklus(self): 
        return wrap_family(&self.cpp_event.get[pxd.CppPHKLUS](), self, PHKLUS)

    @property
    def phwic(self): 
        return wrap_family(&self.cpp_event.get[pxd.CppPHWIC](), self, PHWIC)

    @property
    def phcrid(self): 
        return wrap_family(&self.cpp_event.get[pxd.CppPHCRID](), self, PHCRID)

    @property
    def phktrk(self): 
        return wrap_family(&self.cpp_event.get[pxd.CppPHKTRK](), self, PHKTRK)

    @property
    def phkelid(self): 
        return wrap_family(&self.cpp_event.get[pxd.CppPHKELID](), self, PHKELID)

    def getFamily(self, identifier):
        cdef str py_name
        if isinstance(identifier, str):
            py_name = identifier.upper()
        else:
            # Support passing BankType.MCHEAD
            py_name = str(identifier).upper()
            
        # Pass name to C++ to get the valid pointer
        cdef string cpp_name = py_name.encode('utf-8')
        cdef pxd.CppIFamily* fm_ptr = self.cpp_event.getFamily(cpp_name)
        
        # Look up wrapper based on naming convention
        if py_name not in _WRAPPER_MAP:
            # Safe fallback: C++ has it, but Cython wrapper is missing
            raise NotImplementedError(f"C++ family '{py_name}' exists, but no 'Py{py_name}' wrapper was found.")

        return wrap_family(fm_ptr, self, _WRAPPER_MAP[py_name])

    def getFamilies(self):
        """
        Returns a tuple of Family objects for all bank families in this event.
        """
        if self._cached_families is not None:
            return self._cached_families

        cdef vector[pair[string, pxd.CppIFamily*]] cpp_fams = self.cpp_event.getFamilies()
        cdef size_t i
        cdef string cpp_name
        cdef pxd.CppIFamily* ptr
        cdef str py_name
        
        temp_list = []

        for i in range(cpp_fams.size()):
            cpp_name = cpp_fams[i].first
            ptr = cpp_fams[i].second
            py_name = cpp_name.decode('utf-8')

            if py_name in _WRAPPER_MAP:
                fam_obj = wrap_family(ptr, self, _WRAPPER_MAP[py_name])
                temp_list.append(fam_obj)
        
        self._cached_families = tuple(temp_list)
                
        return self._cached_families

    @staticmethod
    def getKnownBankNames():
        """
        Returns the authoritative list of bank names defined in the C++ core.
        """
        cdef vector[string] cpp_names = pxd.CppJazelleEvent.getKnownBankNames()
        return [n.decode('utf-8') for n in cpp_names]    

# --- JazelleFile Wrapper ---

cdef class JazelleFile:
    """
    Wrapper for the C++ JazelleFile class.
    Handles opening files, reading records, and random access.
    """
    cdef unique_ptr[pxd.CppJazelleFile] cpp_obj
    
    def __cinit__(self, filepath):
        cdef string s_filepath = filepath.encode('UTF-8')
        try:
            self.cpp_obj.reset(new pxd.CppJazelleFile(s_filepath))
        except Exception as e:
            raise RuntimeError(f"Error opening Jazelle file: {e}")
            
    def nextRecord(self, JazelleEvent event):
        """Reads the next logical record into the provided event object."""
        return self.cpp_obj.get().nextRecord(event.cpp_event)

    def readEvent(self, int index, JazelleEvent event):
        """Reads the event at the specified index into the provided object."""
        return self.cpp_obj.get().readEvent(index, event.cpp_event)

    def getTotalEvents(self):
        """Returns the total number of events in the file."""
        return self.cpp_obj.get().getTotalEvents()
        
    @property
    def fileName(self):
        """The internal filename from the Jazelle header."""
        return self.cpp_obj.get().getFileName().decode('UTF-8')

    @property
    def creationDate(self):
        """File creation date as a datetime object."""
        return cpp_to_py_time(self.cpp_obj.get().getCreationDate())

    @property
    def modifiedDate(self):
        """File modification date as a datetime object."""
        return cpp_to_py_time(self.cpp_obj.get().getModifiedDate())

    @property
    def lastRecordType(self):
        """The type of the last read record (e.g., 'MINIDST')."""
        return self.cpp_obj.get().getLastRecordType().decode('UTF-8')

    def __len__(self):
        return self.getTotalEvents()

    def __iter__(self):
        cdef int total = self.getTotalEvents()
        cdef int i
        for i in range(total):
            event = JazelleEvent()
            if self.readEvent(i, event):
                yield event
            else:
                raise IndexError(f"Failed to read event at index {i}")

    def __getitem__(self, int index):
        if index < 0:
            index += len(self)
        
        event = JazelleEvent()
        if self.readEvent(index, event):
            return event
        else:
            raise IndexError(f"Event index {index} out of range for file with {len(self)} events.")