# jazelle_reader/bindings/jazelle_cython.pyx
# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3

"""
Jazelle Cython Bindings
=======================

This module provides high-performance Python bindings for the C++ Jazelle
library. It utilizes direct NumPy memory access to maximize throughput
when converting binary bank data into analysis-ready arrays.
"""

import sys
import inspect
import cython
import datetime
from libc.stdint cimport int16_t, int32_t
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.string_view cimport string_view
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.chrono cimport system_clock, to_time_t

# --- NumPy Setup ---
import numpy as np
cimport numpy as cnp

# Initialize NumPy API (Essential for cimport to work)
cnp.import_array()

# Import the .pxd definitions
cimport jazelle_cython as pxd


# ==============================================================================
# Helper Functions
# ==============================================================================

cdef object cpp_to_py_time(system_clock.time_point tp):
    """
    Converts a C++ system_clock::time_point to a Python datetime object.

    Returns:
        datetime.datetime or None if the timestamp is zero/invalid.
    """
    cdef long time_t_val
    try:
        time_t_val = to_time_t(tp)
        if time_t_val == 0:
            return None
        return datetime.datetime.fromtimestamp(float(time_t_val))
    except (ValueError, OSError):
        return None


# ==============================================================================
# Internal Wrapper Utilities
# ==============================================================================

cdef class Family:
    """
    Wraps a C++ Family template, providing list-like access to Banks.

    This class acts as a container for all banks of a specific type within
    a single event.
    """
    cdef pxd.CppIFamily* _ptr
    cdef JazelleEvent _event_ref
    cdef type _wrapper_class

    def __init__(self):
        raise TypeError("Cannot instantiate Family directly. Use JazelleEvent accessors.")

    def __len__(self):
        return self._ptr.size()

    def __repr__(self):
        return f"<Family '{self.name}' size={len(self)}>"

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
        """Returns the name of the bank family (e.g., 'MCPART')."""
        return (<bytes>self._ptr.name()).decode('UTF-8')

    @property
    def size(self):
        """Returns the number of banks in this family."""
        return self._ptr.size()

    def find(self, int id):
        """
        Finds a bank by its ID.

        Args:
            id (int): The unique bank ID to search for.

        Returns:
            Bank or None: The wrapper object if found, else None.
        """
        cdef pxd.CppBank* raw_ptr = self._ptr.find(id)
        return wrap_bank(raw_ptr, self._event_ref, self._wrapper_class)

    cpdef object to_dict(self, str orient='list'):
        """
        Converts the family data to a Python dictionary.

        Args:
            orient (str):
                'list' (default): Returns column-oriented data (optimized with NumPy).
                                  Format: { 'col_name': np.array([...]) }
                'records': Returns row-oriented data (list of dicts).
                           Format: [ {'col': val}, ... ]
        """
        if orient == 'records':
            return self._to_dict_records()
        elif orient == 'list':
            return self._to_dict_list()
        else:
            raise ValueError(f"Invalid orient '{orient}'. Must be 'records' or 'list'.")

    cdef list _to_dict_records(self):
        """Row-based conversion using Flyweight optimization."""
        cdef list result = []
        cdef size_t size = self._ptr.size()
        
        if size == 0:
            return result
            
        cdef size_t i
        cdef pxd.CppBank* raw_ptr
        # Allocate one Python wrapper to reuse (flyweight)
        cdef Bank flyweight = self._wrapper_class.__new__(self._wrapper_class)
        flyweight._ptr = NULL
        flyweight._event_ref = self._event_ref
        
        for i in range(size):
            raw_ptr = self._ptr.at(i)
            if raw_ptr != NULL:
                flyweight._ptr = raw_ptr
                result.append(flyweight.to_dict())
                
        return result

    cdef dict _to_dict_list(self):
        """
        High-performance columnar conversion.
        Delegates to the specific Bank's static NumPy extractor.
        """
        return self._wrapper_class.bulk_extract(self)


cdef object wrap_family(pxd.CppIFamily* ptr, JazelleEvent event, type py_class):
    cdef Family obj = Family.__new__(Family)
    obj._ptr = ptr
    obj._event_ref = event
    obj._wrapper_class = py_class
    return obj


# ==============================================================================
# Bank Base Class
# ==============================================================================

cdef dict _WRAPPER_MAP = {}

cdef class Bank:
    """
    Abstract base wrapper for all Jazelle Bank types.
    """
    cdef pxd.CppBank* _ptr
    cdef JazelleEvent _event_ref

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.id}>"
    
    @property
    def id(self):
        """The unique ID of the bank."""
        return self._ptr.getId()

    cpdef dict to_dict(self):
        """
        Converts a single bank instance to a Python dictionary.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} missing to_dict()")

    @staticmethod
    def bulk_extract(Family family):
        """
        Extracts all data for a family into a dictionary of NumPy arrays.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("bulk_extract not implemented")


cdef object wrap_bank(pxd.CppBank* ptr, JazelleEvent event, type py_class):
    if ptr == NULL:
        return None
    cdef Bank obj = py_class.__new__(py_class)
    obj._ptr = ptr
    obj._event_ref = event
    return obj


# ==============================================================================
# Helper Structs (PIDVEC, CRIDHYP)
# ==============================================================================

cdef class PIDVEC:
    """Wrapper for PIDVEC (Particle ID Likelihood Vector)."""
    cdef pxd.CppPIDVEC* _ptr
    cdef JazelleEvent _event_ref
    
    def __init__(self):
        raise TypeError("Cannot instantiate PIDVEC directly.")

    def __repr__(self):
        p = self._ptr
        return f"<PIDVEC e={p.e:.2f} mu={p.mu:.2f} pi={p.pi:.2f} k={p.k:.2f} p={p.p:.2f}>"
    
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
        return {'e': self._ptr.e, 'mu': self._ptr.mu, 'pi': self._ptr.pi, 'k': self._ptr.k, 'p': self._ptr.p}


cdef class CRIDHYP:
    """Wrapper for CRIDHYP (CRID Hypothesis Data)."""
    cdef pxd.CppCRIDHYP* _ptr
    cdef JazelleEvent _event_ref
    
    def __init__(self):
        raise TypeError("Cannot instantiate CRIDHYP directly.")

    def __repr__(self):
        return f"<CRIDHYP full={self.is_full} rc={self.rc} nhits={self.nhits}>"
    
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
            'is_full': self._ptr.m_full, 'rc': self._ptr.rc, 'nhits': self._ptr.nhits,
            'besthyp': self._ptr.besthyp, 'nhexp': self._ptr.nhexp, 'nhfnd': self._ptr.nhfnd,
            'nhbkg': self._ptr.nhbkg, 'mskphot': self._ptr.mskphot
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


# ==============================================================================
# Concrete Banks
# ==============================================================================

cdef class IEVENTH(Bank):
    """Wrapper for IEVENTH (Event Header) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        p = <pxd.CppIEVENTH*>self._ptr
        return f"<IEVENTH run={p.run} event={p.event} type={p.evttype}>"

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
    def evttime(self): return cpp_to_py_time((<pxd.CppIEVENTH*>self._ptr).evttime)
    @property
    def header(self): return (<pxd.CppIEVENTH*>self._ptr).header

    cpdef dict to_dict(self):
        cdef pxd.CppIEVENTH* ptr = <pxd.CppIEVENTH*>self._ptr
        return {
            'id': ptr.getId(), 'header': ptr.header, 'run': ptr.run,
            'event': ptr.event, 'evttype': ptr.evttype, 'trigger': ptr.trigger,
            'weight': ptr.weight, 'evttime': cpp_to_py_time(ptr.evttime)
        }

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef pxd.CppIFamily* fam_ptr = family._ptr
        
        # NumPy Buffers
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_header = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_run = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_event = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_evttype = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_trigger = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_weight = np.empty(count, dtype=np.float32)
        cdef list l_evttime = [None] * count

        # Raw Pointers
        cdef size_t i
        cdef pxd.CppIEVENTH* ptr
        cdef int32_t* r_id = <int32_t*>arr_id.data
        cdef int32_t* r_header = <int32_t*>arr_header.data
        cdef int32_t* r_run = <int32_t*>arr_run.data
        cdef int32_t* r_event = <int32_t*>arr_event.data
        cdef int32_t* r_evttype = <int32_t*>arr_evttype.data
        cdef int32_t* r_trigger = <int32_t*>arr_trigger.data
        cdef float* r_weight = <float*>arr_weight.data

        for i in range(count):
            ptr = <pxd.CppIEVENTH*>fam_ptr.at(i)
            r_id[i] = ptr.getId()
            r_header[i] = ptr.header
            r_run[i] = ptr.run
            r_event[i] = ptr.event
            r_evttype[i] = ptr.evttype
            r_trigger[i] = ptr.trigger
            r_weight[i] = ptr.weight
            l_evttime[i] = cpp_to_py_time(ptr.evttime)

        return {
            'id': arr_id, 'header': arr_header, 'run': arr_run,
            'event': arr_event, 'evttype': arr_evttype, 'trigger': arr_trigger,
            'weight': arr_weight, 'evttime': l_evttime
        }


cdef class MCHEAD(Bank):
    """Wrapper for MCHEAD (Monte Carlo Header) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        p = <pxd.CppMCHEAD*>self._ptr
        return f"<MCHEAD id={p.getId()} ntot={p.ntot} origin={p.origin}>"

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
            'id': ptr.getId(), 'ntot': ptr.ntot, 'origin': ptr.origin,
            'ipx': ptr.ipx, 'ipy': ptr.ipy, 'ipz': ptr.ipz
        }

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef pxd.CppIFamily* fam_ptr = family._ptr
        
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_ntot = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_origin = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_ipx = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_ipy = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_ipz = np.empty(count, dtype=np.float32)

        cdef size_t i
        cdef pxd.CppMCHEAD* ptr
        cdef int32_t* r_id = <int32_t*>arr_id.data
        cdef int32_t* r_ntot = <int32_t*>arr_ntot.data
        cdef int32_t* r_origin = <int32_t*>arr_origin.data
        cdef float* r_ipx = <float*>arr_ipx.data
        cdef float* r_ipy = <float*>arr_ipy.data
        cdef float* r_ipz = <float*>arr_ipz.data

        for i in range(count):
            ptr = <pxd.CppMCHEAD*>fam_ptr.at(i)
            r_id[i] = ptr.getId()
            r_ntot[i] = ptr.ntot
            r_origin[i] = ptr.origin
            r_ipx[i] = ptr.ipx
            r_ipy[i] = ptr.ipy
            r_ipz[i] = ptr.ipz
            
        return {
            'id': arr_id, 'ntot': arr_ntot, 'origin': arr_origin,
            'ipx': arr_ipx, 'ipy': arr_ipy, 'ipz': arr_ipz
        }


cdef class MCPART(Bank):
    """Wrapper for MCPART (Monte Carlo Particle) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        p = <pxd.CppMCPART*>self._ptr
        return f"<MCPART id={p.getId()} ptype={p.ptype} e={p.e:.2f}>"

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
            'id': ptr.getId(), 'e': ptr.e, 'ptot': ptr.ptot,
            'ptype': ptr.ptype, 'charge': ptr.charge,
            'origin': ptr.origin, 'parent_id': ptr.parent_id,
            'p': [ptr.p[i] for i in range(3)],
            'xt': [ptr.xt[i] for i in range(3)]
        }

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef pxd.CppIFamily* fam_ptr = family._ptr
        
        # Primitives
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_e = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_ptot = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_ptype = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_charge = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_origin = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_parent = np.empty(count, dtype=np.int32)
        
        # Arrays (3-vectors)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_p = np.empty((count, 3), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_xt = np.empty((count, 3), dtype=np.float32)

        cdef size_t i
        cdef pxd.CppMCPART* ptr
        cdef int32_t* r_id = <int32_t*>arr_id.data
        cdef float* r_e = <float*>arr_e.data
        cdef float* r_ptot = <float*>arr_ptot.data
        cdef int32_t* r_ptype = <int32_t*>arr_ptype.data
        cdef float* r_charge = <float*>arr_charge.data
        cdef int32_t* r_origin = <int32_t*>arr_origin.data
        cdef int32_t* r_parent = <int32_t*>arr_parent.data
        
        cdef float* r_p_base = <float*>arr_p.data
        cdef float* r_xt_base = <float*>arr_xt.data

        for i in range(count):
            ptr = <pxd.CppMCPART*>fam_ptr.at(i)
            r_id[i] = ptr.getId()
            r_e[i] = ptr.e
            r_ptot[i] = ptr.ptot
            r_ptype[i] = ptr.ptype
            r_charge[i] = ptr.charge
            r_origin[i] = ptr.origin
            r_parent[i] = ptr.parent_id
            
            # Fast array copy (memcpy)
            memcpy(r_p_base + i*3, &ptr.p[0], 3 * sizeof(float))
            memcpy(r_xt_base + i*3, &ptr.xt[0], 3 * sizeof(float))

        return {
            'id': arr_id, 'e': arr_e, 'ptot': arr_ptot, 'ptype': arr_ptype, 
            'charge': arr_charge, 'origin': arr_origin, 'parent_id': arr_parent,
            'p': arr_p, 'xt': arr_xt
        }


cdef class PHPSUM(Bank):
    """Wrapper for PHPSUM (Physics Particle Summary) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        p = <pxd.CppPHPSUM*>self._ptr
        return f"<PHPSUM id={p.getId()} charge={p.charge} ptot={p.getPTot():.2f}>"

    # Properties
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
            'id': ptr.getId(), 'px': ptr.px, 'py': ptr.py, 'pz': ptr.pz,
            'x': ptr.x, 'y': ptr.y, 'z': ptr.z,
            'charge': ptr.charge, 'status': ptr.status, 'ptot': ptr.getPTot()
        }

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef pxd.CppIFamily* fam_ptr = family._ptr
        
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_px = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_py = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_pz = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_x = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_y = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_z = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_charge = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_status = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] arr_ptot = np.empty(count, dtype=np.float64)

        cdef size_t i
        cdef pxd.CppPHPSUM* ptr
        cdef int32_t* r_id = <int32_t*>arr_id.data
        cdef float* r_px = <float*>arr_px.data
        cdef float* r_py = <float*>arr_py.data
        cdef float* r_pz = <float*>arr_pz.data
        cdef float* r_x = <float*>arr_x.data
        cdef float* r_y = <float*>arr_y.data
        cdef float* r_z = <float*>arr_z.data
        cdef float* r_charge = <float*>arr_charge.data
        cdef int32_t* r_status = <int32_t*>arr_status.data
        cdef double* r_ptot = <double*>arr_ptot.data

        for i in range(count):
            ptr = <pxd.CppPHPSUM*>fam_ptr.at(i)
            r_id[i] = ptr.getId()
            r_px[i] = ptr.px
            r_py[i] = ptr.py
            r_pz[i] = ptr.pz
            r_x[i] = ptr.x
            r_y[i] = ptr.y
            r_z[i] = ptr.z
            r_charge[i] = ptr.charge
            r_status[i] = ptr.status
            r_ptot[i] = ptr.getPTot()

        return {
            'id': arr_id, 'px': arr_px, 'py': arr_py, 'pz': arr_pz,
            'x': arr_x, 'y': arr_y, 'z': arr_z, 'charge': arr_charge,
            'status': arr_status, 'ptot': arr_ptot
        }


cdef class PHCHRG(Bank):
    """Wrapper for PHCHRG (Charged Track) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        p = <pxd.CppPHCHRG*>self._ptr
        return f"<PHCHRG id={p.getId()} charge={p.charge} nhit={p.nhit}>"

    # Properties
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

    def to_dict(self):
        cdef pxd.CppPHCHRG* ptr = <pxd.CppPHCHRG*>self._ptr
        return {
            'id': ptr.getId(), 'bnorm': ptr.bnorm, 'impact': ptr.impact, 
            'b3norm': ptr.b3norm, 'impact3': ptr.impact3,
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

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef pxd.CppIFamily* fam_ptr = family._ptr
        
        # Primitive Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_bnorm = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_impact = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_b3norm = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_impact3 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_charge = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_smwstat = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_status = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_tkpar0 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_length = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_chi2dt = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_imc = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_ndfdt = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nhit = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nhite = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nhitp = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nmisht = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nwrght = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nhitv = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_chi2 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_chi2v = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_vxdhit = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_mustat = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_estat = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_dedx = np.empty(count, dtype=np.int32)
        
        # Array Allocations
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_hlxpar = np.empty((count, 6), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_dhlxpar = np.empty((count, 15), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_tkpar = np.empty((count, 5), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_dtkpar = np.empty((count, 15), dtype=np.float32)

        # Loop
        cdef size_t i
        cdef pxd.CppPHCHRG* ptr
        # Pointers to primitive buffers
        cdef int32_t* r_id = <int32_t*>arr_id.data
        cdef float* r_bnorm = <float*>arr_bnorm.data
        cdef float* r_impact = <float*>arr_impact.data
        cdef float* r_b3norm = <float*>arr_b3norm.data
        cdef float* r_impact3 = <float*>arr_impact3.data
        cdef int16_t* r_charge = <int16_t*>arr_charge.data
        cdef int16_t* r_smwstat = <int16_t*>arr_smwstat.data
        cdef int32_t* r_status = <int32_t*>arr_status.data
        cdef float* r_tkpar0 = <float*>arr_tkpar0.data
        cdef float* r_length = <float*>arr_length.data
        cdef float* r_chi2dt = <float*>arr_chi2dt.data
        cdef int16_t* r_imc = <int16_t*>arr_imc.data
        cdef int16_t* r_ndfdt = <int16_t*>arr_ndfdt.data
        cdef int16_t* r_nhit = <int16_t*>arr_nhit.data
        cdef int16_t* r_nhite = <int16_t*>arr_nhite.data
        cdef int16_t* r_nhitp = <int16_t*>arr_nhitp.data
        cdef int16_t* r_nmisht = <int16_t*>arr_nmisht.data
        cdef int16_t* r_nwrght = <int16_t*>arr_nwrght.data
        cdef int16_t* r_nhitv = <int16_t*>arr_nhitv.data
        cdef float* r_chi2 = <float*>arr_chi2.data
        cdef float* r_chi2v = <float*>arr_chi2v.data
        cdef int32_t* r_vxdhit = <int32_t*>arr_vxdhit.data
        cdef int16_t* r_mustat = <int16_t*>arr_mustat.data
        cdef int16_t* r_estat = <int16_t*>arr_estat.data
        cdef int32_t* r_dedx = <int32_t*>arr_dedx.data
        
        # Pointers to array buffers
        cdef float* r_hlxpar = <float*>arr_hlxpar.data
        cdef float* r_dhlxpar = <float*>arr_dhlxpar.data
        cdef float* r_tkpar = <float*>arr_tkpar.data
        cdef float* r_dtkpar = <float*>arr_dtkpar.data

        for i in range(count):
            ptr = <pxd.CppPHCHRG*>fam_ptr.at(i)
            r_id[i] = ptr.getId()
            r_bnorm[i] = ptr.bnorm
            r_impact[i] = ptr.impact
            r_b3norm[i] = ptr.b3norm
            r_impact3[i] = ptr.impact3
            r_charge[i] = ptr.charge
            r_smwstat[i] = ptr.smwstat
            r_status[i] = ptr.status
            r_tkpar0[i] = ptr.tkpar0
            r_length[i] = ptr.length
            r_chi2dt[i] = ptr.chi2dt
            r_imc[i] = ptr.imc
            r_ndfdt[i] = ptr.ndfdt
            r_nhit[i] = ptr.nhit
            r_nhite[i] = ptr.nhite
            r_nhitp[i] = ptr.nhitp
            r_nmisht[i] = ptr.nmisht
            r_nwrght[i] = ptr.nwrght
            r_nhitv[i] = ptr.nhitv
            r_chi2[i] = ptr.chi2
            r_chi2v[i] = ptr.chi2v
            r_vxdhit[i] = ptr.vxdhit
            r_mustat[i] = ptr.mustat
            r_estat[i] = ptr.estat
            r_dedx[i] = ptr.dedx
            
            # Fast Copies
            memcpy(r_hlxpar + i*6, &ptr.hlxpar[0], 6 * sizeof(float))
            memcpy(r_dhlxpar + i*15, &ptr.dhlxpar[0], 15 * sizeof(float))
            memcpy(r_tkpar + i*5, &ptr.tkpar[0], 5 * sizeof(float))
            memcpy(r_dtkpar + i*15, &ptr.dtkpar[0], 15 * sizeof(float))

        return {
            'id': arr_id, 'bnorm': arr_bnorm, 'impact': arr_impact, 'b3norm': arr_b3norm,
            'impact3': arr_impact3, 'charge': arr_charge, 'smwstat': arr_smwstat,
            'status': arr_status, 'tkpar0': arr_tkpar0, 'length': arr_length,
            'chi2dt': arr_chi2dt, 'imc': arr_imc, 'ndfdt': arr_ndfdt, 'nhit': arr_nhit,
            'nhite': arr_nhite, 'nhitp': arr_nhitp, 'nmisht': arr_nmisht,
            'nwrght': arr_nwrght, 'nhitv': arr_nhitv, 'chi2': arr_chi2,
            'chi2v': arr_chi2v, 'vxdhit': arr_vxdhit, 'mustat': arr_mustat,
            'estat': arr_estat, 'dedx': arr_dedx, 'hlxpar': arr_hlxpar,
            'dhlxpar': arr_dhlxpar, 'tkpar': arr_tkpar, 'dtkpar': arr_dtkpar
        }


cdef class PHKLUS(Bank):
    """Wrapper for PHKLUS (Calorimeter Cluster) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        p = <pxd.CppPHKLUS*>self._ptr
        return f"<PHKLUS id={p.getId()} status={p.status} eraw={p.eraw:.2f}>"

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
            'id': ptr.getId(), 'status': ptr.status, 'eraw': ptr.eraw,
            'cth': ptr.cth, 'wcth': ptr.wcth, 'phi': ptr.phi, 'wphi': ptr.wphi,
            'nhit2': ptr.nhit2, 'cth2': ptr.cth2, 'wcth2': ptr.wcth2,
            'phi2': ptr.phi2, 'whphi2': ptr.whphi2,
            'nhit3': ptr.nhit3, 'cth3': ptr.cth3, 'wcth3': ptr.wcth3,
            'phi3': ptr.phi3, 'wphi3': ptr.wphi3,
            'elayer': [ptr.elayer[i] for i in range(8)]
        }

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef pxd.CppIFamily* fam_ptr = family._ptr
        
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_status = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_eraw = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_cth = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_wcth = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_phi = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_wphi = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_nhit2 = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_cth2 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_wcth2 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_phi2 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_whphi2 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_nhit3 = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_cth3 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_wcth3 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_phi3 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_wphi3 = np.empty(count, dtype=np.float32)
        
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_elayer = np.empty((count, 8), dtype=np.float32)

        cdef size_t i
        cdef pxd.CppPHKLUS* ptr
        cdef int32_t* r_id = <int32_t*>arr_id.data
        cdef int32_t* r_status = <int32_t*>arr_status.data
        cdef float* r_eraw = <float*>arr_eraw.data
        cdef float* r_cth = <float*>arr_cth.data
        cdef float* r_wcth = <float*>arr_wcth.data
        cdef float* r_phi = <float*>arr_phi.data
        cdef float* r_wphi = <float*>arr_wphi.data
        cdef int32_t* r_nhit2 = <int32_t*>arr_nhit2.data
        cdef float* r_cth2 = <float*>arr_cth2.data
        cdef float* r_wcth2 = <float*>arr_wcth2.data
        cdef float* r_phi2 = <float*>arr_phi2.data
        cdef float* r_whphi2 = <float*>arr_whphi2.data
        cdef int32_t* r_nhit3 = <int32_t*>arr_nhit3.data
        cdef float* r_cth3 = <float*>arr_cth3.data
        cdef float* r_wcth3 = <float*>arr_wcth3.data
        cdef float* r_phi3 = <float*>arr_phi3.data
        cdef float* r_wphi3 = <float*>arr_wphi3.data
        
        cdef float* r_elayer = <float*>arr_elayer.data

        for i in range(count):
            ptr = <pxd.CppPHKLUS*>fam_ptr.at(i)
            r_id[i] = ptr.getId()
            r_status[i] = ptr.status
            r_eraw[i] = ptr.eraw
            r_cth[i] = ptr.cth
            r_wcth[i] = ptr.wcth
            r_phi[i] = ptr.phi
            r_wphi[i] = ptr.wphi
            r_nhit2[i] = ptr.nhit2
            r_cth2[i] = ptr.cth2
            r_wcth2[i] = ptr.wcth2
            r_phi2[i] = ptr.phi2
            r_whphi2[i] = ptr.whphi2
            r_nhit3[i] = ptr.nhit3
            r_cth3[i] = ptr.cth3
            r_wcth3[i] = ptr.wcth3
            r_phi3[i] = ptr.phi3
            r_wphi3[i] = ptr.wphi3
            
            memcpy(r_elayer + i*8, &ptr.elayer[0], 8 * sizeof(float))

        return {
            'id': arr_id, 'status': arr_status, 'eraw': arr_eraw,
            'cth': arr_cth, 'wcth': arr_wcth, 'phi': arr_phi, 'wphi': arr_wphi,
            'nhit2': arr_nhit2, 'cth2': arr_cth2, 'wcth2': arr_wcth2,
            'phi2': arr_phi2, 'whphi2': arr_whphi2,
            'nhit3': arr_nhit3, 'cth3': arr_cth3, 'wcth3': arr_wcth3,
            'phi3': arr_phi3, 'wphi3': arr_wphi3,
            'elayer': arr_elayer
        }


cdef class PHWIC(Bank):
    """Wrapper for PHWIC (Warm Iron Calorimeter) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        p = <pxd.CppPHWIC*>self._ptr
        return f"<PHWIC id={p.getId()} nhit={p.nhit} phwicid={p.phwicid}>"

    # Properties (accessors)
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
            'id': ptr.getId(), 'idstat': ptr.idstat, 'nhit': ptr.nhit, 'nhit45': ptr.nhit45,
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

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef pxd.CppIFamily* fam_ptr = family._ptr
        
        # Scalar Buffers
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_idstat = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nhit = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nhit45 = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_npat = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nhitpat = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_syshit = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_qpinit = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_t1 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_t2 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_t3 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_hitmiss = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_itrlen = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nlayexp = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nlaybey = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_missprob = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_phwicid = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nhitshar = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nother = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_hitsused = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_chi2 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_ndf = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_punfit = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_matchChi2 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_matchNdf = np.empty(count, dtype=np.int16)
        
        # Array Buffers
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_pref1 = np.empty((count, 3), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_pfit = np.empty((count, 4), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] arr_dpfit = np.empty((count, 10), dtype=np.float32)

        cdef size_t i
        cdef pxd.CppPHWIC* ptr
        
        # Primitive Pointers
        cdef int32_t* r_id = <int32_t*>arr_id.data
        cdef int16_t* r_idstat = <int16_t*>arr_idstat.data
        cdef int16_t* r_nhit = <int16_t*>arr_nhit.data
        cdef int16_t* r_nhit45 = <int16_t*>arr_nhit45.data
        cdef int16_t* r_npat = <int16_t*>arr_npat.data
        cdef int16_t* r_nhitpat = <int16_t*>arr_nhitpat.data
        cdef int16_t* r_syshit = <int16_t*>arr_syshit.data
        cdef float* r_qpinit = <float*>arr_qpinit.data
        cdef float* r_t1 = <float*>arr_t1.data
        cdef float* r_t2 = <float*>arr_t2.data
        cdef float* r_t3 = <float*>arr_t3.data
        cdef int32_t* r_hitmiss = <int32_t*>arr_hitmiss.data
        cdef float* r_itrlen = <float*>arr_itrlen.data
        cdef int16_t* r_nlayexp = <int16_t*>arr_nlayexp.data
        cdef int16_t* r_nlaybey = <int16_t*>arr_nlaybey.data
        cdef float* r_missprob = <float*>arr_missprob.data
        cdef int32_t* r_phwicid = <int32_t*>arr_phwicid.data
        cdef int16_t* r_nhitshar = <int16_t*>arr_nhitshar.data
        cdef int16_t* r_nother = <int16_t*>arr_nother.data
        cdef int32_t* r_hitsused = <int32_t*>arr_hitsused.data
        cdef float* r_chi2 = <float*>arr_chi2.data
        cdef int16_t* r_ndf = <int16_t*>arr_ndf.data
        cdef int16_t* r_punfit = <int16_t*>arr_punfit.data
        cdef float* r_matchChi2 = <float*>arr_matchChi2.data
        cdef int16_t* r_matchNdf = <int16_t*>arr_matchNdf.data
        
        # Array Pointers
        cdef float* r_pref1 = <float*>arr_pref1.data
        cdef float* r_pfit = <float*>arr_pfit.data
        cdef float* r_dpfit = <float*>arr_dpfit.data

        for i in range(count):
            ptr = <pxd.CppPHWIC*>fam_ptr.at(i)
            r_id[i] = ptr.getId()
            r_idstat[i] = ptr.idstat
            r_nhit[i] = ptr.nhit
            r_nhit45[i] = ptr.nhit45
            r_npat[i] = ptr.npat
            r_nhitpat[i] = ptr.nhitpat
            r_syshit[i] = ptr.syshit
            r_qpinit[i] = ptr.qpinit
            r_t1[i] = ptr.t1
            r_t2[i] = ptr.t2
            r_t3[i] = ptr.t3
            r_hitmiss[i] = ptr.hitmiss
            r_itrlen[i] = ptr.itrlen
            r_nlayexp[i] = ptr.nlayexp
            r_nlaybey[i] = ptr.nlaybey
            r_missprob[i] = ptr.missprob
            r_phwicid[i] = ptr.phwicid
            r_nhitshar[i] = ptr.nhitshar
            r_nother[i] = ptr.nother
            r_hitsused[i] = ptr.hitsused
            r_chi2[i] = ptr.chi2
            r_ndf[i] = ptr.ndf
            r_punfit[i] = ptr.punfit
            r_matchChi2[i] = ptr.matchChi2
            r_matchNdf[i] = ptr.matchNdf
            
            memcpy(r_pref1 + i*3, &ptr.pref1[0], 3 * sizeof(float))
            memcpy(r_pfit + i*4, &ptr.pfit[0], 4 * sizeof(float))
            memcpy(r_dpfit + i*10, &ptr.dpfit[0], 10 * sizeof(float))

        return {
            'id': arr_id, 'idstat': arr_idstat, 'nhit': arr_nhit, 'nhit45': arr_nhit45,
            'npat': arr_npat, 'nhitpat': arr_nhitpat, 'syshit': arr_syshit,
            'qpinit': arr_qpinit, 't1': arr_t1, 't2': arr_t2, 't3': arr_t3,
            'hitmiss': arr_hitmiss, 'itrlen': arr_itrlen, 'nlayexp': arr_nlayexp,
            'nlaybey': arr_nlaybey, 'missprob': arr_missprob, 'phwicid': arr_phwicid,
            'nhitshar': arr_nhitshar, 'nother': arr_nother, 'hitsused': arr_hitsused,
            'chi2': arr_chi2, 'ndf': arr_ndf, 'punfit': arr_punfit,
            'matchChi2': arr_matchChi2, 'matchNdf': arr_matchNdf,
            'pref1': arr_pref1, 'pfit': arr_pfit, 'dpfit': arr_dpfit
        }


cdef class PHCRID(Bank):
    """Wrapper for PHCRID (Cerenkov Ring Imaging) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        p = <pxd.CppPHCRID*>self._ptr
        return f"<PHCRID id={p.getId()} ctlword={p.ctlword} nhits={p.nhits}>"
    
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
            'id': ptr.getId(), 'ctlword': ptr.ctlword, 'norm': ptr.norm,
            'rc': ptr.rc, 'geom': ptr.geom, 'trkp': ptr.trkp, 'nhits': ptr.nhits,
            'liq': self.liq.to_dict(), 'gas': self.gas.to_dict(), 'llik': self.llik.to_dict()
        }

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef pxd.CppIFamily* fam_ptr = family._ptr
        
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_ctlword = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_norm = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_rc = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_geom = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_trkp = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_nhits = np.empty(count, dtype=np.int16)
        
        # Use lists for the nested wrappers
        cdef list l_liq = [None] * count
        cdef list l_gas = [None] * count
        cdef list l_llik = [None] * count

        cdef size_t i
        cdef pxd.CppPHCRID* ptr
        # Pointers
        cdef int32_t* r_id = <int32_t*>arr_id.data
        cdef int32_t* r_ctlword = <int32_t*>arr_ctlword.data
        cdef float* r_norm = <float*>arr_norm.data
        cdef int16_t* r_rc = <int16_t*>arr_rc.data
        cdef int16_t* r_geom = <int16_t*>arr_geom.data
        cdef int16_t* r_trkp = <int16_t*>arr_trkp.data
        cdef int16_t* r_nhits = <int16_t*>arr_nhits.data

        cdef JazelleEvent event_ref = family._event_ref

        for i in range(count):
            ptr = <pxd.CppPHCRID*>fam_ptr.at(i)
            r_id[i] = ptr.getId()
            r_ctlword[i] = ptr.ctlword
            r_norm[i] = ptr.norm
            r_rc[i] = ptr.rc
            r_geom[i] = ptr.geom
            r_trkp[i] = ptr.trkp
            r_nhits[i] = ptr.nhits
            
            # Wrap internal structs
            l_liq[i] = wrap_cridhyp(cython.address(ptr.liq), event_ref).to_dict()
            l_gas[i] = wrap_cridhyp(cython.address(ptr.gas), event_ref).to_dict()
            l_llik[i] = wrap_pidvec(cython.address(ptr.llik), event_ref).to_dict()

        return {
            'id': arr_id, 'ctlword': arr_ctlword, 'norm': arr_norm, 'rc': arr_rc,
            'geom': arr_geom, 'trkp': arr_trkp, 'nhits': arr_nhits,
            'liq': l_liq, 'gas': l_gas, 'llik': l_llik
        }


cdef class PHKTRK(Bank):
    """Wrapper for PHKTRK (Stub) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        return f"<PHKTRK id={self._ptr.getId()}>"

    cpdef dict to_dict(self):
        return {'id': self._ptr.getId()}

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        cdef int32_t* r_id = <int32_t*>arr_id.data
        for i in range(count):
             r_id[i] = (<pxd.CppPHKTRK*>family._ptr.at(i)).getId()
        return {'id': arr_id}


cdef class PHKELID(Bank):
    """Wrapper for PHKELID (Calorimeter/Electron ID) Bank."""
    def __init__(self): raise TypeError("No direct instantiation")

    def __repr__(self):
        p = <pxd.CppPHKELID*>self._ptr
        return f"<PHKELID id={p.getId()} prob={p.prob} isolat={p.isolat:.2f}>"

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
            'id': ptr.getId(), 'phchrg_id': linked_id, 'idstat': ptr.idstat, 'prob': ptr.prob,
            'phi': ptr.phi, 'theta': ptr.theta, 'qp': ptr.qp, 'dphi': ptr.dphi,
            'dtheta': ptr.dtheta, 'dqp': ptr.dqp, 'tphi': ptr.tphi, 'ttheta': ptr.ttheta,
            'isolat': ptr.isolat, 'em1': ptr.em1, 'em12': ptr.em12, 'dem12': ptr.dem12,
            'had1': ptr.had1, 'emphi': ptr.emphi, 'emtheta': ptr.emtheta,
            'phiwid': ptr.phiwid, 'thewid': ptr.thewid, 'em1x1': ptr.em1x1,
            'em2x2a': ptr.em2x2a, 'em2x2b': ptr.em2x2b, 'em3x3a': ptr.em3x3a,
            'em3x3b': ptr.em3x3b
        }

    @staticmethod
    def bulk_extract(Family family):
        cdef size_t count = len(family)
        cdef pxd.CppIFamily* fam_ptr = family._ptr
        
        cdef cnp.ndarray[cnp.int32_t, ndim=1] arr_id = np.empty(count, dtype=np.int32)
        # Use List for nullable IDs to allow None
        cdef list l_phchrg_id = [None] * count
        
        # Primitives
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_idstat = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] arr_prob = np.empty(count, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_phi = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_theta = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_qp = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_dphi = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_dtheta = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_dqp = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_tphi = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_ttheta = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_isolat = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_em1 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_em12 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_dem12 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_had1 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_emphi = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_emtheta = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_phiwid = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_thewid = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_em1x1 = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_em2x2a = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_em2x2b = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_em3x3a = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] arr_em3x3b = np.empty(count, dtype=np.float32)

        cdef size_t i
        cdef pxd.CppPHKELID* ptr
        cdef int32_t* r_id = <int32_t*>arr_id.data
        cdef int16_t* r_idstat = <int16_t*>arr_idstat.data
        cdef int16_t* r_prob = <int16_t*>arr_prob.data
        cdef float* r_phi = <float*>arr_phi.data
        cdef float* r_theta = <float*>arr_theta.data
        cdef float* r_qp = <float*>arr_qp.data
        cdef float* r_dphi = <float*>arr_dphi.data
        cdef float* r_dtheta = <float*>arr_dtheta.data
        cdef float* r_dqp = <float*>arr_dqp.data
        cdef float* r_tphi = <float*>arr_tphi.data
        cdef float* r_ttheta = <float*>arr_ttheta.data
        cdef float* r_isolat = <float*>arr_isolat.data
        cdef float* r_em1 = <float*>arr_em1.data
        cdef float* r_em12 = <float*>arr_em12.data
        cdef float* r_dem12 = <float*>arr_dem12.data
        cdef float* r_had1 = <float*>arr_had1.data
        cdef float* r_emphi = <float*>arr_emphi.data
        cdef float* r_emtheta = <float*>arr_emtheta.data
        cdef float* r_phiwid = <float*>arr_phiwid.data
        cdef float* r_thewid = <float*>arr_thewid.data
        cdef float* r_em1x1 = <float*>arr_em1x1.data
        cdef float* r_em2x2a = <float*>arr_em2x2a.data
        cdef float* r_em2x2b = <float*>arr_em2x2b.data
        cdef float* r_em3x3a = <float*>arr_em3x3a.data
        cdef float* r_em3x3b = <float*>arr_em3x3b.data

        for i in range(count):
            ptr = <pxd.CppPHKELID*>fam_ptr.at(i)
            r_id[i] = ptr.getId()
            if ptr.phchrg != NULL:
                l_phchrg_id[i] = ptr.phchrg.getId()

            r_idstat[i] = ptr.idstat
            r_prob[i] = ptr.prob
            r_phi[i] = ptr.phi
            r_theta[i] = ptr.theta
            r_qp[i] = ptr.qp
            r_dphi[i] = ptr.dphi
            r_dtheta[i] = ptr.dtheta
            r_dqp[i] = ptr.dqp
            r_tphi[i] = ptr.tphi
            r_ttheta[i] = ptr.ttheta
            r_isolat[i] = ptr.isolat
            r_em1[i] = ptr.em1
            r_em12[i] = ptr.em12
            r_dem12[i] = ptr.dem12
            r_had1[i] = ptr.had1
            r_emphi[i] = ptr.emphi
            r_emtheta[i] = ptr.emtheta
            r_phiwid[i] = ptr.phiwid
            r_thewid[i] = ptr.thewid
            r_em1x1[i] = ptr.em1x1
            r_em2x2a[i] = ptr.em2x2a
            r_em2x2b[i] = ptr.em2x2b
            r_em3x3a[i] = ptr.em3x3a
            r_em3x3b[i] = ptr.em3x3b

        return {
            'id': arr_id, 'phchrg_id': l_phchrg_id, 'idstat': arr_idstat,
            'prob': arr_prob, 'phi': arr_phi, 'theta': arr_theta, 'qp': arr_qp,
            'dphi': arr_dphi, 'dtheta': arr_dtheta, 'dqp': arr_dqp, 'tphi': arr_tphi,
            'ttheta': arr_ttheta, 'isolat': arr_isolat, 'em1': arr_em1,
            'em12': arr_em12, 'dem12': arr_dem12, 'had1': arr_had1, 'emphi': arr_emphi,
            'emtheta': arr_emtheta, 'phiwid': arr_phiwid, 'thewid': arr_thewid,
            'em1x1': arr_em1x1, 'em2x2a': arr_em2x2a, 'em2x2b': arr_em2x2b,
            'em3x3a': arr_em3x3a, 'em3x3b': arr_em3x3b
        }


# ==============================================================================
# JazelleEvent Wrapper
# ==============================================================================

cdef class JazelleEvent:
    """
    Wrapper for the C++ JazelleEvent class.
    Holds all bank data for a single event.
    """
    cdef pxd.CppJazelleEvent cpp_event
    cdef tuple _cached_families
    
    def __cinit__(self):
        self._cached_families = None

    def __repr__(self):
        # Try to pull run/event from ieventh if possible
        try:
            return f"<JazelleEvent run={self.ieventh.run} event={self.ieventh.event}>"
        except:
            return "<JazelleEvent>"
        
    def clear(self):
        """Clears all data from the event."""
        self.cpp_event.clear()
        self._cached_families = None
        
    @property
    def ieventh(self): return wrap_bank(cython.address(self.cpp_event.ieventh), self, IEVENTH)
    @property
    def mchead(self): return wrap_family(&self.cpp_event.get[pxd.CppMCHEAD](), self, MCHEAD)
    @property
    def mcpart(self): return wrap_family(&self.cpp_event.get[pxd.CppMCPART](), self, MCPART)
    @property
    def phpsum(self): return wrap_family(&self.cpp_event.get[pxd.CppPHPSUM](), self, PHPSUM)
    @property
    def phchrg(self): return wrap_family(&self.cpp_event.get[pxd.CppPHCHRG](), self, PHCHRG)
    @property
    def phklus(self): return wrap_family(&self.cpp_event.get[pxd.CppPHKLUS](), self, PHKLUS)
    @property
    def phwic(self): return wrap_family(&self.cpp_event.get[pxd.CppPHWIC](), self, PHWIC)
    @property
    def phcrid(self): return wrap_family(&self.cpp_event.get[pxd.CppPHCRID](), self, PHCRID)
    @property
    def phktrk(self): return wrap_family(&self.cpp_event.get[pxd.CppPHKTRK](), self, PHKTRK)
    @property
    def phkelid(self): return wrap_family(&self.cpp_event.get[pxd.CppPHKELID](), self, PHKELID)

    def getFamily(self, identifier):
        """
        Retrieves a bank family by name.

        Args:
            identifier (str): The name of the family (e.g., "MCPART").
        """
        cdef str py_name
        if isinstance(identifier, str):
            py_name = identifier.upper()
        else:
            py_name = str(identifier).upper()
            
        cdef string cpp_name = py_name.encode('utf-8')
        cdef pxd.CppIFamily* fm_ptr = self.cpp_event.getFamily(cpp_name)
        
        if py_name not in _WRAPPER_MAP:
            raise NotImplementedError(f"C++ family '{py_name}' exists, but no wrapper found.")

        return wrap_family(fm_ptr, self, _WRAPPER_MAP[py_name])

    def getFamilies(self):
        """Returns a tuple of all Family objects present in this event."""
        if self._cached_families is not None:
            return self._cached_families
    
        cdef vector[pxd.CppIFamily*] cpp_fams = self.cpp_event.getFamilies()
        cdef size_t i
        cdef pxd.CppIFamily* ptr
        cdef str py_name
        
        temp_list = []
        for i in range(cpp_fams.size()):
            ptr = cpp_fams[i]
            py_name = ptr.name().decode('utf-8')
            if py_name in _WRAPPER_MAP:
                temp_list.append(wrap_family(ptr, self, _WRAPPER_MAP[py_name]))
        
        self._cached_families = tuple(temp_list)
        return self._cached_families

    @staticmethod
    def getKnownBankNames():
        """Returns the authoritative list of bank names defined in the C++ core."""
        cdef vector[string_view] cpp_names = pxd.getKnownBankNames()
        return [n.decode('utf-8') for n in cpp_names]

    def to_dict(self, str orient='list', bint skip_empty=True):
        """
        Serializes the entire event into a dictionary.
        """
        cdef dict data = {}

        data["IEVENTH"] = self.ieventh.to_dict()

        for family in self.getFamilies():
            if not skip_empty or len(family) > 0:
                data[family.name] = family.to_dict(orient=orient)
        return data


# ==============================================================================
# JazelleFile Wrapper
# ==============================================================================

cdef class JazelleFile:
    """
    Wrapper for the C++ JazelleFile class.
    Handles opening files, reading records, and random access.
    Implements Context Manager protocol.
    """
    cdef unique_ptr[pxd.CppJazelleFile] cpp_obj
    
    def __cinit__(self, filepath):
        cdef string s_filepath = filepath.encode('UTF-8')
        try:
            self.cpp_obj.reset(new pxd.CppJazelleFile(s_filepath))
        except Exception as e:
            raise RuntimeError(f"Error opening Jazelle file: {e}")

    def __enter__(self):
        """Context Manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager exit. Ensures file is closed."""
        self.close()

    def close(self):
        """Explicitly closes the file and frees resources."""
        self.cpp_obj.reset()

    def read(self):
        """Reads and returns the next event. Returns None at End of File."""
        cdef JazelleEvent event = JazelleEvent()
        if self.cpp_obj.get().nextRecord(event.cpp_event):
            return event
        return None
            
    def nextRecord(self, JazelleEvent event):
        """Reads the next logical record into the provided event object."""
        return self.cpp_obj.get().nextRecord(event.cpp_event)

    def readEvent(self, int index, JazelleEvent event):
        """Reads the event at the specified index."""
        return self.cpp_obj.get().readEvent(index, event.cpp_event)

    def getTotalEvents(self):
        """Returns the total number of events in the file."""
        return self.cpp_obj.get().getTotalEvents()
        
    @property
    def fileName(self):
        """The internal filename from the Jazelle header."""
        return (<bytes>self.cpp_obj.get().getFileName()).decode('UTF-8')

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
        return (<bytes>self.cpp_obj.get().getLastRecordType()).decode('UTF-8')

    def rewind(self):
        """Resets the file pointer to the first event."""
        self.cpp_obj.get().rewind()

    def __len__(self):
        return self.getTotalEvents()

    def __iter__(self):
        """Iterates through events sequentially."""
        self.rewind()
        cdef JazelleEvent event
        while True:
            event = JazelleEvent()
            if not self.cpp_obj.get().nextRecord(event.cpp_event):
                break
            yield event

    def __getitem__(self, int index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Event index out of range")
        
        event = JazelleEvent()
        if self.readEvent(index, event):
            return event
        else:
            raise IndexError(f"Failed to read event at index {index}")


# ==============================================================================
# Initialization
# ==============================================================================

def _register_wrappers():
    """
    Introspects the module to find all Bank subclasses 
    and registers them in _WRAPPER_MAP.
    """
    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and issubclass(obj, Bank) and obj is not Bank:
            _WRAPPER_MAP[name] = obj

_register_wrappers()