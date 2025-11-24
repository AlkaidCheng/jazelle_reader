"""
Cython bindings for Jazelle Reader.

This module provides Python bindings for the C++ Jazelle Reader library,
with support for efficient data extraction in multiple formats.
"""

from typing import Dict, List, Optional, Union, Literal
from collections import defaultdict
import sys
import inspect
import cython
import datetime
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int16_t, int32_t
from libc.string cimport memcpy
from libc.stdint cimport uintptr_t
from libcpp.string cimport string
from libcpp.string_view cimport string_view
from libcpp.memory cimport unique_ptr
from libcpp.map cimport map as std_map
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.chrono cimport system_clock, to_time_t
from libcpp.utility cimport move

# --- NumPy Setup ---
import numpy as np
cimport numpy as cnp

# Initialize NumPy API (Essential for cimport to work)
cnp.import_array()

# Import the .pxd definitions
cimport jazelle_cython as pxd

# ============================================================================
# MODULE-LEVEL CONFIGURATION
# ============================================================================

# Global default for number of threads (None means auto-detect)
_default_num_threads = None

def set_default_num_threads(num_threads: Optional[int]):
    """
    Set the global default number of threads for all JazelleFile operations.
    
    This setting affects all new JazelleFile instances unless overridden
    in the constructor or method call.
    
    Parameters
    ----------
    num_threads : int or None
        Default number of threads to use for parallel operations.
        - If None or 0: Auto-detect based on available cores
        - If > 0: Use specified number of threads
        - If < 0: Disable parallel processing (single-threaded)
    
    Examples
    --------
    >>> import jazelle
    >>> jazelle.set_default_num_threads(8)  # Use 8 threads globally
    >>> 
    >>> with jazelle.JazelleFile('data.jazelle') as f:
    ...     # Will use 8 threads unless overridden
    ...     data = f.to_dict()
    """
    global _default_num_threads
    _default_num_threads = num_threads

def get_default_num_threads() -> Optional[int]:
    """
    Get the current global default number of threads.
    
    Returns
    -------
    int or None
        Current default number of threads, or None if auto-detect.
    """
    return _default_num_threads

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

cdef class EventBatchWrapper:
    """Wraps a pointer to a C++ vector of events for dynamic passing."""
    cdef vector[pxd.CppJazelleEvent]* ptr
    
    def __cinit__(self):
        self.ptr = NULL
        
# ==============================================================================
# REGISTRY INFRASTRUCTURE
# ==============================================================================

# Define a C-function pointer type for our extractors
ctypedef dict (*BatchExtractor)(vector[pxd.CppJazelleEvent]*)

# A global C++ map to hold the registry
cdef std_map[string_view, BatchExtractor] _batch_extractors

# Helper to register functions (clean syntax)
cdef void register_extractor(string_view name, BatchExtractor func):
    _batch_extractors[name] = func
        
# ==============================================================================
# Internal Wrapper Utilities
# ==============================================================================

cdef class Family:
    """
    Wrapper for a family of banks (variable-length collection).
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
        Convert family data to dictionary.
        
        Parameters
        ----------
        orient : {'list', 'records'}, default 'list'
            Data orientation:
            
            - 'list' : columnar {attr: np.array([...])}
            - 'records' : row-based [{attr: val, ...}, ...]
            
        Returns
        -------
        dict or list
            Family data in requested format.
            
        """
        if orient == 'records':
            return self._to_dict_records()
        elif orient == 'list':
            return self._to_dict_list()
        else:
            raise ValueError(
                f"Invalid orient '{orient}'. Must be 'records' or 'list'."
            )

    cdef list _to_dict_records(self):
        """
        Row-based conversion using Flyweight optimization.
        
        Returns
        -------
        list of dict
            Each element is a dictionary representing one bank.
        """
        cdef list result = []
        cdef size_t size = self._ptr.size()
        
        if size == 0:
            return result
            
        cdef size_t i
        cdef pxd.CppBank* raw_ptr
        
        # Allocate one Python wrapper to reuse (flyweight pattern)
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
        
        Delegates to the specific Bank's static NumPy extractor
        for optimal performance.
        
        Returns
        -------
        dict
            Dictionary mapping attribute names to numpy arrays.
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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        """
        Extract data directly from C++ vector into NumPy arrays.
        To be implemented by concrete subclasses.
        """
        raise NotImplementedError()        


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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        # Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_header = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_run = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_event = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int64_t, ndim=1] r_time = np.empty(count, dtype=np.int64)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_weight = np.empty(count, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_evttype = np.empty(count, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_trigger = np.empty(count, dtype=np.int32)

        # Pointers
        cdef int32_t* p_id = <int32_t*>r_id.data
        cdef int32_t* p_header = <int32_t*>r_header.data
        cdef int32_t* p_run = <int32_t*>r_run.data
        cdef int32_t* p_event = <int32_t*>r_event.data
        cdef int64_t* p_time = <int64_t*>r_time.data
        cdef float* p_weight = <float*>r_weight.data
        cdef int32_t* p_evttype = <int32_t*>r_evttype.data
        cdef int32_t* p_trigger = <int32_t*>r_trigger.data

        cdef size_t i
        cdef pxd.CppIEVENTH* ptr
        
        for i in range(count):
            ptr = &batch.at(i).ieventh
            p_id[i] = ptr.getId()
            p_header[i] = ptr.header
            p_run[i] = ptr.run
            p_event[i] = ptr.event
            p_time[i] = to_time_t(ptr.evttime)
            p_weight[i] = ptr.weight
            p_evttype[i] = ptr.evttype
            p_trigger[i] = ptr.trigger

        # Return in C++ read order: header, run, event, time, weight, evttype, trigger
        return {
            'id': r_id, 
            'header': r_header, 
            'run': r_run, 
            'event': r_event,
            'evttime': r_time, 
            'weight': r_weight, 
            'evttype': r_evttype,
            'trigger': r_trigger
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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        cdef cnp.ndarray[cnp.int64_t, ndim=1] arr_offsets = np.empty(count + 1, dtype=np.int64)
        cdef int64_t* r_offsets = <int64_t*>arr_offsets.data
        r_offsets[0] = 0
        cdef size_t total = 0, i
        for i in range(count):
            total += batch.at(i).get[pxd.CppMCHEAD]().size()
            r_offsets[i+1] = total
        if total == 0: return {'_offsets': arr_offsets}

        # Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_ntot = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_origin = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_ipx = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_ipy = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_ipz = np.empty(total, dtype=np.float32)

        cdef int32_t* p_id = <int32_t*>r_id.data
        cdef int32_t* p_ntot = <int32_t*>r_ntot.data
        cdef int32_t* p_origin = <int32_t*>r_origin.data
        cdef float* p_ipx = <float*>r_ipx.data
        cdef float* p_ipy = <float*>r_ipy.data
        cdef float* p_ipz = <float*>r_ipz.data

        cdef size_t g_idx = 0, j
        cdef pxd.CppFamily[pxd.CppMCHEAD]* fam
        cdef pxd.CppMCHEAD* b

        for i in range(count):
            fam = &batch.at(i).get[pxd.CppMCHEAD]()
            for j in range(fam.size()):
                b = <pxd.CppMCHEAD*>fam.at(j)
                p_id[g_idx] = b.getId()
                p_ntot[g_idx] = b.ntot
                p_origin[g_idx] = b.origin
                p_ipx[g_idx] = b.ipx
                p_ipy[g_idx] = b.ipy
                p_ipz[g_idx] = b.ipz
                g_idx += 1

        return {
            '_offsets': arr_offsets, 
            'id': r_id, 
            'ntot': r_ntot, 
            'origin': r_origin, 
            'ipx': r_ipx, 
            'ipy': r_ipy, 
            'ipz': r_ipz
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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        cdef cnp.ndarray[cnp.int64_t, ndim=1] arr_offsets = np.empty(count + 1, dtype=np.int64)
        cdef int64_t* r_offsets = <int64_t*>arr_offsets.data
        r_offsets[0] = 0
        cdef size_t total = 0, i
        for i in range(count):
            total += batch.at(i).get[pxd.CppMCPART]().size()
            r_offsets[i+1] = total
        if total == 0: return {'_offsets': arr_offsets}

        # Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_p = np.empty((total, 3), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_e = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_ptot = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_ptype = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_charge = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_origin = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_xt = np.empty((total, 3), dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_parent_id = np.empty(total, dtype=np.int32)

        cdef int32_t* p_id = <int32_t*>r_id.data
        cdef float* p_p = <float*>r_p.data
        cdef float* p_e = <float*>r_e.data
        cdef float* p_ptot = <float*>r_ptot.data
        cdef int32_t* p_ptype = <int32_t*>r_ptype.data
        cdef float* p_charge = <float*>r_charge.data
        cdef int32_t* p_origin = <int32_t*>r_origin.data
        cdef float* p_xt = <float*>r_xt.data
        cdef int32_t* p_parent_id = <int32_t*>r_parent_id.data

        cdef size_t g_idx = 0, j
        cdef pxd.CppFamily[pxd.CppMCPART]* fam
        cdef pxd.CppMCPART* b

        for i in range(count):
            fam = &batch.at(i).get[pxd.CppMCPART]()
            for j in range(fam.size()):
                b = <pxd.CppMCPART*>fam.at(j)
                p_id[g_idx] = b.getId()
                memcpy(p_p + (g_idx*3), &b.p[0], 12)
                p_e[g_idx] = b.e
                p_ptot[g_idx] = b.ptot
                p_ptype[g_idx] = b.ptype
                p_charge[g_idx] = b.charge
                p_origin[g_idx] = b.origin
                memcpy(p_xt + (g_idx*3), &b.xt[0], 12)
                p_parent_id[g_idx] = b.parent_id
                g_idx += 1

        return {
            '_offsets': arr_offsets, 
            'id': r_id, 
            'p': r_p, 
            'e': r_e, 
            'ptot': r_ptot, 
            'ptype': r_ptype, 
            'charge': r_charge, 
            'origin': r_origin, 
            'xt': r_xt, 
            'parent_id': r_parent_id
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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        cdef cnp.ndarray[cnp.int64_t, ndim=1] arr_offsets = np.empty(count + 1, dtype=np.int64)
        cdef int64_t* r_offsets = <int64_t*>arr_offsets.data
        r_offsets[0] = 0
        cdef size_t total = 0, i
        for i in range(count):
            total += batch.at(i).get[pxd.CppPHPSUM]().size()
            r_offsets[i+1] = total
        if total == 0: return {'_offsets': arr_offsets}

        # Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_px = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_py = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_pz = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_x = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_y = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_z = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_charge = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_status = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] r_ptot = np.empty(total, dtype=np.float64)

        cdef int32_t* p_id = <int32_t*>r_id.data
        cdef float* p_px = <float*>r_px.data
        cdef float* p_py = <float*>r_py.data
        cdef float* p_pz = <float*>r_pz.data
        cdef float* p_x = <float*>r_x.data
        cdef float* p_y = <float*>r_y.data
        cdef float* p_z = <float*>r_z.data
        cdef float* p_charge = <float*>r_charge.data
        cdef int32_t* p_status = <int32_t*>r_status.data
        cdef double* p_ptot = <double*>r_ptot.data

        cdef size_t g_idx = 0, j
        cdef pxd.CppFamily[pxd.CppPHPSUM]* fam
        cdef pxd.CppPHPSUM* b

        for i in range(count):
            fam = &batch.at(i).get[pxd.CppPHPSUM]()
            for j in range(fam.size()):
                b = <pxd.CppPHPSUM*>fam.at(j)
                p_id[g_idx] = b.getId()
                p_px[g_idx] = b.px
                p_py[g_idx] = b.py
                p_pz[g_idx] = b.pz
                p_x[g_idx] = b.x
                p_y[g_idx] = b.y
                p_z[g_idx] = b.z
                p_charge[g_idx] = b.charge
                p_status[g_idx] = b.status
                p_ptot[g_idx] = b.getPTot()
                g_idx += 1

        return {
            '_offsets': arr_offsets, 'id': r_id, 
            'px': r_px, 'py': r_py, 'pz': r_pz, 
            'x': r_x, 'y': r_y, 'z': r_z,
            'charge': r_charge, 'status': r_status, 
            'ptot': r_ptot
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

    cpdef dict to_dict(self):
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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        cdef cnp.ndarray[cnp.int64_t, ndim=1] arr_offsets = np.empty(count + 1, dtype=np.int64)
        cdef int64_t* r_offsets = <int64_t*>arr_offsets.data
        r_offsets[0] = 0
        cdef size_t total = 0, i
        for i in range(count):
            total += batch.at(i).get[pxd.CppPHCHRG]().size()
            r_offsets[i+1] = total
        if total == 0: return {'_offsets': arr_offsets}

        # Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_hlxpar = np.empty((total, 6), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_dhlxpar = np.empty((total, 15), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_bnorm = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_impact = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_b3norm = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_impact3 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_charge = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_smwstat = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_status = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_tkpar0 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_tkpar = np.empty((total, 5), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_dtkpar = np.empty((total, 15), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_length = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_chi2dt = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_imc = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_ndfdt = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nhit = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nhite = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nhitp = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nmisht = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nwrght = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nhitv = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_chi2 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_chi2v = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_vxdhit = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_mustat = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_estat = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_dedx = np.empty(total, dtype=np.int32)

        cdef int32_t* p_id = <int32_t*>r_id.data
        cdef float* p_hlxpar = <float*>r_hlxpar.data
        cdef float* p_dhlxpar = <float*>r_dhlxpar.data
        cdef float* p_bnorm = <float*>r_bnorm.data
        cdef float* p_impact = <float*>r_impact.data
        cdef float* p_b3norm = <float*>r_b3norm.data
        cdef float* p_impact3 = <float*>r_impact3.data
        cdef int16_t* p_charge = <int16_t*>r_charge.data
        cdef int16_t* p_smwstat = <int16_t*>r_smwstat.data
        cdef int32_t* p_status = <int32_t*>r_status.data
        cdef float* p_tkpar0 = <float*>r_tkpar0.data
        cdef float* p_tkpar = <float*>r_tkpar.data
        cdef float* p_dtkpar = <float*>r_dtkpar.data
        cdef float* p_length = <float*>r_length.data
        cdef float* p_chi2dt = <float*>r_chi2dt.data
        cdef int16_t* p_imc = <int16_t*>r_imc.data
        cdef int16_t* p_ndfdt = <int16_t*>r_ndfdt.data
        cdef int16_t* p_nhit = <int16_t*>r_nhit.data
        cdef int16_t* p_nhite = <int16_t*>r_nhite.data
        cdef int16_t* p_nhitp = <int16_t*>r_nhitp.data
        cdef int16_t* p_nmisht = <int16_t*>r_nmisht.data
        cdef int16_t* p_nwrght = <int16_t*>r_nwrght.data
        cdef int16_t* p_nhitv = <int16_t*>r_nhitv.data
        cdef float* p_chi2 = <float*>r_chi2.data
        cdef float* p_chi2v = <float*>r_chi2v.data
        cdef int32_t* p_vxdhit = <int32_t*>r_vxdhit.data
        cdef int16_t* p_mustat = <int16_t*>r_mustat.data
        cdef int16_t* p_estat = <int16_t*>r_estat.data
        cdef int32_t* p_dedx = <int32_t*>r_dedx.data

        cdef size_t g_idx = 0, j
        cdef pxd.CppFamily[pxd.CppPHCHRG]* fam
        cdef pxd.CppPHCHRG* b

        for i in range(count):
            fam = &batch.at(i).get[pxd.CppPHCHRG]()
            for j in range(fam.size()):
                b = <pxd.CppPHCHRG*>fam.at(j)
                p_id[g_idx] = b.getId()
                memcpy(p_hlxpar + (g_idx*6), &b.hlxpar[0], 24)
                memcpy(p_dhlxpar + (g_idx*15), &b.dhlxpar[0], 60)
                p_bnorm[g_idx] = b.bnorm
                p_impact[g_idx] = b.impact
                p_b3norm[g_idx] = b.b3norm
                p_impact3[g_idx] = b.impact3
                p_charge[g_idx] = b.charge
                p_smwstat[g_idx] = b.smwstat
                p_status[g_idx] = b.status
                p_tkpar0[g_idx] = b.tkpar0
                memcpy(p_tkpar + (g_idx*5), &b.tkpar[0], 20)
                memcpy(p_dtkpar + (g_idx*15), &b.dtkpar[0], 60)
                p_length[g_idx] = b.length
                p_chi2dt[g_idx] = b.chi2dt
                p_imc[g_idx] = b.imc
                p_ndfdt[g_idx] = b.ndfdt
                p_nhit[g_idx] = b.nhit
                p_nhite[g_idx] = b.nhite
                p_nhitp[g_idx] = b.nhitp
                p_nmisht[g_idx] = b.nmisht
                p_nwrght[g_idx] = b.nwrght
                p_nhitv[g_idx] = b.nhitv
                p_chi2[g_idx] = b.chi2
                p_chi2v[g_idx] = b.chi2v
                p_vxdhit[g_idx] = b.vxdhit
                p_mustat[g_idx] = b.mustat
                p_estat[g_idx] = b.estat
                p_dedx[g_idx] = b.dedx
                g_idx += 1

        return {
            '_offsets': arr_offsets, 'id': r_id, 
            'hlxpar': r_hlxpar, 'dhlxpar': r_dhlxpar,
            'bnorm': r_bnorm, 'impact': r_impact, 'b3norm': r_b3norm, 'impact3': r_impact3,
            'charge': r_charge, 'smwstat': r_smwstat, 'status': r_status, 
            'tkpar0': r_tkpar0, 'tkpar': r_tkpar, 'dtkpar': r_dtkpar, 
            'length': r_length, 'chi2dt': r_chi2dt,
            'imc': r_imc, 'ndfdt': r_ndfdt, 'nhit': r_nhit, 'nhite': r_nhite, 
            'nhitp': r_nhitp, 'nmisht': r_nmisht, 'nwrght': r_nwrght, 'nhitv': r_nhitv, 
            'chi2': r_chi2, 'chi2v': r_chi2v, 'vxdhit': r_vxdhit, 
            'mustat': r_mustat, 'estat': r_estat, 'dedx': r_dedx
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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        cdef cnp.ndarray[cnp.int64_t, ndim=1] arr_offsets = np.empty(count + 1, dtype=np.int64)
        cdef int64_t* r_offsets = <int64_t*>arr_offsets.data
        r_offsets[0] = 0
        cdef size_t total = 0, i
        for i in range(count):
            total += batch.at(i).get[pxd.CppPHKLUS]().size()
            r_offsets[i+1] = total
        if total == 0: return {'_offsets': arr_offsets}

        # Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_status = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_eraw = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_cth = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_wcth = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_phi = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_wphi = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_elayer = np.empty((total, 8), dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_nhit2 = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_cth2 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_wcth2 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_phi2 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_whphi2 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_nhit3 = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_cth3 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_wcth3 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_phi3 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_wphi3 = np.empty(total, dtype=np.float32)

        cdef int32_t* p_id = <int32_t*>r_id.data
        cdef int32_t* p_status = <int32_t*>r_status.data
        cdef float* p_eraw = <float*>r_eraw.data
        cdef float* p_cth = <float*>r_cth.data
        cdef float* p_wcth = <float*>r_wcth.data
        cdef float* p_phi = <float*>r_phi.data
        cdef float* p_wphi = <float*>r_wphi.data
        cdef float* p_elayer = <float*>r_elayer.data
        cdef int32_t* p_nhit2 = <int32_t*>r_nhit2.data
        cdef float* p_cth2 = <float*>r_cth2.data
        cdef float* p_wcth2 = <float*>r_wcth2.data
        cdef float* p_phi2 = <float*>r_phi2.data
        cdef float* p_whphi2 = <float*>r_whphi2.data
        cdef int32_t* p_nhit3 = <int32_t*>r_nhit3.data
        cdef float* p_cth3 = <float*>r_cth3.data
        cdef float* p_wcth3 = <float*>r_wcth3.data
        cdef float* p_phi3 = <float*>r_phi3.data
        cdef float* p_wphi3 = <float*>r_wphi3.data

        cdef size_t g_idx = 0, j
        cdef pxd.CppFamily[pxd.CppPHKLUS]* fam
        cdef pxd.CppPHKLUS* b

        for i in range(count):
            fam = &batch.at(i).get[pxd.CppPHKLUS]()
            for j in range(fam.size()):
                b = <pxd.CppPHKLUS*>fam.at(j)
                p_id[g_idx] = b.getId()
                p_status[g_idx] = b.status
                p_eraw[g_idx] = b.eraw
                p_cth[g_idx] = b.cth
                p_wcth[g_idx] = b.wcth
                p_phi[g_idx] = b.phi
                p_wphi[g_idx] = b.wphi
                memcpy(p_elayer + (g_idx*8), &b.elayer[0], 32)
                p_nhit2[g_idx] = b.nhit2
                p_cth2[g_idx] = b.cth2
                p_wcth2[g_idx] = b.wcth2
                p_phi2[g_idx] = b.phi2
                p_whphi2[g_idx] = b.whphi2
                p_nhit3[g_idx] = b.nhit3
                p_cth3[g_idx] = b.cth3
                p_wcth3[g_idx] = b.wcth3
                p_phi3[g_idx] = b.phi3
                p_wphi3[g_idx] = b.wphi3
                g_idx += 1

        # Order matches src/banks/PHKLUS.cpp read()
        return {
            '_offsets': arr_offsets, 'id': r_id, 'status': r_status, 
            'eraw': r_eraw, 'cth': r_cth, 'wcth': r_wcth, 'phi': r_phi, 'wphi': r_wphi,
            'elayer': r_elayer, 
            'nhit2': r_nhit2, 'cth2': r_cth2, 'wcth2': r_wcth2, 'phi2': r_phi2, 'whphi2': r_whphi2, 
            'nhit3': r_nhit3, 'cth3': r_cth3, 'wcth3': r_wcth3, 'phi3': r_phi3, 'wphi3': r_wphi3
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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        cdef cnp.ndarray[cnp.int64_t, ndim=1] arr_offsets = np.empty(count + 1, dtype=np.int64)
        cdef int64_t* r_offsets = <int64_t*>arr_offsets.data
        r_offsets[0] = 0
        cdef size_t total = 0, i
        for i in range(count):
            total += batch.at(i).get[pxd.CppPHWIC]().size()
            r_offsets[i+1] = total
        if total == 0: return {'_offsets': arr_offsets}

        # Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_idstat = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nhit = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nhit45 = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_npat = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nhitpat = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_syshit = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_qpinit = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_t1 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_t2 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_t3 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_hitmiss = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_itrlen = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nlayexp = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nlaybey = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_missprob = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_phwicid = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nhitshar = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nother = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_hitsused = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_pref1 = np.empty((total, 3), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_pfit = np.empty((total, 4), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=2] r_dpfit = np.empty((total, 10), dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_chi2 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_ndf = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_punfit = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_matchChi2 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_matchNdf = np.empty(total, dtype=np.int16)

        cdef int32_t* p_id = <int32_t*>r_id.data
        cdef int16_t* p_idstat = <int16_t*>r_idstat.data
        cdef int16_t* p_nhit = <int16_t*>r_nhit.data
        cdef int16_t* p_nhit45 = <int16_t*>r_nhit45.data
        cdef int16_t* p_npat = <int16_t*>r_npat.data
        cdef int16_t* p_nhitpat = <int16_t*>r_nhitpat.data
        cdef int16_t* p_syshit = <int16_t*>r_syshit.data
        cdef float* p_qpinit = <float*>r_qpinit.data
        cdef float* p_t1 = <float*>r_t1.data
        cdef float* p_t2 = <float*>r_t2.data
        cdef float* p_t3 = <float*>r_t3.data
        cdef int32_t* p_hitmiss = <int32_t*>r_hitmiss.data
        cdef float* p_itrlen = <float*>r_itrlen.data
        cdef int16_t* p_nlayexp = <int16_t*>r_nlayexp.data
        cdef int16_t* p_nlaybey = <int16_t*>r_nlaybey.data
        cdef float* p_missprob = <float*>r_missprob.data
        cdef int32_t* p_phwicid = <int32_t*>r_phwicid.data
        cdef int16_t* p_nhitshar = <int16_t*>r_nhitshar.data
        cdef int16_t* p_nother = <int16_t*>r_nother.data
        cdef int32_t* p_hitsused = <int32_t*>r_hitsused.data
        cdef float* p_pref1 = <float*>r_pref1.data
        cdef float* p_pfit = <float*>r_pfit.data
        cdef float* p_dpfit = <float*>r_dpfit.data
        cdef float* p_chi2 = <float*>r_chi2.data
        cdef int16_t* p_ndf = <int16_t*>r_ndf.data
        cdef int16_t* p_punfit = <int16_t*>r_punfit.data
        cdef float* p_matchChi2 = <float*>r_matchChi2.data
        cdef int16_t* p_matchNdf = <int16_t*>r_matchNdf.data

        cdef size_t g_idx = 0, j
        cdef pxd.CppFamily[pxd.CppPHWIC]* fam
        cdef pxd.CppPHWIC* b

        for i in range(count):
            fam = &batch.at(i).get[pxd.CppPHWIC]()
            for j in range(fam.size()):
                b = <pxd.CppPHWIC*>fam.at(j)
                p_id[g_idx] = b.getId()
                p_idstat[g_idx] = b.idstat
                p_nhit[g_idx] = b.nhit
                p_nhit45[g_idx] = b.nhit45
                p_npat[g_idx] = b.npat
                p_nhitpat[g_idx] = b.nhitpat
                p_syshit[g_idx] = b.syshit
                p_qpinit[g_idx] = b.qpinit
                p_t1[g_idx] = b.t1
                p_t2[g_idx] = b.t2
                p_t3[g_idx] = b.t3
                p_hitmiss[g_idx] = b.hitmiss
                p_itrlen[g_idx] = b.itrlen
                p_nlayexp[g_idx] = b.nlayexp
                p_nlaybey[g_idx] = b.nlaybey
                p_missprob[g_idx] = b.missprob
                p_phwicid[g_idx] = b.phwicid
                p_nhitshar[g_idx] = b.nhitshar
                p_nother[g_idx] = b.nother
                p_hitsused[g_idx] = b.hitsused
                memcpy(p_pref1 + (g_idx*3), &b.pref1[0], 12)
                memcpy(p_pfit + (g_idx*4), &b.pfit[0], 16)
                memcpy(p_dpfit + (g_idx*10), &b.dpfit[0], 40)
                p_chi2[g_idx] = b.chi2
                p_ndf[g_idx] = b.ndf
                p_punfit[g_idx] = b.punfit
                p_matchChi2[g_idx] = b.matchChi2
                p_matchNdf[g_idx] = b.matchNdf
                g_idx += 1
                
        return {
            '_offsets': arr_offsets, 'id': r_id, 
            'idstat': r_idstat, 'nhit': r_nhit, 'nhit45': r_nhit45, 'npat': r_npat, 
            'nhitpat': r_nhitpat, 'syshit': r_syshit, 
            'qpinit': r_qpinit, 't1': r_t1, 't2': r_t2, 't3': r_t3, 'hitmiss': r_hitmiss,
            'itrlen': r_itrlen, 'nlayexp': r_nlayexp, 'nlaybey': r_nlaybey, 'missprob': r_missprob,
            'phwicid': r_phwicid, 'nhitshar': r_nhitshar, 'nother': r_nother, 'hitsused': r_hitsused,
            'pref1': r_pref1, 'pfit': r_pfit, 'dpfit': r_dpfit, 
            'chi2': r_chi2, 'ndf': r_ndf, 'punfit': r_punfit, 
            'matchChi2': r_matchChi2, 'matchNdf': r_matchNdf
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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        cdef cnp.ndarray[cnp.int64_t, ndim=1] arr_offsets = np.empty(count + 1, dtype=np.int64)
        cdef int64_t* r_offsets = <int64_t*>arr_offsets.data
        r_offsets[0] = 0
        cdef size_t total = 0, i
        for i in range(count):
            total += batch.at(i).get[pxd.CppPHCRID]().size()
            r_offsets[i+1] = total
        if total == 0: return {'_offsets': arr_offsets}

        # Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_ctlword = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_norm = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_rc = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_geom = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_trkp = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_nhits = np.empty(total, dtype=np.int16)
        
        # Flattened LIQ
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_liq_rc = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_liq_nhits = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_liq_besthyp = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_liq_nhexp = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_liq_nhfnd = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_liq_nhbkg = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_liq_mskphot = np.empty(total, dtype=np.int16)
        
        # Flattened GAS
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_gas_rc = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_gas_nhits = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_gas_besthyp = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_gas_nhexp = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_gas_nhfnd = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_gas_nhbkg = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_gas_mskphot = np.empty(total, dtype=np.int16)

        # Flattened LLIK
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_llik_e = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_llik_mu = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_llik_pi = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_llik_k = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_llik_p = np.empty(total, dtype=np.float32)

        # Pointers
        cdef int32_t* p_id = <int32_t*>r_id.data
        cdef int32_t* p_ctlword = <int32_t*>r_ctlword.data
        cdef float* p_norm = <float*>r_norm.data
        cdef int16_t* p_rc = <int16_t*>r_rc.data
        cdef int16_t* p_geom = <int16_t*>r_geom.data
        cdef int16_t* p_trkp = <int16_t*>r_trkp.data
        cdef int16_t* p_nhits = <int16_t*>r_nhits.data
        
        cdef int16_t* p_liq_rc = <int16_t*>r_liq_rc.data
        cdef int16_t* p_liq_nhits = <int16_t*>r_liq_nhits.data
        cdef int32_t* p_liq_besthyp = <int32_t*>r_liq_besthyp.data
        cdef int16_t* p_liq_nhexp = <int16_t*>r_liq_nhexp.data
        cdef int16_t* p_liq_nhfnd = <int16_t*>r_liq_nhfnd.data
        cdef int16_t* p_liq_nhbkg = <int16_t*>r_liq_nhbkg.data
        cdef int16_t* p_liq_mskphot = <int16_t*>r_liq_mskphot.data

        cdef int16_t* p_gas_rc = <int16_t*>r_gas_rc.data
        cdef int16_t* p_gas_nhits = <int16_t*>r_gas_nhits.data
        cdef int32_t* p_gas_besthyp = <int32_t*>r_gas_besthyp.data
        cdef int16_t* p_gas_nhexp = <int16_t*>r_gas_nhexp.data
        cdef int16_t* p_gas_nhfnd = <int16_t*>r_gas_nhfnd.data
        cdef int16_t* p_gas_nhbkg = <int16_t*>r_gas_nhbkg.data
        cdef int16_t* p_gas_mskphot = <int16_t*>r_gas_mskphot.data

        cdef float* p_llik_e = <float*>r_llik_e.data
        cdef float* p_llik_mu = <float*>r_llik_mu.data
        cdef float* p_llik_pi = <float*>r_llik_pi.data
        cdef float* p_llik_k = <float*>r_llik_k.data
        cdef float* p_llik_p = <float*>r_llik_p.data

        cdef size_t g_idx = 0, j
        cdef pxd.CppFamily[pxd.CppPHCRID]* fam
        cdef pxd.CppPHCRID* b

        for i in range(count):
            fam = &batch.at(i).get[pxd.CppPHCRID]()
            for j in range(fam.size()):
                b = <pxd.CppPHCRID*>fam.at(j)
                p_id[g_idx] = b.getId()
                p_ctlword[g_idx] = b.ctlword
                p_norm[g_idx] = b.norm
                p_rc[g_idx] = b.rc
                p_geom[g_idx] = b.geom
                p_trkp[g_idx] = b.trkp
                p_nhits[g_idx] = b.nhits
                
                # LIQ
                p_liq_rc[g_idx] = b.liq.rc
                p_liq_nhits[g_idx] = b.liq.nhits
                p_liq_besthyp[g_idx] = b.liq.besthyp
                p_liq_nhexp[g_idx] = b.liq.nhexp
                p_liq_nhfnd[g_idx] = b.liq.nhfnd
                p_liq_nhbkg[g_idx] = b.liq.nhbkg
                p_liq_mskphot[g_idx] = b.liq.mskphot
                
                # GAS
                p_gas_rc[g_idx] = b.gas.rc
                p_gas_nhits[g_idx] = b.gas.nhits
                p_gas_besthyp[g_idx] = b.gas.besthyp
                p_gas_nhexp[g_idx] = b.gas.nhexp
                p_gas_nhfnd[g_idx] = b.gas.nhfnd
                p_gas_nhbkg[g_idx] = b.gas.nhbkg
                p_gas_mskphot[g_idx] = b.gas.mskphot
                
                # LLIK
                p_llik_e[g_idx] = b.llik.e
                p_llik_mu[g_idx] = b.llik.mu
                p_llik_pi[g_idx] = b.llik.pi
                p_llik_k[g_idx] = b.llik.k
                p_llik_p[g_idx] = b.llik.p
                
                g_idx += 1

        return {
            '_offsets': arr_offsets, 'id': r_id, 'ctlword': r_ctlword, 'norm': r_norm,
            'rc': r_rc, 'geom': r_geom, 'trkp': r_trkp, 'nhits': r_nhits,
            'liq_rc': r_liq_rc, 'liq_nhits': r_liq_nhits, 'liq_besthyp': r_liq_besthyp,
            'liq_nhexp': r_liq_nhexp, 'liq_nhfnd': r_liq_nhfnd, 'liq_nhbkg': r_liq_nhbkg,
            'liq_mskphot': r_liq_mskphot,
            'gas_rc': r_gas_rc, 'gas_nhits': r_gas_nhits, 'gas_besthyp': r_gas_besthyp,
            'gas_nhexp': r_gas_nhexp, 'gas_nhfnd': r_gas_nhfnd, 'gas_nhbkg': r_gas_nhbkg,
            'gas_mskphot': r_gas_mskphot,
            'llik_e': r_llik_e, 'llik_mu': r_llik_mu, 'llik_pi': r_llik_pi,
            'llik_k': r_llik_k, 'llik_p': r_llik_p
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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        cdef cnp.ndarray[cnp.int64_t, ndim=1] arr_offsets = np.empty(count + 1, dtype=np.int64)
        cdef int64_t* r_offsets = <int64_t*>arr_offsets.data
        r_offsets[0] = 0
        cdef size_t total = 0, i
        for i in range(count):
            total += batch.at(i).get[pxd.CppPHKTRK]().size()
            r_offsets[i+1] = total
        if total == 0: return {'_offsets': arr_offsets}

        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(total, dtype=np.int32)
        cdef int32_t* p_id = <int32_t*>r_id.data

        cdef size_t g_idx = 0, j
        cdef pxd.CppFamily[pxd.CppPHKTRK]* fam
        cdef pxd.CppPHKTRK* b

        for i in range(count):
            fam = &batch.at(i).get[pxd.CppPHKTRK]()
            for j in range(fam.size()):
                b = <pxd.CppPHKTRK*>fam.at(j)
                p_id[g_idx] = b.getId()
                g_idx += 1

        return {'_offsets': arr_offsets, 'id': r_id}


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

    @staticmethod
    cdef dict extract_from_vector(vector[pxd.CppJazelleEvent]* batch):
        cdef size_t count = batch.size()
        if count == 0: return {}

        cdef cnp.ndarray[cnp.int64_t, ndim=1] arr_offsets = np.empty(count + 1, dtype=np.int64)
        cdef int64_t* r_offsets = <int64_t*>arr_offsets.data
        r_offsets[0] = 0
        cdef size_t total = 0, i
        for i in range(count):
            total += batch.at(i).get[pxd.CppPHKELID]().size()
            r_offsets[i+1] = total
        if total == 0: return {'_offsets': arr_offsets}

        # Allocations
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_id = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int32_t, ndim=1] r_phchrg_id = np.empty(total, dtype=np.int32)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_idstat = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.int16_t, ndim=1] r_prob = np.empty(total, dtype=np.int16)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_phi = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_theta = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_qp = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_dphi = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_dtheta = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_dqp = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_tphi = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_ttheta = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_isolat = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_em1 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_em12 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_dem12 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_had1 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_emphi = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_emtheta = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_phiwid = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_thewid = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_em1x1 = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_em2x2a = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_em2x2b = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_em3x3a = np.empty(total, dtype=np.float32)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] r_em3x3b = np.empty(total, dtype=np.float32)

        cdef int32_t* p_id = <int32_t*>r_id.data
        cdef int32_t* p_phchrg_id = <int32_t*>r_phchrg_id.data
        cdef int16_t* p_idstat = <int16_t*>r_idstat.data
        cdef int16_t* p_prob = <int16_t*>r_prob.data
        cdef float* p_phi = <float*>r_phi.data
        cdef float* p_theta = <float*>r_theta.data
        cdef float* p_qp = <float*>r_qp.data
        cdef float* p_dphi = <float*>r_dphi.data
        cdef float* p_dtheta = <float*>r_dtheta.data
        cdef float* p_dqp = <float*>r_dqp.data
        cdef float* p_tphi = <float*>r_tphi.data
        cdef float* p_ttheta = <float*>r_ttheta.data
        cdef float* p_isolat = <float*>r_isolat.data
        cdef float* p_em1 = <float*>r_em1.data
        cdef float* p_em12 = <float*>r_em12.data
        cdef float* p_dem12 = <float*>r_dem12.data
        cdef float* p_had1 = <float*>r_had1.data
        cdef float* p_emphi = <float*>r_emphi.data
        cdef float* p_emtheta = <float*>r_emtheta.data
        cdef float* p_phiwid = <float*>r_phiwid.data
        cdef float* p_thewid = <float*>r_thewid.data
        cdef float* p_em1x1 = <float*>r_em1x1.data
        cdef float* p_em2x2a = <float*>r_em2x2a.data
        cdef float* p_em2x2b = <float*>r_em2x2b.data
        cdef float* p_em3x3a = <float*>r_em3x3a.data
        cdef float* p_em3x3b = <float*>r_em3x3b.data

        cdef size_t g_idx = 0, j
        cdef pxd.CppFamily[pxd.CppPHKELID]* fam
        cdef pxd.CppPHKELID* b

        for i in range(count):
            fam = &batch.at(i).get[pxd.CppPHKELID]()
            for j in range(fam.size()):
                b = <pxd.CppPHKELID*>fam.at(j)
                p_id[g_idx] = b.getId()
                
                if b.phchrg == NULL:
                    p_phchrg_id[g_idx] = -1
                else:
                    p_phchrg_id[g_idx] = b.phchrg.getId()

                p_idstat[g_idx] = b.idstat
                p_prob[g_idx] = b.prob
                p_phi[g_idx] = b.phi
                p_theta[g_idx] = b.theta
                p_qp[g_idx] = b.qp
                p_dphi[g_idx] = b.dphi
                p_dtheta[g_idx] = b.dtheta
                p_dqp[g_idx] = b.dqp
                p_tphi[g_idx] = b.tphi
                p_ttheta[g_idx] = b.ttheta
                p_isolat[g_idx] = b.isolat
                p_em1[g_idx] = b.em1
                p_em12[g_idx] = b.em12
                p_dem12[g_idx] = b.dem12
                p_had1[g_idx] = b.had1
                p_emphi[g_idx] = b.emphi
                p_emtheta[g_idx] = b.emtheta
                p_phiwid[g_idx] = b.phiwid
                p_thewid[g_idx] = b.thewid
                p_em1x1[g_idx] = b.em1x1
                p_em2x2a[g_idx] = b.em2x2a
                p_em2x2b[g_idx] = b.em2x2b
                p_em3x3a[g_idx] = b.em3x3a
                p_em3x3b[g_idx] = b.em3x3b
                g_idx += 1

        return {
            '_offsets': arr_offsets, 'id': r_id, 'phchrg_id': r_phchrg_id, 
            'idstat': r_idstat, 'prob': r_prob, 
            'phi': r_phi, 'theta': r_theta, 'qp': r_qp, 'dphi': r_dphi, 'dtheta': r_dtheta, 'dqp': r_dqp, 
            'tphi': r_tphi, 'ttheta': r_ttheta, 'isolat': r_isolat, 
            'em1': r_em1, 'em12': r_em12, 'dem12': r_dem12, 'had1': r_had1, 
            'emphi': r_emphi, 'emtheta': r_emtheta, 'phiwid': r_phiwid, 'thewid': r_thewid, 
            'em1x1': r_em1x1, 'em2x2a': r_em2x2a, 'em2x2b': r_em2x2b, 'em3x3a': r_em3x3a, 'em3x3b': r_em3x3b
        }


# ==============================================================================
# JazelleEvent Wrapper
# ==============================================================================

cdef class JazelleEvent:
    """
    Wrapper for a single Jazelle event.
    
    This class provides access to event data and supports conversion
    to various dictionary formats for different use cases.
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

    def to_dict(
        self,
        orient: Literal['list', 'records'] = 'list',
        skip_empty: bool = True
    ) -> Dict:
        """
        Convert event to dictionary format.
        
        Parameters
        ----------
        orient : {'list', 'records'}, default 'list'
            Data orientation for multi-bank families:
            
            - 'list' : columnar format {attr: np.array([...])}
            - 'records' : row format [{attr: val, ...}, ...]
            
        skip_empty : bool, default True
            Whether to skip families with no banks.
            
        Returns
        -------
        dict
            Event data in requested format.
            
        Examples
        --------
        >>> event = file[0]
        >>> 
        >>> # Default: columnar per-family
        >>> data = event.to_dict()
        >>> data['PHCHRG']['px']  # numpy array
        >>> 
        >>> # Row-oriented per-family
        >>> data = event.to_dict(orient='records')
        >>> data['PHCHRG'][0]['px']  # first bank's px value
        """
        cdef dict data = {}
        cdef object family
        
        # Add IEVENTH (single bank per event)
        data["IEVENTH"] = self.ieventh.to_dict()
        
        # Add other families
        for family in self.getFamilies():
            if not skip_empty or len(family) > 0:
                data[family.name] = family.to_dict(orient=orient)
        
        return data


# ============================================================================
# JAZELLE FILE CLASS
# ============================================================================

cdef class JazelleFile:
    """
    High-performance Jazelle file reader with parallel processing support.
    
    This class provides efficient access to Jazelle data files with:
    - Sequential and random event access
    - Parallel batch reading
    - Flexible iteration strategies
    
    Parameters
    ----------
    filepath : str
        Path to the Jazelle file
    num_threads : int or None, optional
        Number of threads for parallel operations.
        - If None: Use global default (see set_default_num_threads)
        - If 0: Auto-detect based on available cores
        - If > 0: Use specified number of threads
        - If < 0: Disable parallel processing
    
    Examples
    --------
    >>> # Using context manager (recommended)
    >>> with JazelleFile('data.jazelle', num_threads=8) as f:
    ...     print(f"File contains {len(f)} events")
    ...     data = f.to_dict()
    
    >>> # Custom threading for specific operations
    >>> with JazelleFile('data.jazelle') as f:
    ...     # Override default for this operation
    ...     data = f.to_dict(num_threads=16)
    """
    cdef unique_ptr[pxd.CppJazelleFile] cpp_obj
    cdef object _num_threads

    def __cinit__(self, filepath: str, num_threads: Optional[int] = None):
        """Initialize Jazelle file reader."""
        cdef string s_filepath = filepath.encode('UTF-8')
        try:
            self.cpp_obj.reset(new pxd.CppJazelleFile(s_filepath))
        except Exception as e:
            raise RuntimeError(f"Error opening Jazelle file: {e}")
        
        # Store the user-provided value (may be None)
        self._num_threads = num_threads

    def __len__(self):
        return self.getTotalEvents()

    def __iter__(self):
        """
        Default iterator (sequential, single events).
        
        Equivalent to iterate(batch_size=1).
        
        Examples
        --------
        >>> with JazelleFile('data.jazelle') as f:
        ...     for event in f:
        ...         print(event.ieventh.run)
        """
        return self.iterate(batch_size=1)

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

    @property
    def num_threads(self) -> int:
        """
        Get current default number of threads for this file.
        
        This property automatically syncs with the global default if no
        instance-specific value was set. If the global default changes,
        this property will reflect that change (unless an instance-specific
        value was set via the constructor or set_num_threads()).
        
        Returns
        -------
        int
            Number of threads to use:
            - If instance value set: returns that value
            - If global default set: returns global default
            - Otherwise: returns 0 (auto-detect)
        """
        global _default_num_threads
        
        if self._num_threads is not None:
            return self._num_threads
        elif _default_num_threads is not None:
            return _default_num_threads
        else:
            return 0

    def set_num_threads(self, num_threads: Optional[int] = None):
        """
        Set the default number of threads for this file instance.
        
        This overrides both the global default and any value set during
        construction. Set to None to resume syncing with global default.
        
        Parameters
        ----------
        num_threads : int or None
            Number of threads:
            - If None: Resume syncing with global default
            - If 0: Auto-detect based on available cores
            - If > 0: Use specified number of threads
            - If < 0: Disable parallelization (force sequential)
        
        Examples
        --------
        >>> import jazelle
        >>> 
        >>> jazelle.set_default_num_threads(8)
        >>> f = jazelle.JazelleFile('data.jazelle')
        >>> print(f.num_threads)  # 8 (uses global)
        >>> 
        >>> f.set_num_threads(16)
        >>> print(f.num_threads)  # 16 (overridden)
        >>> 
        >>> jazelle.set_default_num_threads(4)
        >>> print(f.num_threads)  # Still 16 (not synced)
        >>> 
        >>> f.set_num_threads(None)
        >>> print(f.num_threads)  # 4 (now syncs with global again)
        """
        self._num_threads = num_threads

    def _resolve_num_threads(self, num_threads: Optional[int] = None):
        return num_threads if num_threads is not None else self.num_threads

    # ========================================================================
    # Context Manager Protocol
    # ========================================================================
        
    def __enter__(self):
        """Context Manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager exit. Ensures file is closed."""
        self.close()

    def close(self):
        """Explicitly closes the file and frees resources."""
        self.cpp_obj.reset()
        
    # ========================================================================
    # File Metadata Properties
    # ========================================================================
        
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

    def getTotalEvents(self):
        """Returns the total number of events in the file."""
        return self.cpp_obj.get().getTotalEvents()
    
    def rewind(self):
        """Resets the file pointer to the first event."""
        self.cpp_obj.get().rewind()

    # ========================================================================
    # Basic Reading Operations
    # ========================================================================

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

    # ========================================================================
    # Batch Reading
    # ========================================================================

    def read_batch(
        self,
        start: int = 0,
        count: int = -1,
        num_threads: Optional[int] = None
    ) -> List[JazelleEvent]:
        """
        Read a batch of events in parallel.
        
        This method reads multiple events efficiently using parallel threads.
        The entire batch is returned as a list.
        
        Parameters
        ----------
        start : int, default 0
            Starting event index
        count : int, default -1
            Number of events to read (-1 for all remaining)
        num_threads : int or None, optional
            Number of threads (None uses instance default)
        
        Returns
        -------
        list of JazelleEvent
            Batch of events
        
        Examples
        --------
        >>> with JazelleFile('data.jazelle') as f:
        ...     # Read first 1000 events using 8 threads
        ...     events = f.read_batch(0, 1000, num_threads=8)
        ...     print(f"Read {len(events)} events")
        
        >>> with JazelleFile('data.jazelle', num_threads=8) as f:
        ...     # Uses instance default (8 threads)
        ...     events = f.read_batch(1000, 500)
        
        Notes
        -----
        For very large batches, consider using iterate() instead to
        process data in smaller chunks and avoid memory issues.
        """
        if count < 0:
            count = len(self) - start
        
        cdef int resolved_threads = self._resolve_num_threads(num_threads)
        
        # Call C++ batch read method
        cdef vector[CppJazelleEvent] cpp_events = self.cpp_obj.get().readEventsBatch(
            start, count, resolved_threads
        )
        
        # Convert to Python list
        cdef list py_events = []
        cdef JazelleEvent py_event
        cdef size_t i
        
        for i in range(cpp_events.size()):
            py_event = JazelleEvent.__new__(JazelleEvent)
            py_event.cpp_event = move(cpp_events[i])
            py_events.append(py_event)
        
        return py_events

    # ========================================================================
    # Iteration Methods
    # ========================================================================
    
    def iterate(
        self,
        batch_size: int = 100,
        num_threads: Optional[int] = None
    ):
        """
        Flexible iterator with optional batching and parallelization.
        
        This is the unified iteration method that replaces both sequential
        iteration and parallel batching. Behavior depends on batch_size:
        
        - batch_size = 1: Sequential iteration (no parallelization overhead)
        - batch_size > 1: Parallel batch iteration
        
        Parameters
        ----------
        batch_size : int, default 1
            Number of events per iteration:
            - 1: Yield single events sequentially (most memory-efficient)
            - >1: Yield batches using parallel reading
        num_threads : int or None, optional
            Number of threads for parallel reading (only used if batch_size > 1)
            None uses instance default
        
        Yields
        ------
        JazelleEvent or list of JazelleEvent
            - If batch_size=1: Single event
            - If batch_size>1: List of events (batch)
        
        Examples
        --------
        >>> # Sequential iteration (most efficient for one-by-one processing)
        >>> with JazelleFile('data.jazelle') as f:
        ...     for event in f.iterate():
        ...         print(event.ieventh.run, event.ieventh.event)
        
        >>> # Parallel batch iteration
        >>> with JazelleFile('data.jazelle', num_threads=8) as f:
        ...     for batch in f.iterate(batch_size=1000):
        ...         # Process 1000 events at a time
        ...         for event in batch:
        ...             process(event)
        
        >>> # Override threading for this iteration
        >>> with JazelleFile('data.jazelle') as f:
        ...     for batch in f.iterate(batch_size=500, num_threads=16):
        ...         analyze_batch(batch)
        
        Notes
        -----
        - For batch_size=1, this is equivalent to __iter__() and uses
          sequential reading regardless of num_threads
        - For large files, batch_size=100-1000 often provides good
          balance between memory and parallelization benefits
        """
        cdef JazelleEvent event
        cdef int total
        cdef int start
        cdef int count
        cdef int resolved_threads
        
        if batch_size <= 1:
            # Sequential iteration (no parallelization overhead)
            self.rewind()
            while True:
                event = JazelleEvent()
                if not self.cpp_obj.get().nextRecord(event.cpp_event):
                    break
                yield event
        else:
            # Parallel batch iteration
            total = len(self)
            start = 0
            resolved_threads = self._resolve_num_threads(num_threads)
            
            while start < total:
                count = min(batch_size, total - start)
                yield self.read_batch(start, count, resolved_threads)
                start += count

    # ========================================================================
    # Dictionary Conversion
    # ========================================================================
    
    def to_dict(
        self,
        layout: Literal['columnar', 'jagged'] = 'columnar',
        max_events: Optional[int] = None,
        batch_size: int = 1000,
        num_threads: Optional[int] = None
    ) -> Dict:
        """
        Convert file to dictionary format with flexible parallelization.
        
        Parameters
        ----------
        layout : {'columnar', 'jagged'}, default 'columnar'
            The structural layout of the output dictionary:
            
            - 'columnar' : Columnar arrays with offset tracking (recommended)
              Best for performance, HDF5, and vectorization.
              
            - 'jagged': 
              Best for interactive analysis. 
              Returns a list of arrays (one per event).
            
        max_events : int, optional
            Maximum number of events to process (None for all)
            
        batch_size : int, default 1000
            Events per batch for parallel processing:
            - 0 or 1: Sequential processing (no parallelization)
            - >1: Parallel batch processing
            
        num_threads : int or None, optional
            Number of threads (None uses instance default)
            Set to 0 for auto-detect, negative to force sequential
        
        Returns
        -------
        dict
            File data in specified format
        
        Examples
        --------
        >>> # Recommended: Offset format for production
        >>> with JazelleFile('data.jazelle', num_threads=8) as f:
        ...     data = f.to_dict(output_format='columnar')
        ...     # Uses instance default (8 threads), batch_size=100
        ...     
        ...     # Access event 5
        ...     from jazelle import get_event
        ...     event5 = get_event(data, 'PHCHRG', 5)
        
        >>> # Interactive: Jagged format for easy access
        >>> with JazelleFile('data.jazelle') as f:
        ...     data = f.to_dict(
        ...         layout='jagged',
        ...         batch_size=1000,
        ...         num_threads=16  # Override instance default
        ...     )
        ...     event5_px = data['PHCHRG']['px'][5]  # Direct access
        
        >>> # Sequential processing (explicit)
        >>> with JazelleFile('data.jazelle') as f:
        ...     data = f.to_dict(batch_size=0)  # No parallelization
        
        See Also
        --------
        iterate : For processing events without loading all into memory
        read_batch : For reading specific event ranges
        
        Notes
        -----
        - The 'columnar' format is recommended for most use cases
        - For files <1000 events, sequential processing is often faster
        - For files >10k events, parallel processing with batch_size=100-1000
          provides significant speedup
        - Memory usage scales with batch_size * num_threads
        """
        cdef int n_threads = self._resolve_num_threads(num_threads)
        cdef int total = self.getTotalEvents()
        cdef int start = 0
        cdef int count_to_read
        cdef int total_processed = 0

        if layout not in ['columnar', 'jagged']:
            raise ValueError(
                f"Invalid layout '{layout}'. Must be 'columnar' or 'jagged'."
            )
        
        if max_events is not None:
            total = min(total, max_events)

        accumulators = defaultdict(list)
        cdef std_map[string_view, BatchExtractor].iterator it
        cdef string_view bank_name
        cdef BatchExtractor func_ptr
        cdef vector[pxd.CppJazelleEvent] cpp_batch
        cdef dict batch_data

        cdef EventBatchWrapper batch_wrapper = EventBatchWrapper()
        while start < total:
            count_to_read = min(batch_size, total - start)
            
            # 1. Read C++ Batch (Parallel)
            cpp_batch = self.cpp_obj.get().readEventsBatch(start, count_to_read, n_threads)
            
            if cpp_batch.empty(): break

            it = _batch_extractors.begin()
            while it != _batch_extractors.end():
                bank_name = deref(it).first
                func_ptr = deref(it).second
                
                # Call the cdef function via pointer
                batch_data = func_ptr(&cpp_batch)
                
                # Store results (PyStr creation happens here)
                py_name = (<bytes>bank_name).decode('utf-8')
                for k, v in batch_data.items():
                    accumulators[f"{py_name}_{k}"].append(v)
                
                inc(it) # Next extractor

            start += cpp_batch.size()
            total_processed += cpp_batch.size()
            
            # Clear immediately to free memory
            cpp_batch.clear()

        # 3. Stitch & Format
        cdef dict flat_arrays = {}
        cdef dict family_offsets = {}
        cdef str key, fam, attr
        
        # Stitch chunks together
        for key, arr_list in accumulators.items():
            fam, attr = key.rsplit('_', 1)
            
            # Determine if concatenation is needed
            if len(arr_list) == 1:
                data = arr_list[0]
            else:
                data = np.concatenate(arr_list)

            if attr == '_offsets':
                family_offsets[fam] = self._stitch_offsets(arr_list)
            else:
                if fam not in flat_arrays: flat_arrays[fam] = {}
                flat_arrays[fam][attr] = data

        # Build final structure based on format
        cdef dict final_output = {}
        
        for fam, attrs in flat_arrays.items():
            final_output[fam] = {}
            
            # IEVENTH is scalar, always just arrays
            if fam == 'IEVENTH':
                final_output[fam] = attrs
                continue

            if layout == 'columnar':
                final_output[fam] = attrs
                if fam in family_offsets:
                    final_output[fam]['_offsets'] = family_offsets[fam]
            
            elif layout == 'jagged':
                if fam in family_offsets:
                    # Use offsets to split the big array into list of event-arrays
                    offsets = family_offsets[fam]
                    # Calculate split indices from offsets (offsets is [0, 5, 10...])
                    # Indices needed are [5, 10...]
                    indices = offsets[1:-1]
                    
                    for attr_name, big_arr in attrs.items():
                        final_output[fam][attr_name] = np.split(big_arr, indices)
                else:
                    final_output[fam] = attrs
            
        return final_output

    cdef object _stitch_offsets(self, list offsets_list):
        if not offsets_list: return np.array([0], dtype=np.int64)
        cdef list result = [offsets_list[0]]
        cdef int64_t last_val = offsets_list[0][-1]
        for arr in offsets_list[1:]:
            result.append(arr[1:] + last_val)
            last_val += arr[-1]
        return np.concatenate(result)
        
    @property
    def metadata(self) -> dict:
        """
        Retrieve metadata from the Jazelle file header.

        Returns
        -------
        dict
            Dictionary containing file header information:
            - filename: Original internal filename
            - creation_date: ISO 8601 formatted string
            - modified_date: ISO 8601 formatted string
            - last_record_type: Type of the last record processed
        """
        def fmt_date(dt):
            return dt.isoformat() if dt else None

        return {
            'filename': self.fileName,
            'creation_date': fmt_date(self.creationDate),
            'modified_date': fmt_date(self.modifiedDate),
            'last_record_type': self.lastRecordType
        }

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

cdef void _init_extractors():
    """
    Populate the C++ function pointer map.
    This runs once when the module is imported.
    """
    # Single source of truth for active banks
    register_extractor(b"IEVENTH", IEVENTH.extract_from_vector)
    register_extractor(b"MCHEAD",  MCHEAD.extract_from_vector)
    register_extractor(b"MCPART",  MCPART.extract_from_vector)
    register_extractor(b"PHPSUM",  PHPSUM.extract_from_vector)
    register_extractor(b"PHCHRG",  PHCHRG.extract_from_vector)
    register_extractor(b"PHKLUS",  PHKLUS.extract_from_vector)
    register_extractor(b"PHWIC",  PHWIC.extract_from_vector)
    register_extractor(b"PHCRID",  PHCRID.extract_from_vector)
    register_extractor(b"PHKTRK",  PHKTRK.extract_from_vector)
    register_extractor(b"PHKELID",  PHKELID.extract_from_vector)

# Run initialization immediately
_register_wrappers()
_init_extractors()