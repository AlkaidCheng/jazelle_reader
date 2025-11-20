from libc.time cimport time_t

cdef extern from "<chrono>" namespace "std::chrono" nogil:
    cdef cppclass system_clock:
        pass
    
    # Declare time_point in the system_clock namespace
    cdef cppclass time_point "std::chrono::system_clock::time_point":
        pass
    
    # Declare to_time_t as a standalone function
    time_t to_time_t "std::chrono::system_clock::to_time_t"(time_point) nogil