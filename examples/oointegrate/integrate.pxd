
cdef class DoubleFunction:
    cpdef double evaluate(self, double x) except *
