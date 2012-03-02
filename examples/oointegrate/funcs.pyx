cimport libc.math

cimport integrate

cdef class Sin(integrate.DoubleFunction):
    """
    a * sin(b * x)
    """
    cdef double a, b
    def __init__(self, double a, double b):
        self.a = a
        self.b = b

    cpdef double evaluate(self, double x) except *:
        return self.a * libc.math.sin(self.b * x)

