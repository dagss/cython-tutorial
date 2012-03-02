from __future__ import division

cdef class DoubleFunction:
    cpdef double evaluate(self, double x) except *:
        raise NotImplementedError()

def integrate(DoubleFunction func, double a, double b, ssize_t N):
    cdef:
        double s, dx
        ssize_t i
    if func is None:
        raise ValueError("If we didn't raise this we would crash, at best")
    s = 0
    dx = (b - a ) / N
    for i in range ( N ):
        s += func.evaluate(a + i * dx)
    return s * dx

