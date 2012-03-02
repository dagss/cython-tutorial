from __future__ import division


cimport cython

@cython.cdivision(True)
cpdef double f(double x) except *:
    return 1.0 / (x * x * x + 2 * x * x)

def integrate_f(double a, double b, ssize_t N):
    cdef:
        double s
        double dx
        ssize_t i

    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx

