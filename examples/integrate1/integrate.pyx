from __future__ import division

def f(x):
    return 1 / (x**3 + 2*x**2)

def integrate_f(a, b, N):
    s = 0
    dx = (b - a ) / N
    for i in range ( N ):
        s += f(a + i * dx)
    return s * dx

