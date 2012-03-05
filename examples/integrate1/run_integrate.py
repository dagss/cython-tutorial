import numpy as np
import time

import integrate

outer_repeats = 2
inner_repeats = 500
n = 10000

dt_list = []
for i in range(outer_repeats):
    t0 = time.clock()
    for j in range(inner_repeats):
        integrate.integrate_f(.1, .3, n)
    dt = (time.clock() - t0) / inner_repeats # mean
    dt_list.append(dt)

dt_min = np.min(dt_list) # min
print '%.2f us' % (dt_min * 1e6)

# Uncomment to raise exceptionxs
#print integrate.integrate_f(0, 1, 10)
