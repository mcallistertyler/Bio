
import numpy as np
import scipy
import matcompat
import random
anslist = []
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def sphere(x):
    global anslist
    # Local Variables: y, x, s, j, n
    # Function calls: sphere
    #% 
    #% Sphere function 
    #% Matlab Code by A. Hedar (Nov. 23, 2005).
    #% The number of variables n should be adjusted below.
    #% The default value of n = 30.
    #% 
    n = 2.
    s = 0.
    for j in np.arange(1., (n)+1):
        s = s+x[int(j)-1]**2.
        print "X",x, "\n"   
    y = s
    #print random.randint(1,30)
    
    print "Ans", y

#Generate some dummy datacd Doc*/
dumblist = []
for _ in range(0, 30):
    dumblist.append([random.uniform(0.1, 0.5),random.uniform(0.1, 0.5)])
print dumblist
for i in range(0, len(dumblist)):
    sphere(dumblist[i])