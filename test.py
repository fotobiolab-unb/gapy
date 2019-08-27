import numpy as np
#from gapy2.gapy import GA
from gapy3.gapy import GA
import os
import matplotlib.pyplot as plt
import time

hist = []
timer = []

mask = np.array([[-10.,10.] for i in range(4)])

def parab(x):
    U = 53.0-(x**2.0).sum(axis=1)
    return U

names = ["1","53","106.67"]

functions = [
    lambda x: (np.cos(10.0*x.sum(axis=1))**2.0)/(1+x.sum(axis=1)**2.0),
    lambda x: 53.0-(x**2.0).sum(axis=1),
    lambda x: -(x.sum(axis=1)**4)/4+8*x.sum(axis=1)**2+(1/3.0)*x.sum(axis=1)**3-16*x.sum(axis=1)
]


for f,l in zip(functions,names):
    t1 = time.time()
    ga = GA(
            population_size=8,
            chromosome_size=4,
            resolution=8,
            iterations=10000,
            elitism=True,
            mutation=0.2,
            fitness=f,
            range_mask=mask,
            has_mask=True,
            time_print=.5
            )
    ga.G[0]=np.zeros(ga.G[0].shape)
    ga.run()
    hist.append(ga.F[0])
    t2 = time.time()
    timer.append(t2-t1)

print("Done ----------------------------------------")
print("Optimal found:")
print(" ".join(list(map(str,hist))))
print("Actual values:")
print(" ".join(names))

#
# plt.figure(0)
# plt.plot(hist,"o")
# plt.figure(1)
# plt.plot(timer)
# plt.show()
