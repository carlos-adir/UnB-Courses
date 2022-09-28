import numpy as np
from CoolProp.CoolProp import PropsSI
from matplotlib import pyplot as plt

fluido = "Air"
Ps = [15e+4,4e+4,94082.090068594,68548.462880174,54899.605981277,109290.401866385]  # Pa
vs = [0.675,1.9,0.94222389,1.29319300,1.38434509,0.92643085]  # m^3/kg
ds = [1/v for v in vs]  # kg/m^3
Ts = [PropsSI("T", "P", P, "D", d, fluido) for P, d in zip(Ps, ds)]
ss = [PropsSI("S", "P", P, "D", d, fluido) for P, d in zip(Ps, ds)]
us = [PropsSI("U", "P", P, "D", d, fluido) for P, d in zip(Ps, ds)]
print("Ts = ", Ts)
print("ss = ", ss)
print("us = ", us)

npts = 11
Psample = np.linspace(min(Ps), max(Ps), npts)
vsample = np.linspace(min(vs), max(vs), npts)
ssample = np.linspace(min(ss), max(ss), npts)
Tsample = np.linspace(min(Ts), max(Ts), npts)
pairsdotted = [(0, 5), (4, 1), (5, 3), (2, 4)]
pairssolid = [(0, 2), (2, 3), (3, 1)]

dv = max(vs)-min(vs)
dP = max(Ps)-min(Ps)
dT = max(Ts)-min(Ts)
ds = max(ss)-min(ss)
fig1 = plt.figure()
ax1 = plt.gca()
fig2 = plt.figure()
ax2 = plt.gca()
for i in range(6):
    ax1.scatter(vs[i], Ps[i], label="%d" % (i+1))
    ax2.scatter(ss[i], Ts[i], label="%d" % (i+1))
for a, b in pairsdotted:
    ssample = np.linspace(ss[a], ss[b], npts)
    Tsample = np.linspace(Ts[a], Ts[b], npts)
    Psample = [PropsSI("P", "T", T, "S", s, fluido) for s, T in zip(ssample, Tsample)]
    dsample = [PropsSI("D", "T", T, "S", s, fluido) for s, T in zip(ssample, Tsample)]
    vsample = [1/d for d in dsample]
    ax1.plot(vsample, Psample, color="k", ls="dotted")
    ax2.plot(ssample, Tsample, color="k", ls="dotted")
for a, b in pairssolid:
    ssample = np.linspace(ss[a], ss[b], npts)
    Tsample = np.linspace(Ts[a], Ts[b], npts)
    Psample = [PropsSI("P", "T", T, "S", s, fluido) for s, T in zip(ssample, Tsample)]
    dsample = [PropsSI("D", "T", T, "S", s, fluido) for s, T in zip(ssample, Tsample)]
    vsample = [1/d for d in dsample]
    ax1.plot(vsample, Psample, color="k", ls="solid")
    ax2.plot(ssample, Tsample, color="k", ls="solid")
ax1.set_xlim(min(vs)-0.1*dv, max(vs)+0.1*dv)
ax1.set_ylim(min(Ps)-0.1*dP, max(Ps)+0.1*dP)
ax1.set_xlabel(r"Volume específico $v \ (m^3/kg)$")
ax1.set_ylabel(r"Pressão $P \ (Pa)$")
ax1.legend()
ax2.set_xlim(min(ss)-0.1*ds, max(ss)+0.1*ds)
ax2.set_ylim(min(Ts)-0.1*dT, max(Ts)+0.1*dT)
ax2.set_xlabel(r"Entropia específico $s \ (J/kg\cdot K)$")
ax2.set_ylabel(r"Temperatura $T \ (K)$")
ax2.legend()

plt.show()