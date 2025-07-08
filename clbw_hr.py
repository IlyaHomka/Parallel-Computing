# -*- coding: utf-8 -*-

#pip install mpi4py

# Commented out IPython magic to ensure Python compatibility.
# import matplotlib.pyplot as plt
# %%writefile Qqp.py
from mpi4py import MPI
import numpy as np
import sys
import os
# # Suppress warnings
# import warnings
# warnings.filterwarnings('ignore'),
import scipy
import scipy.integrate as integrate 

comm = MPI.COMM_WORLD
size = comm.size
rank = comm.Get_rank() 

comm.Barrier()
t_start = MPI.Wtime() 

#Physical constants
pi = np.pi
hbar = 6.626 /(2*pi)*1e-34    #Planck constant J.s
e = 1.6*1e-19                 #elementary charge C
me = 9.31*1e-31               #electron mass kg
m = 2 * me                    #Cooper paris charge C            
q = 2 * e                     #Cooper paris mass kg
kB = 1.38*1e-23               #Boltzmann constant J.K^-1 

# Junction characteristics
TcL = 1.4                                    # K
TcR = 0.75*TcL                               # K 
deltacL = 1.764*kB*TcL                       # J   
deltacR = 1.764*kB*TcR                       # J 
TL = 0.77*TcL #TcL                           # K
TR = 0.01*TcL #TcL                           # K                                
deltac = 1.764*kB*TcL                        # J 
GammaL = 1e-4*deltacL                        # Dynes parameter 1e-4
GammaR = 1e-4*deltacR                        # Dynes parameter  

# Superconducting gap
delta = lambda T, Tc, deltac: deltac*np.tanh(1.74*abs(Tc/T-1)**0.5)
#deltaL = delta(TL, TcL, deltacL)
#deltaR = delta(TR, TcR, deltacR) 
deltaL1 = delta(TR, TcL, deltacL)
deltaR1 = delta(TL, TcR, deltacR)
V0 = 1e-6 # 
t = np.linspace(0,1,100)
Omega = 1.5 * pi *np.linspace(1e6,1e9,300) 
L = np.size(t) # row size
K = np.size(Omega) # column size 

# Functions
# Fermi-Dirac destribution
f = lambda x: 1./(1. + np.exp(x)) 
# Density of states
N1 = lambda x, y, t: -(((x+1j*y)/(t**2-(x+1j*y)**2)**0.5).imag) #Im
P = lambda x, y, t: -(((x+1j*y)/(t**2-(x+1j*y)**2)**0.5).real) # Re
# Density of states
N = lambda x, y, t: abs(((x+1j*y)/((x+1j*y)**2-t**2)**0.5).real) 
# # Anomalous Gree function: Imaginary part
def M(x, y, t):
  Z = x+1j*y;  
  return (t/(t**2-Z**2)**0.5).imag
# Anomalous Gree function: Real part
def F(x, y, t):
  Z = x+1j*y;  
  return (t/(t**2-Z**2)**0.5).real  

# dimensionless integrands for kinetic integrals expressed with reduced variable x = E/eV 
# Quasiparticles
def dQqp(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): 
  return (x-muL/deltac)*N(x-muL/deltac, GammaL/deltac, deltaL/deltac) * np.heaviside(muR,0)* N(x-muR/deltac, GammaR/deltac, deltaR/deltac)*(f((x-muL/deltac)*deltac/(kB*TL))-f((x-muR/deltac)*deltac/(kB*TR))) 
# Interference
def dQint(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):
  return (x-muL/deltac)*M(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*M(x-muR/deltac, GammaR/deltac, deltaR/deltac)*(f((x-muL/deltac)*deltac/(kB*TL))-f((x-muR/deltac)*deltac/(kB*TR)))
# Cooper pairs
def dQj1(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):
  return (x-muL/deltac)*M(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*F(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muL/deltac)*deltac/(2*kB*TL))#+ImF(x-muL/deltac, GammaL/deltac, deltaL/deltac) * ReF(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR)))
def dQj2(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): 
  return (x-muL/deltac)*F(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*M(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR))

if rank == 0:
  V = np.zeros((L, K), dtype=float)# Voltage
  for i in range(L):
    for j in range(K):
      V[i,j] = V0 * np.sin(Omega[j]*t[i])
  phi = np.zeros((L, K), dtype=float) # Superconducting Phase
  for i in range(L):
    for j in range(K):
      phi[i,j] = 2*e*V0/(hbar*Omega[j]) * (1. - np.cos(Omega[j]*t[i]))
  muL = 0*V                    # J
  muR = e*V                    # J      
  QqpR = np.zeros((np.size(t), np.size(Omega)), dtype=float)
  QintR = np.zeros((np.size(t), np.size(Omega)), dtype=float)
  QjR1 = np.zeros((np.size(t), np.size(Omega)), dtype=float)
  QjR2 = np.zeros((np.size(t), np.size(Omega)), dtype=float)
else:
  V,phi,muL,muR,Qqp, Qint = None, None, None, None, None, None
  QjR1, QjR2 = None, None

displ = np.empty(size, dtype=int)
count = np.empty(size, dtype=int) 

if rank == 0:
   quot, rem = divmod(L, size)
   
   for r in range(size):
      count[r] = quot + 1 if r < rem else quot # count: the size of each sub-task
      displ = np.array([np.sum(count[:r]) for r in range(size)]) # displacement: the starting index of each sub-task 
comm.Bcast(count, root=0)
comm.Bcast(displ, root=0)
# # comm.Bcast(deltaL, root=0)
# # comm.Bcast(deltaR, root=0)
# # comm.Bcast(TL, root=0)

# comm.Bcast(TR, root=0)
#  
QqpR_part = np.zeros(count[rank]*K, dtype=float)
QintR_part = np.zeros(count[rank]*K, dtype=float)
QjR1_part = np.zeros(count[rank]*K, dtype=float)
QjR2_part = np.zeros(count[rank]*K, dtype=float)

muL_part = np.zeros(count[rank]*K,dtype=float)
muR_part = np.zeros(count[rank]*K,dtype=float)

arg1, arg2 = count * K * 2 ,  displ * K * 2
# #comm.Scatterv([Qqp.reshape(-1) if rank == 0 else None, arg1, arg2, MPI.FLOAT], Qqp_part, root=0)
comm.Scatterv([muL.reshape(-1) if rank == 0 else None, arg1, arg2, MPI.FLOAT], muL_part, root=0)
comm.Scatterv([muR.reshape(-1) if rank == 0 else None, arg1, arg2, MPI.FLOAT], muR_part, root=0)
# 
for i in range(int(K*count[rank])): #/size
    QqpR_part[i] = integrate.quad(dQqp,-2,2,args=(GammaL, GammaR, deltaL1, deltaR1, TR, muL_part[i], TL, muR_part[i], deltac), points = [deltaL1/deltac, deltaR1/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
    QintR_part[i] = integrate.quad(dQint,-2,2,args=(GammaL, GammaR, deltaL1, deltaR1, TR, muL_part[i], TL, muR_part[i], deltac), points = [deltaL1/deltac, deltaR1/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
    QjR1_part[i] = integrate.quad(dQj1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL1, deltaR1, TR, muL_part[i], TL, muR_part[i], deltac), points = [deltaL1/deltac, deltaR1/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
    QjR2_part[i] = integrate.quad(dQj2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL1, deltaR1, TR, muL_part[i], TL, muR_part[i], deltac), points = [deltaL1/deltac, deltaR1/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
comm.Barrier()
arg = displ * K * 2 if rank == 0 else K * 2 # * 2
comm.Gatherv(QqpR_part, [QqpR, count * K * 2, arg, MPI.FLOAT], root=0)      
comm.Gatherv(QintR_part, [QintR, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(QjR1_part, [QjR1, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(QjR2_part, [QjR2, count * K * 2, arg, MPI.FLOAT], root=0)
#comm.Barrier()
#print(f'rank:{rank}', Qqp_part,'\n') # img_part.shape, 
#comm.Barrier()
 
if rank == 0:
#   #print("Qqp \n", Qqp)
#   data = Qqp #np.column_stack([r, QL_avg, QR_avg1])
#   datafile_path = "Qqp.txt"
   #data = Qint
   #datafile_path = "Qint.txt"
   QjR = 0.5*(QjR1 + QjR2)
   Qbw = -QqpR + QjR*np.sin(phi) - QintR*np.cos(phi)
   data = Qbw
   datafile_path = "Qbw.txt"
   dtqp, dtj, dti = QqpR, QjR, QintR 
   dtqp_path, dtj_path, dti_path = "QqpR.txt","QjR.txt","QintR.txt"
   np.savetxt(datafile_path , data)
   np.savetxt(dtqp_path , dtqp)
   np.savetxt(dtj_path , dtj)
   np.savetxt(dti_path , dti)
# 
comm.Barrier()
t_diff = MPI.Wtime() - t_start
if comm.rank==0: 
  print("Execution time", t_diff)
