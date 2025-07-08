# -*- coding: utf-8 -*-

#pip install mpi4py

# Commented out IPython magic to ensure Python compatibility.
# import matplotlib.pyplot as plt
#from signal import signal, SIGPIPE, SIG_DFL
#signal(SIGPIPE,SIG_DFL) 
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
deltaL = delta(TL, TcL, deltacL)
deltaR = delta(TR, TcR, deltacR) 
#deltaL1 = delta(TR, TcL, deltacL)
#deltaR1 = delta(TL, TcR, deltacR)
V0 = 1e-6 # 
#t = np.linspace(0,1,1000) #1000
Omega = 1.5 * pi *1e9
Omega0 = 1.5 * pi *np.linspace(1e12,1e13,200)#1000 1e9 1e6
T = 2 * pi/Omega
t = np.linspace(0,T,200) 
L = np.size(t) # row size
K = np.size(Omega0) # column size 

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
  return (x-muL/deltac)*N1(x-muL/deltac, GammaL/deltac, deltaL/deltac) * np.heaviside(muR,0)* N1(x-muR/deltac, GammaR/deltac, deltaR/deltac)*(f((x-muL/deltac)*deltac/(kB*TL))-f((x-muR/deltac)*deltac/(kB*TR))) 
# Interference
def dQint(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):
  return (x-muL/deltac)*M(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*M(x-muR/deltac, GammaR/deltac, deltaR/deltac)*(f((x-muL/deltac)*deltac/(kB*TL))-f((x-muR/deltac)*deltac/(kB*TR)))
# Cooper pairs
def dQj1(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):
  return (x-muL/deltac)*M(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*F(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muL/deltac)*deltac/(2*kB*TL))#+ImF(x-muL/deltac, GammaL/deltac, deltaL/deltac) * ReF(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR)))
def dQj2(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): 
  return (x-muL/deltac)*F(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*M(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR))
def dQr1(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):#*np.heaviside(muR,0)
    return (x-muL/deltac)*N(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*P(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muL/deltac)*deltac/(2*kB*TL))
def dQr2(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):#*np.heaviside(muR,0)
    return (x-muL/deltac)*N(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*P(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR))

if rank == 0:
  V = V0 * np.sin(Omega*t)
    #  V = np.zeros((L, K), dtype=float)# Voltage
#  for i in range(L):
#    for j in range(K):
#      V[i,j] = V0 * np.sin(Omega[j]*t[i,j])
  Vr = np.zeros((L, K), dtype=float)# Voltage
  for i in range(L):
    for j in range(K):
      Vr[i,j] = V0 * np.cos(Omega0[j]*t[i])
  phi = 2*e*V0/(hbar*Omega) * (1. - np.cos(Omega*t))
#  phi = np.zeros((L, K), dtype=float) # Superconducting Phase
#  for i in range(L):
#    for j in range(K):
#      phi[i,j] = 2*e*V0/(hbar*Omega[j]) * (1. - np.cos(Omega[j]*t[i,j]))
  muL = 0*V                    # J
  muR = e*V                    # J      
  Vg = V + hbar*Omega0/e
  mul = 0 * Vg 
  mur = e * Vg
  QqpF = np.zeros((L,K), dtype=float)
  QintF = np.zeros((L,K), dtype=float)
  Qj1F = np.zeros((L,K), dtype=float)
  Qj2F = np.zeros((L,K), dtype=float)
  Qr1F = np.zeros((L,K), dtype=float)
  Qr2F = np.zeros((L,K), dtype=float)
else:
  V,phi,muL,muR,QqpF, QintF = None, None, None, None, None, None
  Qj1F, Qj2F = None, None
  Qr1F, Qr2F = None, None
  Vg, mul, mur = None, None, None

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
QqpF_part = np.zeros(count[rank]*K, dtype=float)
QintF_part = np.zeros(count[rank]*K, dtype=float)
Qj1F_part = np.zeros(count[rank]*K, dtype=float)
Qj2F_part = np.zeros(count[rank]*K, dtype=float)
Qr1F_part = np.zeros(count[rank]*K, dtype=float)
Qr2F_part = np.zeros(count[rank]*K, dtype=float)

mul_part = np.zeros(count[rank]*K,dtype=float)
mur_part = np.zeros(count[rank]*K,dtype=float)

arg1, arg2 = count * K * 2 ,  displ * K * 2
# #comm.Scatterv([Qqp.reshape(-1) if rank == 0 else None, arg1, arg2, MPI.FLOAT], Qqp_part, root=0)
comm.Scatterv([np.tile(mul,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], mul_part, root=0)
comm.Scatterv([np.tile(mur,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], mur_part, root=0)
# 
for i in range(int(K*count[rank])): #/size
    QqpF_part[i] = integrate.quad(dQqp,-2,2,args=(GammaL, GammaR, deltaL, deltaR, TL, mul_part[i], TR, mur_part[i], deltac), points = [deltaL/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
    QintF_part[i] = integrate.quad(dQint,-2,2,args=(GammaL, GammaR, deltaL, deltaR, TL, mul_part[i], TR, mur_part[i], deltac), points = [deltaL/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
    Qj1F_part[i] = integrate.quad(dQj1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL, deltaR, TL, mul_part[i], TR, mur_part[i], deltac), points = [deltaL/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
    Qj2F_part[i] = integrate.quad(dQj2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL, deltaR, TL, mul_part[i], TR, mur_part[i], deltac), points = [deltaL/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
    Qr1F_part[i] = integrate.quad(dQr1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL, deltaR, TL, mul_part[i], TR, mur_part[i], deltac), points = [deltaL/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
    Qr2F_part[i] = integrate.quad(dQr2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL, deltaR, TL, mul_part[i], TR, mur_part[i], deltac), points = [deltaL/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]

comm.Barrier()
arg = displ * K * 2 if rank == 0 else K * 2 # * 2
comm.Gatherv(QqpF_part, [QqpF, count * K * 2, arg, MPI.FLOAT], root=0)      
comm.Gatherv(QintF_part, [QintF, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj1F_part, [Qj1F, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj2F_part, [Qj2F, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr1F_part, [Qr1F, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr2F_part, [Qr2F, count * K * 2, arg, MPI.FLOAT], root=0)

#comm.Barrier()
#print(f'rank:{rank}', Qqp_part,'\n') # img_part.shape, 
#comm.Barrier()
 
if rank == 0:
#   data = Qqp #np.column_stack([r, QL_avg, QR_avg1])
#   datafile_path = "Qqp.txt"
   #data = Qint
   #datafile_path = "Qint.txt"
   QjF = 0.5*(Qj1F + Qj2F)
   QrF = 0.5*(Qr1F + Qr2F)
   #Qfw = (Qqp + Qj*np.sin(phi) + Qint*np.cos(phi))
   #data = Qfw
   #datafile_path = "Qfw.txt"
   dtqp, dtj, dti, dtr = QqpF,QjF,QintF,QrF 
   dtqp_path, dtj_path, dti_path, dtr_path = "QqpF.txt","QjF.txt","QintF.txt","QrF.txt"
#   np.savetxt(datafile_path , data)
   np.savetxt(dtqp_path , dtqp)
   np.savetxt(dtj_path , dtj)
   np.savetxt(dti_path , dti)
   np.savetxt(dtr_path , dtr)
# 
comm.Barrier()
t_diff = MPI.Wtime() - t_start
if comm.rank==0: 
    print("Execution time", t_diff)
