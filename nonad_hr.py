# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import math 
import cmath as cth
import sympy as sp
from sympy import *
pi = math.pi
import numpy as np
import sys
# Suppress warnings
import warnings
warnings.filterwarnings('ignore'),
import scipy.integrate as integrate
#from sympy import symbols, integrate
import scipy.special as special

from numpy import inf
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#Physical constants
hbar = 6.626 /(2*pi)*1e-34  #Planck constant J.s
e = 1.6*1e-19               #elementary charge C
me = 9.31*1e-31             #electron mass kg
m = 2 * me                  #Cooper paris charge C            
q = 2 * e                   #Cooper paris mass kg
kB=1.38*1e-23               #Boltzmann constant J.K^-1

# Circuit parameters
GT = 1./1000 # Ohm^(-1)
L = 1*1e-9 # Hn
C = 100*1e-15 # F
R = 1e3 # Ohm
Ph0 = 6.626 /q*1e-34

Omega = 1.5 * pi * 1e6
T = 2 * pi/Omega
t = np.linspace(0,T,200) #T
r = np.linspace(0,1,200)

# Voltage
V0 = 1e-6 # V
Omega0 = 1.5 * pi * 1e12 #
V1 = 1e-6# V
V = V0 * np.sin(Omega*t)
phi = 2*e*V0/(hbar*Omega) * (1. - np.cos(Omega*t))
Vr = V1 * np.cos(Omega0*t)  
muL = 0*V#                     
muR = e*V#
Vg = V + hbar*Omega0/e
mul = 0 * Vg 
mur = e * Vg

# Junction characteristics
TcL = 1.4 #1.6                               # K
TcR = 0.75*TcL#*r #0.77#                     # K 
deltacL = 1.764*kB*TcL                       # J   
deltacR = 1.764*kB*TcR                       # J 
TL = r*TcL #r*TcL# 0.77                      # K
TR = 0.01*TcL                                # K                                
deltac = 1.764*kB*TcL                        # J 
GammaL = 1e-4*deltacL                        # Dynes parameter 
GammaR = 1e-4*deltacR                        # Dynes parameter

delta = lambda T, Tc, deltac: deltac*np.tanh(1.74*abs(Tc/T-1)**0.5)
deltaL = delta(TL, TcL, deltacL)
deltaR = delta(TR, TcR, deltacR)

f = lambda x: 1./(1. + np.exp(x))

# Density of states
#N = lambda x, y, t: abs(((x+1j*y)/((x+1j*y)**2-t**2)**0.5).real)
N = lambda x, y, t: -(((x+1j*y)/(t**2-(x+1j*y)**2)**0.5).imag)
P = lambda x, y, t: -(((x+1j*y)/(t**2-(x+1j*y)**2)**0.5).real)

# Anomalous Gree function: Imaginary part
def M(x, y, t):
  Z = x+1j*y;  
  return (t/(t**2-Z**2)**0.5).imag#*np.sign(x)
# Anomalous Gree function: Real part
def F(x, y, t):
  Z = x+1j*y;  
  return (t/(t**2-Z**2)**0.5).real#*np.sign(x)

# dimensionless integrands for kinetic integrals
# expressed with reduced variable z = E/eV 
def dQqp(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): #np.heaviside(muL,0)*
     return (x-muL/deltac)*N(x-muL/deltac, GammaL/deltac, deltaL/deltac) * np.heaviside(muR,0)* N(x-muR/deltac, GammaR/deltac, deltaR/deltac)*(f((x-muL/deltac)*deltac/(kB*TL))-f((x-muR/deltac)*deltac/(kB*TR))) 
# dimensionless integrands for kinetic integrals 
# expressed with reduced variable x=E/deltac
def dQint(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): #np.heaviside(muL,0)*
    return (x-muL/deltac)*M(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*M(x-muR/deltac, GammaR/deltac, deltaR/deltac)*(f((x-muL/deltac)*deltac/(kB*TL))-f((x-muR/deltac)*deltac/(kB*TR)))# dimensionless integrands for kinetic integrals
# expressed with reduced variable x=E/deltac
def dQj1(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):
    return (x-muL/deltac)*M(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*F(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muL/deltac)*deltac/(2*kB*TL))#+ImF(x-muL/deltac, GammaL/deltac, deltaL/deltac) * ReF(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR)))
def dQj2(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): #L
    return (x-muL/deltac)*F(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*M(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR))# dimensionless integrands for kinetic integrals
# expressed with reduced variable x=E/deltac
def dQr1(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):
    return (x-muL/deltac)*N(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*P(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muL/deltac)*deltac/(2*kB*TL))#+ImF(x-muL/deltac, GammaL/deltac, deltaL/deltac) * ReF(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR)))
def dQr2(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): 
    return (x-muL/deltac)*N(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*P(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR))comm.Barrier()
t_start = MPI.Wtime() 

L = np.size(r) # row size
K = np.size(t) # column size 
   
if rank == 0:
   Qqp = np.zeros((L, K), dtype=float)
   Qint = np.zeros((L, K), dtype=float)
   Qj1 = np.zeros((L, K), dtype=float)
   Qj2 = np.zeros((L, K), dtype=float)
   QqpF = np.zeros((L, K), dtype=float)
   QintF = np.zeros((L, K), dtype=float)
   Qj1F = np.zeros((L, K), dtype=float)
   Qj2F = np.zeros((L, K), dtype=float)
   Qr1 = np.zeros((L, K), dtype=float)
   Qr2 = np.zeros((L, K), dtype=float)
   Qr1F = np.zeros((L, K), dtype=float)
   Qr2F = np.zeros((L, K), dtype=float)
else:
   Qqp, Qint, QqpF, QintF = None, None, None, None #, None, None
   Qj1, Qj2, Qj1F, Qj2F = None, None, None, None
   Qr1, Qr2, Qr1F, Qr2F = None, None, None, None
   
displ = np.empty(size, dtype=int)
count = np.empty(size, dtype=int) 

if rank == 0:
   quot, rem = divmod(L, size)
   
   for k in range(size):
      count[k] = quot + 1 if k < rem else quot # count: the size of each sub-task
      displ = np.array([np.sum(count[:k]) for k in range(size)]) # displacement: the starting index of each sub-task 
comm.Bcast(count, root=0)
comm.Bcast(displ, root=0)
# Arrays to fill
Qqp_part = np.zeros(count[rank]*K, dtype=float)
Qint_part = np.zeros(count[rank]*K, dtype=float)
Qj1_part = np.zeros(count[rank]*K, dtype=float)
Qj2_part = np.zeros(count[rank]*K, dtype=float)
QqpF_part = np.zeros(count[rank]*K, dtype=float)
QintF_part = np.zeros(count[rank]*K, dtype=float)
Qj1F_part = np.zeros(count[rank]*K, dtype=float)
Qj2F_part = np.zeros(count[rank]*K, dtype=float)
Qr1_part = np.zeros(count[rank]*K, dtype=float)
Qr2_part = np.zeros(count[rank]*K, dtype=float)
Qr1F_part = np.zeros(count[rank]*K, dtype=float)
Qr2F_part = np.zeros(count[rank]*K, dtype=float)

muL_part = np.zeros(count[rank]*K, dtype=float)
muR_part = np.zeros(count[rank]*K, dtype=float)
deltaL_part = np.zeros(count[rank]*K, dtype=float)
TL_part = np.zeros(count[rank]*K, dtype=float)
mul_part = np.zeros(count[rank]*K, dtype=float)
mur_part = np.zeros(count[rank]*K, dtype=float)


arg1, arg2 = count * K * 2,  displ * K * 2
comm.Scatterv([np.tile(muL,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], muL_part, root=0)
comm.Scatterv([np.tile(muR,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], muR_part, root=0) 
comm.Scatterv([np.tile(deltaL,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], deltaL_part, root=0) 
comm.Scatterv([np.tile(TL,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], TL_part, root=0)
comm.Scatterv([np.tile(mul,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], mul_part, root=0)
comm.Scatterv([np.tile(mur,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], mur_part, root=0) 

for i in range(int(K*count[rank])): 
   Qqp_part[i] = integrate.quad(dQqp,-2,2,args=(GammaL, GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
   Qint_part[i] = integrate.quad(dQint,-2,2,args=(GammaL, GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
   Qj1_part[i] = integrate.quad(dQj1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
   Qj2_part[i] = integrate.quad(dQj2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]

   QqpF_part[i] = integrate.quad(dQqp,-2,2,args=(GammaL, GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]

   QintF_part[i] = integrate.quad(dQint,-2,2,args=(GammaL, GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]

   Qj1F_part[i] = integrate.quad(dQj1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]

   Qj2F_part[i] = integrate.quad(dQj2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]  

   Qr1_part[i] = integrate.quad(dQr1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
   Qr2_part[i] = integrate.quad(dQr2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]

   Qr1F_part[i] = integrate.quad(dQr1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
   Qr2F_part[i] = integrate.quad(dQr2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]  
 
comm.Barrier()
arg = displ * K * 2 if rank == 0 else K * 2 # * 2
comm.Gatherv(Qqp_part, [Qqp, count * K * 2, arg, MPI.FLOAT], root=0)      
comm.Gatherv(Qint_part, [Qint, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj1_part, [Qj1, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj2_part, [Qj2, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(QqpF_part, [QqpF, count * K * 2, arg, MPI.FLOAT], root=0)      
comm.Gatherv(QintF_part, [QintF, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj1F_part, [Qj1F, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj2F_part, [Qj2F, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr1_part, [Qr1, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr2_part, [Qr2, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr1F_part, [Qr1F, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr2F_part, [Qr2F, count * K * 2, arg, MPI.FLOAT], root=0)

if rank == 0:
   Qj = 0.5*(Qj1 + Qj2)
   QjF = 0.5*(Qj1F + Qj2F)
   Qr = 0.5*(Qr1 + Qr2)
   QrF = 0.5*(Qr1F + Qr2F)
   Qfw1 = e*V0/(hbar*Omega0)*((QqpF-Qqp)*np.cos(-Omega0*t) + (QintF-Qint)*np.cos(phi-Omega0*t) + (QjF-Qj)*np.sin(phi-Omega0*t) +(QrF-Qr)*np.sin(-Omega0*t))
   Qfw2 = -e*V0/(hbar*Omega0)*((QqpF-Qqp)*np.cos(Omega0*t) + (QintF-Qint)*np.cos(phi+Omega0*t) + (QjF-Qj)*np.sin(phi+Omega0*t) +(QrF-Qr)*np.sin(Omega0*t))
   QfwT = Qfw1+Qfw2
   data = QfwT
   datafile_path = "QnL.txt"
   np.savetxt(datafile_path, data)
   dtqp, dtj, dti, dtr = Qqp, Qj, Qint, Qr 
   dtqp_path, dtj_path, dti_path, dtr_path = "Qmqp.txt","Qmj.txt","Qmint.txt","Qmr.txt"
   np.savetxt(datafile_path , data)
   np.savetxt(dtqp_path , dtqp)
   np.savetxt(dtj_path , dtj)
   np.savetxt(dti_path , dti)
   np.savetxt(dtr_path , dtr)
   dfqp, dfj, dfi, dfr = QqpF, QjF, QintF, QrF 
   dfqp_path, dfj_path, dfi_path, dfr_path = "Qfqp.txt","Qfj.txt","Qfint.txt","Qfr.txt"
   np.savetxt(dfqp_path , dfqp)
   np.savetxt(dfj_path , dfj)
   np.savetxt(dfi_path , dfi)
   np.savetxt(dfr_path , dfr)
# 
comm.Barrier()
t_diff = MPI.Wtime() - t_start
if comm.rank==0: 
  print("Execution time", t_diff)
