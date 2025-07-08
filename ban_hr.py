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
t = np.linspace(0,T,200) #
r = np.linspace(0,1,200)

# Voltage
V0 = 1e-6 # V
Omega0 = 1.5 * pi * 1e12 #
V1 = 1e-6# V 

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
deltaL1 = delta(TR, TcL, deltacL)
deltaR1 = delta(TL, TcR, deltacR)

f = lambda x: 1./(1. + np.exp(x))

# Density of states
#N = lambda x, y, t: abs(((x+1j*y)/((x+1j*y)**2-t**2)**0.5).real)
N = lambda x, y, t: -(((x+1j*y)/(t**2-(x+1j*y)**2)**0.5).imag)
P = lambda x, y, t: -(((x+1j*y)/(t**2-(x+1j*y)**2)**0.5).real)

# Anomalous Green function: Imaginary part
def M(x, y, t):
  Z = x+1j*y;  
  return (t/(t**2-Z**2)**0.5).imag
# Anomalous Gree function: Real part
def F(x, y, t):
  Z = x+1j*y;  
  return (t/(t**2-Z**2)**0.5).real

# dimensionless integrands for kinetic integrals
# expressed with reduced variable z = E/eV 
def dQqp(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): #np.heaviside(muL,0)*
     return (x-muL/deltac)*N(x-muL/deltac, GammaL/deltac, deltaL/deltac) * np.heaviside(muR,0)* N(x-muR/deltac, GammaR/deltac, deltaR/deltac)*(f((x-muL/deltac)*deltac/(kB*TL))-f((x-muR/deltac)*deltac/(kB*TR))) 
# dimensionless integrands for kinetic integrals 
# expressed with reduced variable x=E/deltac
def dQint(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): #np.heaviside(muL,0)*
    return (x-muL/deltac)*M(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*M(x-muR/deltac, GammaR/deltac, deltaR/deltac)*(f((x-muL/deltac)*deltac/(kB*TL))-f((x-muR/deltac)*deltac/(kB*TR)))
# dimensionless integrands for kinetic integrals
# expressed with reduced variable x=E/deltac
def dQj1(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):
    return (x-muL/deltac)*M(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*F(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muL/deltac)*deltac/(2*kB*TL))#+ImF(x-muL/deltac, GammaL/deltac, deltaL/deltac) * ReF(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR)))
def dQj2(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): #L
    return (x-muL/deltac)*F(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*M(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR))
# dimensionless integrands for kinetic integrals
# expressed with reduced variable x=E/deltac
def dQr1(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac):
    return (x-muL/deltac)*N(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*P(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muL/deltac)*deltac/(2*kB*TL))#+ImF(x-muL/deltac, GammaL/deltac, deltaL/deltac) * ReF(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR)))
def dQr2(x, GammaL, GammaR, deltaL, deltaR, TL, muL, TR, muR, deltac): 
    return (x-muL/deltac)*N(x-muL/deltac, GammaL/deltac, deltaL/deltac)*np.heaviside(muR,0)*P(x-muR/deltac, GammaR/deltac, deltaR/deltac)*np.tanh((x-muR/deltac)*deltac/(2*kB*TR))

comm.Barrier()
t_start = MPI.Wtime() 

L = np.size(r) # row size
K = np.size(t) # column size 

V = V0 * np.sin(Omega*t)
phi = 2*e*V0/(hbar*Omega) * (1.-np.cos(Omega*t))
Vr = V1 * np.cos(Omega0*t)  
muL = 0*V#                     
muR = e*V#
Vg = V + hbar*Omega0/e
mul = 0 * Vg 
mur = e * Vg
   
if rank == 0:
   QqpR = np.zeros((L, K), dtype=float)
   QintR = np.zeros((L, K), dtype=float)
   Qj1R = np.zeros((L, K), dtype=float)
   Qj2R = np.zeros((L, K), dtype=float)
   QqpRF = np.zeros((L, K), dtype=float)
   QintRF = np.zeros((L, K), dtype=float)
   Qj1RF = np.zeros((L, K), dtype=float)
   Qj2RF = np.zeros((L, K), dtype=float)
   Qr1R = np.zeros((L, K), dtype=float)
   Qr2R = np.zeros((L, K), dtype=float)
   Qr1RF = np.zeros((L, K), dtype=float)
   Qr2RF = np.zeros((L, K), dtype=float)
else:
   QqpR, QintR, QqpRF, QintRF = None, None, None, None #, None, None
   Qj1R, Qj2R, Qj1RF, Qj2RF = None, None, None, None
   Qr1R, Qr2R, Qr1RF, Qr2RF = None, None, None, None
   
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
QqpR_part = np.zeros(count[rank]*K, dtype=float)
QintR_part = np.zeros(count[rank]*K, dtype=float)
Qj1R_part = np.zeros(count[rank]*K, dtype=float)
Qj2R_part = np.zeros(count[rank]*K, dtype=float)
QqpRF_part = np.zeros(count[rank]*K, dtype=float)
QintRF_part = np.zeros(count[rank]*K, dtype=float)
Qj1RF_part = np.zeros(count[rank]*K, dtype=float)
Qj2RF_part = np.zeros(count[rank]*K, dtype=float)
Qr1R_part = np.zeros(count[rank]*K, dtype=float)
Qr2R_part = np.zeros(count[rank]*K, dtype=float)
Qr1RF_part = np.zeros(count[rank]*K, dtype=float)
Qr2RF_part = np.zeros(count[rank]*K, dtype=float)

muL_part = np.zeros(count[rank]*K, dtype=float)
muR_part = np.zeros(count[rank]*K, dtype=float)
deltaL1_part = np.zeros(count[rank]*K, dtype=float)
TL_part = np.zeros(count[rank]*K, dtype=float)
mul_part = np.zeros(count[rank]*K, dtype=float)
mur_part = np.zeros(count[rank]*K, dtype=float)


arg1, arg2 = count * K * 2,  displ * K * 2
comm.Scatterv([np.tile(muL,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], muL_part, root=0)
comm.Scatterv([np.tile(muR,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], muR_part, root=0) 
comm.Scatterv([np.tile(deltaL1,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], deltaL1_part, root=0) 
comm.Scatterv([np.tile(TL,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], TL_part, root=0)
comm.Scatterv([np.tile(mul,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], mul_part, root=0)
comm.Scatterv([np.tile(mur,L) if rank == 0 else None, arg1, arg2, MPI.FLOAT], mur_part, root=0) 



for i in range(int(K*count[rank])): 
   QqpR_part[i] = integrate.quad(dQqp,-2,2,args=(GammaL, GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
   QintR_part[i] = integrate.quad(dQint,-2,2,args=(GammaL, GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
   Qj1R_part[i] = integrate.quad(dQj1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
   Qj2R_part[i] = integrate.quad(dQj2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
   QqpRF_part[i] = integrate.quad(dQqp,-2,2,args=(GammaL, GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
   QintRF_part[i] = integrate.quad(dQint,-2,2,args=(GammaL, GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=1.49e-5,epsrel=1.49e-10)[0]
   Qj1RF_part[i] = integrate.quad(dQj1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
   Qj2RF_part[i] = integrate.quad(dQj2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]  
   Qr1R_part[i] = integrate.quad(dQr1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
   Qr2R_part[i] = integrate.quad(dQr2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], muL_part[i], TR, muR_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
   Qr1RF_part[i] = integrate.quad(dQr1,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]
   Qr2RF_part[i] = integrate.quad(dQr2,-1.05,1.05,args=(100*GammaL, 100*GammaR, deltaL_part[i], deltaR, TL_part[i], mul_part[i], TR, mur_part[i], deltac), points = [deltaL_part[i]/deltac, deltaR/deltac], epsabs=.2e-3, epsrel=1.4e-12)[0]  
 
comm.Barrier()
arg = displ * K * 2 if rank == 0 else K * 2 # * 2
comm.Gatherv(QqpR_part, [QqpR, count * K * 2, arg, MPI.FLOAT], root=0)      
comm.Gatherv(QintR_part, [QintR, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj1R_part, [Qj1R, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj2R_part, [Qj2R, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(QqpRF_part, [QqpRF, count * K * 2, arg, MPI.FLOAT], root=0)      
comm.Gatherv(QintRF_part, [QintRF, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj1RF_part, [Qj1RF, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qj2RF_part, [Qj2RF, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr1R_part, [Qr1R, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr2R_part, [Qr2R, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr1RF_part, [Qr1RF, count * K * 2, arg, MPI.FLOAT], root=0)
comm.Gatherv(Qr2RF_part, [Qr2RF, count * K * 2, arg, MPI.FLOAT], root=0)

if rank == 0:
   QjR = 0.5*(Qj1R + Qj2R)
   QjRF = 0.5*(Qj1RF + Qj2RF)
   QrR = 0.5*(Qr1R + Qr2R)
   QrRF = 0.5*(Qr1RF + QrR2F)
   Qbw1 = e*V0/(hbar*Omega0)*(+(QqpRF-QqpR)*np.cos(-Omega0*t) + (QintRF-QintR)*np.cos(phi-Omega0*t) + (QjRF-QjR)*np.sin(phi-Omega0*t) +(QrRF-QrR)*np.sin(-Omega0*t))
   Qbw2 = -e*V0/(hbar*Omega0)*(+(QqpRF-QqpR)*np.cos(Omega0*t) + (QintRF-QintR)*np.cos(phi+Omega0*t) + (QjRF-QjR)*np.sin(phi+Omega0*t) +(QrRF-QrR)*np.sin(Omega0*t))
   QbwT = Qbw1+Qbw2
   data = QbwT
   datafile_path = "QnR.txt"
   np.savetxt(datafile_path, data)
   dtqp, dtj, dti, dtr = QqpR, QjR, QintR, QrR 
   dtqp_path, dtj_path, dti_path, dtr_path = "Qrqp.txt","Qrj.txt","Qrint.txt","Qrr.txt"
   np.savetxt(datafile_path , data)
   np.savetxt(dtqp_path , dtqp)
   np.savetxt(dtj_path , dtj)
   np.savetxt(dti_path , dti)
   np.savetxt(dtr_path , dtr)
   dfqp, dfj, dfi, dfr = QqpF, QjF, QintF, QrF 
   dfqp_path, dfj_path, dfi_path, dfr_path = "Qfrqp.txt","Qfrj.txt","Qfrint.txt","Qfrr.txt"
   np.savetxt(dfqp_path , dfqp)
   np.savetxt(dfj_path , dfj)
   np.savetxt(dfi_path , dfi)
   np.savetxt(dfr_path , dfr)

# 
comm.Barrier()
t_diff = MPI.Wtime() - t_start
if comm.rank==0: 
  print("Execution time", t_diff)
