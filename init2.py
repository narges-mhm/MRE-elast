import numpy as np
import importlib
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import vect2im;importlib.reload(vect2im);from vect2im import *
def im2bw(Ig,level):
    S=np.copy(Ig)
    S[Ig > level] = 1
    S[Ig <= level] = 0
    return(S)

def init2(E,um,vertices,ymeas_noise_coef,scale,back_init,dispplot):
    start=timer()
    initE=0#initialization according to noisy E
    N=len(E)
    #umx=um[0::2]/np.linalg.norm(um[0::2])
    np.random.seed(1)
    varibx0=ymeas_noise_coef*2#*1e-2
    stdx0=varibx0*np.abs(E)
    E_interp=vect2im(np.ravel(E)+np.random.normal(0,stdx0,N),vertices) 
    #umx_interp=vect2im(np.ravel(umx)+np.random.normal(0,stdx0,N),vertices) 
    plt.figure(figsize=(10,5))
    plt.subplot(121);plt.imshow(np.abs(E_interp));plt.colorbar();plt.title('mask')#,cmap='gray'
    if initE==1:
        x1=E_interp-back_init*np.ones(N)
    else:
        I1=im2bw(np.abs(E_interp),0.22)#np.random.normal(0,stdx0,N)0.22
        x1=E_interp-back_init*np.ones(N)+scale*I1#+np.ravel(w)
    x00=np.zeros(len(E))
    for j in range(len(E)):
        idx=vertices[j,:]/30e-3*N
        x00[j]=x1[int(np.floor(idx[1]))-1,int(np.floor(idx[0]))-1]
    #SNRx0=10*np.log10(np.linalg.norm(x1)**2/np.sum(stdx0**2))
    #print('SNRx0:',SNRx0)
    if dispplot==1:#
        x00_interp=vect2im(np.ravel(x00),vertices) #plt.figure(figsize=(8,4))
        #plt.subplot(122);plt.imshow(np.abs(x00_interp-E_interp));plt.colorbar();plt.title('residual')#,cmap='gray'
        plt.subplot(122);plt.imshow(np.abs(x00_interp));plt.colorbar();plt.title('initial elasticity modulus')#,cmap='gray'
    print("--- %s seconds ---" % (timer()-start)) 
    return x00