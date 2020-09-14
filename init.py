from scipy.interpolate import griddata
import numpy as np
import random
from timeit import default_timer as timer
from numba import jit
#import IP
import matplotlib.pyplot as plt
def im2bw(Ig,level):
    S=np.copy(Ig)
    S[Ig > level] = 0
    S[Ig <= level] = 1
    return(S)

def init(um, disp1, E, vertices, ymeas_noise_coef,Randinit,MA,scale):
    start = timer() 
    N=len(E)
    np.random.seed(1)
    varibx0=ymeas_noise_coef*2#*1e-2
    stdx0=varibx0*np.abs(E)
    umx=um[0::2]
    umy=um[1::2]
    x = np.array(vertices[:,0])
    y =  np.array(vertices[:,1])   
    x_new = np.linspace(x.min(),x.max(),N)
    y_new = np.linspace(y.min(),y.max(),N)
    X, Y = np.meshgrid(x_new, y_new)
    umx_interp=np.around(griddata((x,y),np.ravel(umx),(X,Y),method='linear'),decimals=10)
    umy_interp=np.around(griddata((x,y),np.ravel(umy),(X,Y),method='linear'),decimals=10)
    I1=im2bw(np.abs(umy_interp),0.5)#,0.5
    xx, yy = np.indices((I1.shape[0], I1.shape[1]))
    x1=int(I1.shape[0]/2)
    y1=int(I1.shape[1]/2)
    dia=np.where(I1[x1,y1-150:y1+150]==0)#dia=np.where(I1[x1,:]==0)
    r1=(np.max(dia[0])-np.min(dia[0]))/2#(np.max(dia[0])-np.min(dia[0])+20)/2
    mask_circle1 = (xx - x1) ** 2 + (yy - y1) ** 2 < r1 ** 2
    I3=mask_circle1*1
    plt.imshow(I3)
    mask=(1-I1)*I3
    x1=scale*mask+0.06*np.ones(N)+np.random.normal(0,stdx0,N)#++np.ravel(w)
    x00=np.zeros(len(E))
    for j in range(len(E)):
        idx=vertices[j,:]/30e-3*N
        x00[j]=x1[int(np.floor(idx[0]))-1,int(np.floor(idx[1]))-1]
    #x00=np.ravel(E)-np.ones(N)*0.05
    #w2=np.zeros(N)
    #w2=(E>0.2)*0.5
    #x1=0.1*np.ones(N)+np.ravel(w2)+np.random.normal(0,stdx0,N)#+np.ravel(w)
    #x1 = x1 + np.random.normal(0,(varibx0*(max(x1)-min(x1))),N)
    #plt.imshow(x1)
    SNRx0=10*np.log10(np.linalg.norm(x1)**2/np.sum(stdx0**2))
    print('SNRx0:',SNRx0)
    print("--- %s seconds ---" % (timer()-start)) 
    return x00


