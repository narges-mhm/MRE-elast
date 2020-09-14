import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def addnoise(disp1,ymeas_noise_coef,vertices,dispplot):
    dispx=disp1[0::2]#-np.mean(disp1[0::2])
    dispy=disp1[1::2]#-np.mean(disp1[1::2])
    N=len(dispx)
    M=2*N
    n_iter=1
    np.random.seed(1)
    um=np.zeros(2*N)
    um_mat=np.zeros((M,n_iter))
    Noisemat=np.zeros((M,n_iter))

    xmeas_noise_coef=1.7*ymeas_noise_coef#e-3# the power is 1.7**2=3 times the power in y irection
    stdx=xmeas_noise_coef*np.abs(dispx)
    umx = dispx + np.random.normal(0,stdx,N)
    umx = umx + np.random.normal(0,(xmeas_noise_coef*(max(umx)-min(umx))),N)
    #ymeas_noise_coef=1e-2
    stdy=ymeas_noise_coef*np.abs(dispy)
    umy = dispy + np.random.normal(0,stdy,N)
    umy = umy + np.random.normal(0,(ymeas_noise_coef*(max(umy)-min(umy))),N)
    um[0::2]=umx
    um[1::2]=umy
    SNR=10*np.log10(np.linalg.norm(um)**2/(np.linalg.norm(um-disp1)**2))#np.linalg.norm(nx)**2+np.linalg.norm(ny)**2
    SNR=np.around(SNR,decimals=1)
    #SNR2=10*np.log10(np.sum(um ** 2)/np.sum((um-disp1) ** 2))
    Nxlevel=np.linalg.norm(umx-dispx)/np.linalg.norm(umx)
    Nylevel=np.linalg.norm(umy-dispy)/np.linalg.norm(umy)
    Ntlevel=np.linalg.norm(um-disp1)/np.linalg.norm(um)
    Ntlevel=np.around(Ntlevel*1e2,decimals=1)
    print('SNR:',SNR)
    print('Nx:',Nxlevel)
    print('Ny:',Nylevel)
    print('Nt:',Ntlevel)
    if dispplot==1:
        plt.figure(figsize=(10,5))
        x = np.array(vertices[:,0])
        y =  np.array(vertices[:,1])   
        x_new = np.linspace(x.min(),x.max(),N)
        y_new = np.linspace(y.min(),y.max(),N)
        X, Y = np.meshgrid(x_new, y_new)
        umx_interp=np.around(griddata((x,y),np.ravel(umx),(X,Y),method='linear'),decimals=10)
        umy_interp=np.around(griddata((x,y),np.ravel(umy),(X,Y),method='linear'),decimals=10)
        plt.subplot(121);plt.imshow(np.abs(umx_interp),cmap='gray');plt.axis('off');plt.colorbar();plt.title('Lateral synthetic measured displacements')#,vmin=-25,vmax=15
        plt.subplot(122);plt.imshow(np.abs(umy_interp),cmap='gray');plt.axis('off');plt.colorbar();plt.title('Axial synthetic measured displacements')#,vmin=-25,vmax=15
    return  um, Ntlevel,SNR