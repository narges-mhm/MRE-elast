
import importlib
import scipy
from calc_Ae import *
from genBe import *
import global_stiffness3;importlib.reload(global_stiffness3);from global_stiffness3 import *
from global_stiffness3 import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from Meshneighbor import *
from Ggradient import *
from Nodeneighbor import *
from boundaries2 import *
from boundariesTens import *
from timeit import default_timer as timer
from scipy.io import loadmat
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.io as sio
from pyunlocbox import functions, solvers
from scipy.fftpack import dct
import imp
import random
import numpy as np
import openfile;importlib.reload(openfile);from openfile import *
import addnoise;importlib.reload(addnoise);from addnoise import *
import init;importlib.reload(init);from init import *
import proxsolve;importlib.reload(proxsolve);from proxsolve import *
import openplot;importlib.reload(openplot);from openplot import *
import calc_cnr;importlib.reload(calc_cnr);from calc_cnr import *
import matlab_cnr;importlib.reload(matlab_cnr);from matlab_cnr import *;
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


### MRE estimation for any inclusion image
## E-clean6330:disp image
#freq=200Hz
import openfile_general;importlib.reload(openfile_general);from openfile_general import *
import addnoise;importlib.reload(addnoise);from addnoise import *
from global_stiffness3 import *
directory0="/home/nmohamm4/Documents/MRE_res1/"
filename=directory0+'E_clean'+str(630)+'.png'
ymeas_noise_coef=1e-3;dispplot=0
E, disp1, Tens, KT, matTens, triangles, vertices, fxy=openfile_general(filename)
um, Ntlevel,SNR=addnoise(disp1,ymeas_noise_coef,vertices,dispplot)

#filename='10Nd_E5'
#E, disp1, Tens, KT, matTens, triangles, vertices, fxy=openfile(filename)
#um, Ntlevel,SNR=addnoise(disp1,ymeas_noise_coef,vertices,dispplot)#pnoiselevel=Ntlevel#psnr=SNR
N=len(E)

### E_clean631
directory0="/home/nmohamm4/Documents/MRE_res1/"
filename=directory0+'E_clean'+str(631)+'.png'
import openfile_general;importlib.reload(openfile_general);from openfile_general import *
import init2;importlib.reload(init2);from init2 import *
import proxsolve;importlib.reload(proxsolve);from proxsolve import *
Randinit=0;scale=0.4;MA=0;vmin=0;vmax=0.6;dispplot=1;savefig=0;ymeas_noise_coef=1.5e-2#freq=90
Itr1=400#300#400
Itr2=1#20#50#500#
stepT=[0.6*0.1,0.6*0.12,0.6*0.15,0.6*0.17,0.6*0.2]
E, disp1, Tens, KT, matTens, triangles, vertices, fxy=openfile_general(filename)
N=len(E)
um, Ntlevel,SNR=addnoise(disp1,ymeas_noise_coef,vertices,dispplot)#pnoiselevel=Ntlevel#psnr=SNR
back_init=np.random.uniform(0.05,0.15)#0.12
scale=np.random.uniform(E.max()-2*back_init,E.max()-back_init,1)
x1=init2(E,vertices,ymeas_noise_coef,scale,back_init,dispplot)#init
#x1=init(um, disp1, E, vertices, ymeas_noise_coef,Randinit,MA,scale)
A=np.zeros((N,len(stepT)))
for j in range(len(stepT)):
    step=stepT[j]
    solmatf=proxsolve(E, um, matTens, Tens, KT, triangles, vertices, fxy, ymeas_noise_coef, x1, vmin,vmax,filename, Ntlevel, SNR, Itr1, Itr2,savefig,MA,step)#
    A[:,j]=solmatf


### E_noisy630 or E_clean630 for one noise-level
directory0="/home/nmohamm4/Documents/MRE_res1/"
filename=directory0+'E_clean'+str(630)+'.png'#+'E_noisy'+
import openfile_general;importlib.reload(openfile_general);from openfile_general import *
import init2;importlib.reload(init2);from init2 import *
import addnoise;importlib.reload(addnoise);from addnoise import *
import proxsolve;importlib.reload(proxsolve);from proxsolve import *
Randinit=0;MA=0;vmin=0;vmax=0.6;dispplot=1;savefig=0;ymeas_noise_coef=1e-3#freq=90
Itr1=400#300#400
Itr2=1#20#50#500#
stept=[0.2,0.22]
stepT = [i * 0.6*ymeas_noise_coef/1e-3 for i in stept]
E, disp1, Tens, KT, matTens, triangles, vertices, fxy=openfile_general(filename)
N=len(E)
um, Ntlevel,SNR=addnoise(disp1,ymeas_noise_coef,vertices,dispplot)#pnoiselevel=Ntlevel#psnr=SNR

back_init=0.1
scale=0.34
x1=init2(E,um,vertices,ymeas_noise_coef,scale,back_init,dispplot)#init
#x1=init(um, disp1, E, vertices, ymeas_noise_coef,Randinit,MA,scale)
A=np.zeros((N,len(stepT)))
for j in range(len(stepT)):
    step=stepT[j]
    solmatf=proxsolve(E, um, matTens, Tens, KT, triangles, vertices, fxy, ymeas_noise_coef, x1, vmin,vmax,filename, Ntlevel, SNR, Itr1, Itr2,savefig,MA,step)#
    A[:,j]=solmatf

### displaying mean of E for one noise-level

vmin=0.1;vmax=0.5
plt.figure(figsize=(4,4))
x = np.array(vertices[:,0])
y =  np.array(vertices[:,1])   
x_new = np.linspace(x.min(),x.max(),N)
y_new = np.linspace(y.min(),y.max(),N)
X, Y = np.meshgrid(x_new, y_new)
E_interp=np.around(griddata((x,y),np.ravel(np.mean(A,axis=1)),(X,Y),method='linear'),decimals=10)#+0.1*np.ones(N)
E_true=np.around(griddata((x,y),np.ravel(E),(X,Y),method='linear'),decimals=10)
plt.imshow(np.abs(E_true),vmin=vmin,vmax=vmax);plt.colorbar();plt.axis('off')#
#solmatf[0]

### Matlab output image for E_clean630 SNR48
import matlab_cnr;importlib.reload(matlab_cnr);from matlab_cnr import *;
import openplot;importlib.reload(openplot);from openplot import *;
directory = './matlabsim/multiE_tv/'
#directory0 = './matlabsim/multiE_notv/'
cnrm,rmsm=matlab_cnr(directory,1,E.min())#thresh=0.4
#cnrm0,rmsm0=matlab_cnr(directory0,0)##thresh=0.2
print('cnrm',cnrm)
print('rmsm',rmsm)
#print('cnrm0',cnrm0)
#sio.savemat('E5comp.mat', {'cnrp':cnrp,'rmsp':rmsp,'cnrm':cnrm,'rmsm':rmsm,'cnrm0':cnrm0,'rmsm0':rmsm0})
vmin=0.1;vmax=0.5
triname='./matlabsim/tri'
savefig=0
oneE=1
#back_init=0.1433
mnoiselevel=5.4#Alakiiiiiiiiiii-pnoiselevel[0]
for j,file in enumerate(sorted_alphanumeric(os.listdir(directory))):
#for j,file in enumerate(os.listdir(directory)):
    filename=directory+file
    #print(filename)
    #openplot(filename,triname,vmin,vmax, savefig,oneE,E.min(),j)

### with For: different noise levels
import numpy as np
import importlib
import scipy.io as sio
import addnoise;importlib.reload(addnoise);from addnoise import *
import openplot;importlib.reload(openplot);from openplot import *
import openfile_general;importlib.reload(openfile_general);from openfile_general import *
import init2;importlib.reload(init2);from init2 import *
import proxsolve;importlib.reload(proxsolve);from proxsolve import *
directory0="/home/nmohamm4/Documents/MRE_res1/"
filename=directory0+'E_clean'+str(630)+'.png'#+'E_noisy'+
Randinit=0;MA=0;vmin=0;vmax=0.6;dispplot=1;savefig=0;ymeas_noise_coef=1e-3#freq=90
back_init=0.04#0.03
scale=0.02#0.34
Itr1=400#300#400
Itr2=1#20#50#500#
ncoeff=[1e-3,5e-3,1e-2,1.5e-2,2e-2,2.5e-2,3e-2]
#stepT=[0.6*0.1,0.6*0.12,0.6*0.15,0.6*0.17,0.6*0.2]
#stepT=[0.6*0.17,0.6*0.2]
stept=[0.16,0.18]
stepT = [i * 0.6 for i in stept]#*ymeas_noise_coef/1e-3
E, disp1, Tens, KT, matTens, triangles, vertices, fxy=openfile_general(filename)
N=len(E)
solmatf=np.zeros((N,len(ncoeff)))
pnoiselevel=np.zeros(len(ncoeff))
psnr=np.zeros(len(ncoeff))
for j, ymeas_noise_coef in enumerate(ncoeff):
    print('j is:',j)
    um, Ntlevel,SNR=addnoise(disp1,ymeas_noise_coef,vertices,dispplot)
    pnoiselevel[j]=Ntlevel
    psnr[j]=SNR
    x1=init2(E,um,vertices,ymeas_noise_coef,scale,back_init,dispplot)#ini
    A=np.zeros((N,len(stepT)))
    for jj in range(len(stepT)):
        step=stepT[jj]
        sol=proxsolve(E, um, matTens, Tens, KT, triangles, vertices, fxy, ymeas_noise_coef, x1, vmin,vmax,filename, Ntlevel, SNR, Itr1, Itr2,savefig,MA,step)#
        A[:,jj]=sol
    solmatf[:,j]=np.ravel(np.mean(A,axis=1))
sio.savemat('E_clean630.mat', {'sol': solmatf,'pnl':  pnoiselevel,'psnr': psnr})

### Python analysis: plotting E with different noise level &calculating cnr/rms
Data=loadmat('E_clean630.mat')
solmat=Data['sol']
pnl=Data['pnl'][0]
psnr=Data['psnr'][0]
x = np.array(vertices[:,0])
y =  np.array(vertices[:,1])   
x_new = np.linspace(x.min(),x.max(),N)
y_new = np.linspace(y.min(),y.max(),N)
X, Y = np.meshgrid(x_new, y_new)
savefig=1
directory0="/home/nmohamm4/Documents/MRE_res1/"
vmin=0.1;vmax=0.5
for j in range(len(pnl)):
    sol=solmat[:,j]
    Ntlevel=pnl[j]
    snr=psnr[j]
    Emax=np.max(E)
    E_interp=np.around(griddata((x,y),np.ravel(sol),(X,Y),method='linear'),decimals=10)#+0.1*ones(len(E))
    fig=plt.figure(figsize=(5,5))
    plt.imshow(np.abs(E_interp),vmin=vmin,vmax=vmax);plt.colorbar();plt.axis('off');plt.show()#;plt.title('Reconstructed Young\' modulus ' + r'$\mathbf{\mu}$'+' using \n noisy measurements ('+str(Ntlevel)+'% noise level)',fontsize=5);plt.colorbar();plt.axis('off');plt.show()
    if savefig: fig.savefig(directory0+'MRE_img/pyt_img/Enoiselevel'+str(Ntlevel)+'snr'+str(snr)+'.png')
        #matplotlib.image.imsave(directory+'pytimg/Enoiselevel'+str(Ntlevel)+'snr'+str(snr)+'.png')
ncoeff=[1e-3,5e-3,1e-2,1.5e-2,2e-2,2.5e-2,3e-2]
#solmat=Data['sol_MA0']+0.1*np.ones((len(E),len(ncoeff)))
pnoiselevel=Data['pnl']
import calc_cnr;importlib.reload(calc_cnr);from calc_cnr import *
oneE=1
cnrp,rmsp=calc_cnr(E,solmat,oneE)
#print('cnrp',cnrp)
#print('rmsp',rmsp)


### reconstruction vs. noise level matlab & performance computation (CNR,RMS)

#filename=['./matlabsim/E5noiselevel34tv20coeff0.0011snr49','./matlabsim/E5noiselevel182tv20coeff0.0058snr35','./matlabsim/E5noiselevel350tv20coeff0.011snr29','./matlabsim/E5noiselevel540tv20coeff0.0175snr25','./matlabsim/E5noiselevel733tv20coeff0.023snr23','./matlabsim/E5noiselevel947tv20coeff0.029snr20']#,'./matlabsim/E5noiselevel1087tv20coeff0.0345snr19'

import matlab_cnr;importlib.reload(matlab_cnr);from matlab_cnr import *;
directory = './matlabsim/multiE_tv/'
directory0 = './matlabsim/multiE_notv/'
Emin=E.min()
cnrm,rmsm=matlab_cnr(directory,1,Emin)#thresh=0.4
cnrm0,rmsm0=matlab_cnr(directory0,0,back_init)##thresh=0.2
print('cnrm',cnrm)
print('rmsm',rmsm)
print('cnrm0',cnrm0)

sio.savemat('E5comp.mat', {'cnrp':cnrp,'rmsp':rmsp,'cnrm':cnrm,'rmsm':rmsm,'cnrm0':cnrm0,'rmsm0':rmsm0})

triname='./matlabsim/tri'
savefig=1
oneE=1
mnoiselevel=pnoiselevel[0]
vmin=0.1;vmax=0.5
#back_init=0.1433
for j,file in enumerate(sorted_alphanumeric(os.listdir(directory))):
#for j,file in enumerate(os.listdir(directory)):
    filename=directory+file
    #print(filename)
    openplot(filename,triname,vmin,vmax, savefig,oneE,E.min(),j)


### Matlab & Python performance comparison

import calc_cnr;importlib.reload(calc_cnr);from calc_cnr import *
from scipy import interpolate
oneE=1
cnrp,rmsp=calc_cnr(E,solmat,oneE)
xnew = np.linspace(mnoiselevel.min(), mnoiselevel.max(), 50) 
a1_BSpline = interpolate.make_interp_spline(mnoiselevel,cnrp)
cnrp_new = a1_BSpline(xnew)
a2_BSpline = interpolate.make_interp_spline(mnoiselevel,cnrm)
cnrm_new = a2_BSpline(xnew)
a3_BSpline = interpolate.make_interp_spline(mnoiselevel,cnrm0)
cnrm0_new = a3_BSpline(xnew)

_ =plt.figure(figsize=(13,3))
_ = plt.subplot(121)
_ = plt.plot(xnew,cnrp_new,'--',label='statistiical_tv')
_ = plt.plot(xnew,cnrm_new,'-.',label='OpenQSEI_tv')
_ = plt.plot(xnew,cnrm0_new,'-',label='OpenQSEI_ws')
_ = plt.xlabel('noise level %');plt.ylabel('CNR(dB)')
_ = plt.legend(numpoints=1,fontsize=8)#;plt.title('Contrat to noise ratio for one inclusion with ' + r'$\mathbf{\mu}$'+'=50 KPa',fontsize=10)
_ = plt.grid(True)

a1_BSpline = interpolate.make_interp_spline(mnoiselevel,rmsp)
rmsp_new = a1_BSpline(xnew)
a2_BSpline = interpolate.make_interp_spline(mnoiselevel,rmsm)
rmsm_new = a2_BSpline(xnew)
a3_BSpline = interpolate.make_interp_spline(mnoiselevel,rmsm0)
rmsm0_new = a3_BSpline(xnew)

_ = plt.subplot(122)
_ = plt.plot(xnew,rmsp_new,'--',label='statistical_tv')
_ = plt.plot(xnew,rmsm_new,'-.',label='OpenQSEI_tv')
_ = plt.plot(xnew,rmsm0_new,'-',label='OpenQSEI_ws')
_ = plt.xlabel('noise level %');plt.ylabel('RMS');plt.legend(numpoints=1,fontsize=8)#;plt.title('RMS error',fontsize=10)
_ =plt.grid('on')
_ =plt.show()            