import numpy as np
import os
from scipy.io import loadmat
import importlib
import openplot
importlib.reload(openplot)
from openplot import *
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def matlab_cnr(directory,tv,Emin):
    cnrm=np.zeros(len(os.listdir(directory)))
    rmsm=np.zeros(len(os.listdir(directory)))
    #noise_level=np.zeros(len(os.listdir(directory)))
    for j,file in enumerate(sorted_alphanumeric(os.listdir(directory))):
    #for j,file in enumerate(os.listdir(directory)):
        filename=directory+file
        #print('First 4 letters {}'.format(file[14:17]))   
        #if noise_level.endswith('t'): noise_level.replace('t','')
        #print(file)
        Data=loadmat(filename)
        Emat=Data['theta']
        sol=Emat[:,-1]*Emin/50#*1e-3*2
        Etrue=np.ravel(Data['Esim'])*Emin/50#*1e-3*2
        #N=len(sol)
        if tv==1:thresh=np.max(Etrue)*0.8
        else:thresh=np.max(Etrue)*0.8#/2
       
        idxe=np.where(Etrue>=thresh)
        idxb=np.where(Etrue<thresh)        
        EE=sol[idxe[0]]
        BB=sol[idxb[0]]
        cnrm[j]=10*np.log10(2*(np.mean(EE)-np.mean(BB))**2/(np.var(EE)+np.var(BB)))
        rmsm[j]=np.sqrt(np.mean(np.abs(2*(sol-Etrue)/(sol+Etrue))**2))
        #noise_level[j]=file[14:16]

    #noise_level[0:-1]=noise_level[0:-1]/10
    #noise_level[0]=noise_level[0]/10
        #if j==0: noise_level[j]=noise_level[j]/10
    #print(noise_level)
    
    return  cnrm,rmsm#,noise_level