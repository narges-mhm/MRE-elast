
from pyunlocbox import functions, solvers
import imp
import Ggradient
imp.reload(Ggradient)
from Ggradient import *
import numpy as np
import Nodeneighbor
imp.reload(Nodeneighbor)
from Nodeneighbor import *
import matplotlib.pyplot as plt
import random
def optGFB(D,GK0,triangles,ft,E,maxit,x1,step,tau1,tau2,tau3,flag):
####
    N=len(E)
    w=np.ones((2*N,1))
    w[:102*2,:]=0.99
##
    f1= functions.norm_l2(y=np.ravel(ft), A=D, lambda_=tau1, tight=False,w=np.ravel(w),dim=1)#['EVAL', 'GRAD', 'PROX']
####
    f2 = functions.func()
    f2._eval = lambda x: 0
    f2._prox = lambda x,T:np.clip(x, 0,0.7)# np.inf)#pprox#['EVAL', 'GRAD', 'PROX']
###
    T1=Nodeneighbor(triangles,N)
    g = lambda x: Ggradient(x,T1)
    f3 = functions.norm_l1(A=g,At=None, dim=1, y=np.zeros(N),lambda_=tau2)#['EVAL', 'PROX']
######
    #x0=np.ravel(E)-np.ones(N)*0.1
    yy2=np.ravel(ft)#-D@np.ones(N)*0.1#+nf*scale2
###
    f7 = functions.norm_l1(lambda_=tau3,y=np.zeros(N),dim=1)#,tight=False)#if remove A it works
####    
    N_cov=np.zeros((2*N,2*N))
    for j in np.arange(0,2*N,2):
        N_cov[j,j]=3
        N_cov[j+1,j+1]=1
    gamma=GK0@N_cov@GK0.T#varf*np.ones((2*N,2*N))+
    gamma+= np.eye(gamma.shape[1])*1e-5
    gamma_inv=np.linalg.inv(gamma)
    f8=functions.func(lambda_=tau1)
    f8._eval = lambda x: 0.5*(yy2-D@x).T@gamma_inv@(yy2-D@x)*1e-4#/np.linalg.norm(gamma_inv@(yy2-D@x))#*1e-5#*1e-6#
    f8._grad = lambda x: -1*D.T@gamma_inv@(yy2-D@x)*1e-4#/(np.linalg.norm(gamma_inv@(yy2-D@x)))#*1e-5#*1e-6#-
#######
    #step = 0.08#0.5 /tau1# (np.linalg.norm(func(x0),2)**2/np.linalg.norm(x0,2)**2) #0.5/tau#2e3/scale
    solver2 = solvers.generalized_forward_backward(step=step)#generalized_forward_backward(step=step*0.1)#step*0.1)douglas_rachford
    #solver2 = solvers.forward_backward(step=step*0.1)#step*0.005
    # without f3 -->singular matrix
    #ret2 = solvers.solve([f8,f3], x1, solver2, rtol=1e-15, maxit=maxit)#f8,f2,f3,f7
    ret2 = solvers.solve([f8,f3,f2], np.copy(x1), solver2, rtol=1e-15, maxit=maxit)#f8,f2,f3,f7
    objective = np.array(ret2['objective'])
    sol=ret2['sol']
    
    import matplotlib.pyplot as plt
 
    #print('True f7',f7.eval(x0))  
    #print('Est f7',f7.eval(ret2['sol'])) 
    if flag==1:
        _ = plt.figure(figsize=(10,4))
        _ = plt.subplot(121)
        _ = plt.plot(E, 'o', label='Original E')#-np.ones(N)*0.1
        _ = plt.plot(ret2['sol'], 'xr', label='Reconstructed E')
        _ = plt.plot(x1,'*g')
        _ = plt.grid(True)
        _ = plt.title('Achieved reconstruction')
        _ = plt.legend(numpoints=1)
        _ = plt.xlabel('Signal dimension number')
        _ = plt.ylabel('Signal value')

        _ = plt.subplot(122)
    #_ = plt.semilogy(objective[:, 3], '-',label='sparsity')
    #_ = plt.semilogy(objective[:, 2], '--',label='tv-norm')
    #_ = plt.semilogy(objective[:, 1], '--',label='positivity constraint')#l2-norm
        _ = plt.semilogy(objective[:, 0], '-.',label='l2-norm')#
        _ = plt.semilogy(np.sum(objective, axis=1), label='Global objective')
        _ = plt.grid(True)
        _ = plt.title('Convergence')
        _ = plt.legend(numpoints=1)
        _ = plt.xlabel('Iteration number')
        _ = plt.ylabel('Objective function value')
        _ =plt.show()
    return sol
