import numpy as np
def Ggradient(E,A):
    
    N=np.size(E)
    grad1=np.zeros((N,1))#[0]*N
    #print(E1.shape)
    for n in range(N):
        S1=np.zeros((N,1))
        for k in range(10):
            if A[n,k]!=0:                
                S1[n]=S1[n]+(E[n]-E[int(A[n,k])])**2                
        grad1[n]=np.sqrt(S1[n])
        #print('np.max(np.max(grad1))=' ,np.max(np.max(grad1)))
        if np.max(np.max(grad1))!=0:
            result1=grad1/np.max(np.ceil(grad1))
        else:
            result1 =grad1
        #print('np.max(np.max(grad1))=' ,np.max(np.max(grad1)))
    return np.ravel(result1)