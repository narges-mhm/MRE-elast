import numpy as np
def Nodeneighbor(triangles,N):
    T=[0]*N
    for node in range(N):
        neigh_t=[]
        for elem in triangles:        
            if node in elem:       
                neigh_t.extend(elem)
        T[node]=neigh_t
        T[node]=list(dict.fromkeys(T[node]))
    T1=np.zeros((N,10))
    for i in range(N):
        for j in range(len(T[i])):
            T1[i,j]=T[i][j]+0
    return T1