import numpy as np
def Meshneighbor(triangles):
    
    Ne=triangles.shape[0]
    P=[0]*Ne
    for k in range(Ne):
        nei=[]
        for j in range(Ne):
            if (all(x in triangles[j,:] for x in [triangles[k,0], triangles[k,1]]) or all(x in triangles[j,:] for x in [triangles[k,0], triangles[k,2]])) or all(x in triangles[j,:] for x in [triangles[k,1], triangles[k,2]]):
                nei.extend([j])
        P[k]=nei
        P[k]=list(dict.fromkeys(P[k]))
    P1=np.zeros((Ne,10))
    for i in range(Ne):
        for j in range(len(P[i])):
            P1[i,j]=int(P[i][j])+0   
    return P1