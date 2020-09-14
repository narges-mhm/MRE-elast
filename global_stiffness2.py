import numpy as np
from numba import jit
# from mpi4py import MPI
# from mpi4py.MPI import ANY_SOURCE
@jit(nopython=True, parallel=True)
def global_stiffness2(Ae, Be, E, v, ele_connect, n_element, n_node):
    GK = np.zeros((2 * n_node, 2 * n_node))
    inn = np.copy(ele_connect)
    for i in range(n_element): # loop through each element
        # First calculate the element area
        # 3rd calculate the local stiffness matrix 6 by 6
        c11 = np.mean(E[ele_connect[i, :]]) * (1-v)/((1+v)*(1-2*v))
        c12 = np.mean(E[ele_connect[i, :]]) * v/((1+v)*(1-2*v))
        c66 = np.mean(E[ele_connect[i, :]]) / (2*(1+v))
        C = np.array([[c11, c12, 0], [c12, c11, 0], [0, 0, c66]])
        Ke = Ae[i] * np.dot(np.dot((np.transpose(Be[:, :, i])), C), Be[:, :, i])

        # 4th list global nodal index
        a = inn[i, 0] # node 1
        b = inn[i, 1] # node 2
        c = inn[i, 2] # node 3

        GK[2*a, 2*a] += Ke[0, 0]
        GK[2*a, 2*a+1] += Ke[0, 1]
        GK[2*a, 2*b] += Ke[0, 2]
        GK[2*a, 2*b+1] += Ke[0, 3]
        GK[2*a, 2*c] += Ke[0, 4]
        GK[2*a, 2*c+1] += Ke[0, 5]

        GK[2*a+1, 2*a] += Ke[1, 0]
        GK[2*a+1, 2*a+1] += Ke[1, 1]
        GK[2*a+1, 2*b] += Ke[1, 2]
        GK[2*a+1, 2*b+1] += Ke[1, 3]
        GK[2*a+1, 2*c] += Ke[1, 4]
        GK[2*a+1, 2*c+1] += Ke[1, 5]

        GK[2*b, 2*a] += Ke[2, 0]
        GK[2*b, 2*a+1] += Ke[2, 1]
        GK[2*b, 2*b] += Ke[2, 2]
        GK[2*b, 2*b+1] += Ke[2, 3]
        GK[2*b, 2*c] += Ke[2, 4]
        GK[2*b, 2*c+1] += Ke[2, 5]

        GK[2*b+1, 2*a] += Ke[3, 0,]
        GK[2*b+1, 2*a+1] += Ke[3, 1]
        GK[2*b+1, 2*b] += Ke[3, 2]
        GK[2*b+1, 2*b+1] += Ke[3, 3]
        GK[2*b+1, 2*c] += Ke[3, 4]
        GK[2*b+1, 2*c+1] += Ke[3, 5]

        GK[2*c, 2*a] += Ke[4, 0]
        GK[2*c, 2*a+ 1] += Ke[4, 1]
        GK[2*c, 2*b] += Ke[4, 2]
        GK[2*c, 2*b+1] += Ke[4, 3]
        GK[2*c, 2*c] += Ke[4, 4]
        GK[2*c, 2*c+1] += Ke[4, 5]

        GK[2*c+1, 2*a] += Ke[5, 0]
        GK[2*c+1, 2*a+1] += Ke[5, 1]
        GK[2*c+1, 2*b] += Ke[5, 2]
        GK[2*c+1, 2*b+1] += Ke[5, 3]
        GK[2*c+1, 2*c] += Ke[5, 4]
        GK[2*c+1, 2*c+1] += Ke[5, 5]
    return GK
