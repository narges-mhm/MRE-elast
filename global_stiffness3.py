import numpy as np
#from numba import jit
# from mpi4py import MPI
# from mpi4py.MPI import ANY_SOURCE
#@jit(nopython=True,parallel=True)
#@jit
def global_stiffness(Ae, Be, E, v, ele_connect, n_element, n_node):
    Tens = np.zeros((2 * n_node, 2 * n_node, n_node)).astype('float32')
    KT = np.zeros((2 * n_node, 2 * n_node))
    inn = np.copy(ele_connect)
    for i in range(n_element): # loop through each element
        # First calculate the element area
        # 3rd calculate the local stiffness matrix 6 by 6
        c11 =  (1-v)/((1+v)*(1-2*v))
        c12 =  v/((1+v)*(1-2*v))
        c66 = 1 / (2*(1+v))
        C = np.array([[c11, c12, 0], [c12, c11, 0], [0, 0, c66]])
        rho=1000
        w=2*np.pi*90
        d=rho*w**2
        kt=-1*Ae[i]/12 *np.array([[2*d,0,d,0,d,0],[0,2*d,0,d,0,d],[d,0,2*d,0,d,0],[0,d,0,2*d,0,d],[d,0,d,0,2*d,0],[0,d,0,d,0,2*d]])       
        Ke = Ae[i]/3 * np.dot(np.dot((np.transpose(Be[:, :, i])), C), Be[:, :, i])

        # 4th list global nodal index
        a = inn[i, 0] # node 1
        b = inn[i, 1] # node 2
        c = inn[i, 2] # node 3
        
        Tens[2*a, 2*a,[a,b,c]] += Ke[0, 0]
        Tens[2*a, 2*a+1,[a,b,c]] += Ke[0, 1]
        Tens[2*a, 2*b,[a,b,c]] += Ke[0, 2]
        Tens[2*a, 2*b+1,[a,b,c]] += Ke[0, 3]
        Tens[2*a, 2*c,[a,b,c]] += Ke[0, 4]
        Tens[2*a, 2*c+1,[a,b,c]] += Ke[0, 5]

        KT[2*a, 2*a] += kt[0, 0]
        KT[2*a, 2*a+1] += kt[0, 1]
        KT[2*a, 2*b] += kt[0, 2]
        KT[2*a, 2*b+1] += kt[0, 3]
        KT[2*a, 2*c] += kt[0, 4]
        KT[2*a, 2*c+1] += kt[0, 5]

        Tens[2*a+1, 2*a,[a,b,c]] += Ke[1, 0]
        Tens[2*a+1, 2*a+1,[a,b,c]] += Ke[1, 1]
        Tens[2*a+1, 2*b,[a,b,c]] += Ke[1, 2]
        Tens[2*a+1, 2*b+1,[a,b,c]] += Ke[1, 3]
        Tens[2*a+1, 2*c,[a,b,c]] += Ke[1, 4]
        Tens[2*a+1, 2*c+1,[a,b,c]] += Ke[1, 5]

        KT[2*a+1, 2*a] += kt[1, 0]
        KT[2*a+1, 2*a+1] += kt[1, 1]
        KT[2*a+1, 2*b] += kt[1, 2]
        KT[2*a+1, 2*b+1] += kt[1, 3]
        KT[2*a+1, 2*c] += kt[1, 4]
        KT[2*a+1, 2*c+1] += kt[1, 5]

        Tens[2*b, 2*a,[a,b,c]] += Ke[2, 0]
        Tens[2*b, 2*a+1,[a,b,c]] += Ke[2, 1]
        Tens[2*b, 2*b,[a,b,c]] += Ke[2, 2]
        Tens[2*b, 2*b+1,[a,b,c]] += Ke[2, 3]
        Tens[2*b, 2*c,[a,b,c]] += Ke[2, 4]
        Tens[2*b, 2*c+1,[a,b,c]] += Ke[2, 5]

        KT[2*b, 2*a] += kt[2, 0]
        KT[2*b, 2*a+1] += kt[2, 1]
        KT[2*b, 2*b] += kt[2, 2]
        KT[2*b, 2*b+1] += kt[2, 3]
        KT[2*b, 2*c] += kt[2, 4]
        KT[2*b, 2*c+1] += kt[2, 5]

        Tens[2*b+1, 2*a,[a,b,c]] += Ke[3, 0,]
        Tens[2*b+1, 2*a+1,[a,b,c]] += Ke[3, 1]
        Tens[2*b+1, 2*b,[a,b,c]] += Ke[3, 2]
        Tens[2*b+1, 2*b+1,[a,b,c]] += Ke[3, 3]
        Tens[2*b+1, 2*c,[a,b,c]] += Ke[3, 4]
        Tens[2*b+1, 2*c+1,[a,b,c]] += Ke[3, 5]

        KT[2*b+1, 2*a] += kt[3, 0]
        KT[2*b+1, 2*a+1] += kt[3, 1]
        KT[2*b+1, 2*b] += kt[3, 2]
        KT[2*b+1, 2*b+1] += kt[3, 3]
        KT[2*b+1, 2*c] += kt[3, 4]
        KT[2*b+1, 2*c+1] += kt[3, 5]

        Tens[2*c, 2*a,[a,b,c]] += Ke[4, 0]
        Tens[2*c, 2*a+ 1,[a,b,c]] += Ke[4, 1]
        Tens[2*c, 2*b,[a,b,c]] += Ke[4, 2]
        Tens[2*c, 2*b+1,[a,b,c]] += Ke[4, 3]
        Tens[2*c, 2*c,[a,b,c]] += Ke[4, 4]
        Tens[2*c, 2*c+1,[a,b,c]] += Ke[4, 5]

        KT[2*c, 2*a] += kt[4, 0]
        KT[2*c, 2*a+1] += kt[4, 1]
        KT[2*c, 2*b] += kt[4, 2]
        KT[2*c, 2*b+1] += kt[4, 3]
        KT[2*c, 2*c] += kt[4, 4]
        KT[2*c, 2*c+1] += kt[4, 5]

        Tens[2*c+1, 2*a,[a,b,c]] += Ke[5, 0]
        Tens[2*c+1, 2*a+1,[a,b,c]] += Ke[5, 1]
        Tens[2*c+1, 2*b,[a,b,c]] += Ke[5, 2]
        Tens[2*c+1, 2*b+1,[a,b,c]] += Ke[5, 3]
        Tens[2*c+1, 2*c,[a,b,c]] += Ke[5, 4]
        Tens[2*c+1, 2*c+1,[a,b,c]] += Ke[5, 5]

        KT[2*c+1, 2*a] += kt[5, 0]
        KT[2*c+1, 2*a+1] += kt[5, 1]
        KT[2*c+1, 2*b] += kt[5, 2]
        KT[2*c+1, 2*b+1] += kt[5, 3]
        KT[2*c+1, 2*c] += kt[5, 4]
        KT[2*c+1, 2*c+1] += kt[5, 5]

    GK=np.squeeze(Tens@E)/3+KT

    #D=np.transpose(Tens, (2, 0, 1)).dot(uxy)
    return GK,Tens,KT#,D
