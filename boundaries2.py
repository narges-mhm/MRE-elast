from numpy import *

#from numba import jit
#from numba import size
#@jit(nopython=True, parallel=True)
def boundaries2(flag,inn,vertices,GK,f):
    #bcflag=2
    height = 30e-3 #mm
    tb=height
    width = 30e-3 #mm
    bcy=-0.2
    verindex = arange(0, shape(vertices)[0])
    nelement = shape(inn)[0]
    t1=0
    t2=-f
    r=0
    mesh = zeros((nelement, 3, 4))
    for i in range(nelement):  # loop through each element
        for j in range(3):  # loop through each local nodes of each element
            # Insert the global nodal index
            mesh[i, j, 0] = inn[i, j]
            # Insert the (x,y) coordinates of the global/local node
            mesh[i, j, 1:3] = vertices[inn[i, j], :]
           
    w_centre = width/2
    n = size(verindex) # number of nodes
    fxy = zeros(2 * n)  # Global residual force vector

    inn2 = zeros((shape(inn)[0],shape(inn)[1]+3)) # element connectivity 2, nelement by 6, top boundary
    inn2[:,:-3] = inn.astype(int)  # first 3 columns of inn2 = inn
    for i in range(shape(inn)[0]): # loop through each element
        for j in range(3): # loop through the 3 nodes of each element i
            if mesh[i,j,2] == 0:  # y of global nodal j in element i is on the specified bottom boundary.
               fxy[(inn[i, j]) * 2 + 1] = 0  # set the corresponding force vector to zero
               GK[(inn[i, j]) * 2 + 1,:] = 0
               GK[(inn[i, j]) * 2 + 1,(inn[i, j]) * 2 + 1] = 1
               if mesh[i,j,1] == (w_centre):  # station leftest node. or the middle node
                  fxy[(inn[i,j])*2] = 0
                  GK[(inn[i, j]) * 2,:] = 0
                  GK[(inn[i, j]) * 2,(inn[i, j]) * 2] = 1
            if mesh[i,j,2] == tb: # y of global nodal j in element i is on specified top boundary.
               inn2[i,j+3] = 1
    if flag == 1:  # Dirichlet BC
        print("Boundary conditions: top boundary Dirichlet BC...")
        
        # Assignment states that the top is distorted by 20%
        for i in range(shape(inn)[0]): # loop through each element
            for j in range(3): # loop through the 3 nodes of each element i
                if mesh[i,j,2] == tb: # y of global nodal j in element i is on specified top boundary.
                   GK[(inn[i,j])*2 + 1,:] = 0
                   GK[(inn[i, j]) * 2 + 1,(inn[i, j]) * 2 + 1] = 1
                   fxy[(inn[i,j])*2 + 1] = tb * bcy

    if flag == 2:  # Neumann BC
        #print("Boundary conditions: top boundary Neumann BC...")
        for i in range(shape(inn)[0]):     # loop through each element
            if sum(inn2[i,3:6]) == 2:       # if two of the nodes fall on the boundary, force is applied to that face.
               lindex = inn2[i,where(inn2[i,3:6] == 1)].astype(int)
               lindex = lindex.ravel()
               l = abs(vertices[lindex[0],0]-vertices[lindex[1],0])
               # if the force is applied in the y direction, t1 = 0. vice versa for t2.
               r1 = t1 * l/2
               r2 = t2 * l/2
               r3 = t1 * l/2
               r4 = t2 * l/2
               a = 2*lindex[0]
               b = 2*lindex[0] + 1
               c = 2*lindex[1]
               d = 2*lindex[1] + 1
               fxy[a] += r1
               fxy[b] += r2
               fxy[c] += r3
               fxy[d] += r4
    return inn2,fxy,GK

