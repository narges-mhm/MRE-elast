from numpy import *

def boundariesTens2(flag,inn,vertices,Tens):
    #bcflag=2
    height = 30e-3 #mm
    tb=height
    width = 30e-3 #mm
    bcy=-0.2
    verindex = arange(0, shape(vertices)[0])
    nelement = shape(inn)[0]
    t1=0
    f=10
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

    inn2 = zeros((shape(inn)[0],shape(inn)[1]+3)) # element connectivity 2, nelement by 6, top boundary
    inn2[:,:-3] = inn.astype(int)  # first 3 columns of inn2 = inn
    for i in range(shape(inn)[0]): # loop through each element
        for j in range(3): # loop through the 3 nodes of each element i
            if mesh[i,j,2] == 0:  # y of global nodal j in element i is on the specified bottom boundary.
               Tens[(inn[i, j]) * 2 + 1,:,:] = 0
               Tens[(inn[i, j]) * 2 + 1,(inn[i, j]) * 2 + 1,:] = 1/10000
               if mesh[i,j,1] == (w_centre):  # station leftest node. or the middle node
                  
                  Tens[(inn[i, j]) * 2,:,:] = 0
                  Tens[(inn[i, j]) * 2,(inn[i, j]) * 2,:] = 1/10000
            if mesh[i,j,2] == tb: # y of global nodal j in element i is on specified top boundary.
               inn2[i,j+3] = 1
    if flag == 1:  # Dirichlet BC
        print("Boundary conditions: top boundary Dirichlet BC...")
        
        # Assignment states that the top is distorted by 20%
        for i in range(shape(inn)[0]): # loop through each element
            for j in range(3): # loop through the 3 nodes of each element i
                if mesh[i,j,2] == tb: # y of global nodal j in element i is on specified top boundary.
                   Tens[(inn[i,j])*2 + 1,:,:] = 0
                   Tens[(inn[i, j]) * 2 + 1,(inn[i, j]) * 2 + 1,:] = 1/10000

    if flag == 2:  # Neumann BC
        print("Boundary conditions: top boundary Neumann BC...")

    return Tens

