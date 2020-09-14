import numpy as np
def GridData(x,y,z,vertices):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # make coordinate arrays.
    N=vertices.shape[0]
    #print('N=',N)
    binsize=(xmax-xmin)/(N-1)
    xi      = np.arange(xmin, xmax+binsize, binsize)#np.linspace(xmin,xmax, N)##
    yi      = np.arange(ymin, ymax+binsize, binsize)#np.linspace(ymin,ymax, N)#
    xi, yi = np.meshgrid(xi,yi)

    # make the grid.
    grid           = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape
    zvect=np.zeros((N,1))
    #print('zvect', zvect.shape[0])
    for row in range(nrow):
        for col in range(ncol):
            xc = xi[row, col]    # x coordinate.
            yc = yi[row, col]    # y coordinate.

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
            ind  = np.where(ibin == True)[0]
            zvect[ind]=z[row,col]
    return zvect
            #if retloc: wherebin[row][col] = ind