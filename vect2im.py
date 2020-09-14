import numpy as np
from scipy.interpolate import griddata
def vect2im(A,vertices):
    N=len(A)
    x = np.array(vertices[:,0])
    y =  np.array(vertices[:,1])   
    x_new = np.linspace(x.min(),x.max(),N)
    y_new = np.linspace(y.min(),y.max(),N)
    X, Y = np.meshgrid(x_new, y_new)
    A_im=np.around(griddata((x,y),np.ravel(A),(X,Y),method='linear'),decimals=10)
    return A_im