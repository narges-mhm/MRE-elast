import importlib
from calc_Ae import *
from genBe import *
import global_stiffness3;importlib.reload(global_stiffness3);from global_stiffness3 import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from boundaries2 import *
from boundariesTens import *
from timeit import default_timer as timer
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import scipy.io as sio
import img2mesh;importlib.reload(img2mesh);from img2mesh import *
import matplotlib.image as mpimg

import imp
import random
from numba import jit
#@jit(nopython=True,parallel=True)
def openfile_general(filename):
    start = timer() 
    Eimg=mpimg.imread(filename)
    plotfig=0
    z, Evect, vertices, triangles=img2mesh(Eimg,plotfig)
    E=np.ravel(Evect)
    E = E.astype('float32')*1e+5
    v = 0.495 # Poissons Ratio
    flag=2
    f=10
    N=len(vertices)
    Ae = calc_Ae(triangles, vertices)
    Be = genBe(triangles, vertices, Ae)
    GK,Tens,KT= global_stiffness(Ae,Be,E,v, triangles, len(triangles), len(vertices))
    [inn2,fxy,GK] = boundaries2(flag,triangles,vertices,GK,f)
    GK_s = csr_matrix(GK)
    disp1 = scipy.sparse.linalg.spsolve(GK_s,fxy)#fxy = np.reshape(fxy, (-1, 1))
    disp1=np.around(disp1,decimals=10)*1e+5#disp1 = np.reshape(disp1, (-1, 1))# make 1D vector to 2D matrix
    Tens = boundariesTens2(flag,triangles,vertices,Tens)#.astype(float16)
    matTens=np.transpose(Tens, (2, 0, 1))

    print("--- %s seconds ---" % (timer()-start)) #
    return E*1e-5, disp1, Tens, KT, matTens, triangles, vertices, fxy

