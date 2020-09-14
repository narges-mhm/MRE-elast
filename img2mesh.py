import triangle
import numpy as np
import importlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage.transform import resize#
import GridData
importlib.reload(GridData)
from GridData import *
import scipy.io as sio
from timeit import default_timer as timer

def img2mesh(AE,plotfig):
    start=timer()
    #AE= np.copy(E[25])
    #print('AE.shape[0]',AE.shape[0])
    #print('AE.shape[1]',AE.shape[1])

    A = resize(AE, (np.minimum(AE.shape[0],AE.shape[1]), np.minimum(AE.shape[0],AE.shape[1])))
    area ='qa0.000001'
    height = 30e-3 #mm
    width = A.shape[1]/A.shape[0]*30e-3 #mm
    verta = height#.astype(float32)
    vertb = width#.astype(float32)
    pts2 = np.array([[0, 0], [verta, 0], [0,verta/2],[verta, vertb],[verta,vertb/2], [0, vertb]])
    roi = dict(vertices=pts2)#vertices=pts
    mesh_raw = triangle.triangulate(roi,area)
        
    vertices = mesh_raw['vertices']
    triangles = mesh_raw['triangles']
    N=len(vertices)
    #print('vertices size',len(vertices))
    x = np.array(vertices[:,0])
    y =  np.array(vertices[:,1])  
    x_new = np.linspace(x.min(),x.max(),N)
    y_new = np.linspace(y.min(),y.max(),N)
    X,Y=np.meshgrid(x_new,y_new,indexing='ij')#x_new, y_new
    #xx=X[:,0]#yy=Y[0,:]
    step=10
    x0 = np.arange(0,A.shape[0],step)
    y0 = np.arange(0,A.shape[1],step)
    xi,yi=np.meshgrid(x0, y0)
    #disp_fig_x = np.around(griddata(x,y,disp[::2],X,Y,interp=u'linear'),decimals=10)
    xi = xi/np.max(xi)*30e-3
    yi = yi/np.max(yi)*30e-3
    x1 = np.squeeze(np.reshape(xi,(xi.size,1)))
    y1 = np.squeeze(np.reshape(yi,(yi.size,1)))
    A1 = np.squeeze(np.reshape(A[::step,::step],(A[::step,::step].size,1)))
    z = griddata((x1,y1),A1,(X,Y),method='linear')
    #print('shape of z=',z.shape)
    zvect=GridData(y,x,z,vertices)
    E_interp=np.around(griddata((x,y),np.ravel(zvect),(X,Y),method='linear'),decimals=10)#+0.1*np.ones(N)
    if plotfig==1:
        plt.subplot(121);plt.imshow(np.abs(E_interp),vmin=0,vmax=1)
        plt.subplot(122);plt.imshow(z)
    
    print("--- %s seconds ---" % (timer()-start))  
    return z, np.ravel(zvect), vertices, triangles
#F=scatteredInterpolant(x,y,double(A),'natural');
#int=F(Ex,Ey);