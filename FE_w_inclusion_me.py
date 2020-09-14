# Abrar F
# Usage : module load python3/3.5.2 

import triangle
from calc_Ae import *
from genBe import *
from local_stiffness import *
from global_stiffness3 import *
from set_boundary_point import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.mlab import griddata
from cd import cd
import pickle
from numpy import *
import os,shutil
import scipy.io as sio

def boundaries2(flag,mesh,inn,tb,verindex,t1,t2,vertices,GK,bcy,width):
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
        print("Boundary conditions: top boundary Neumann BC...")
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


def save_obj(obj, name ):
    with open('mesh_dict/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('mesh_dict/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def boundaries(mask, sample_flag, value, sampling):
    """
    This function generates boundary points from any image arrays.
    :param sample_flag: = 1 : sampling;  = 0: no sampling
    :param mask: input mask array
    :param sampling: sampling: by how much the input is sampled.
    :return: idx2: x, y coordinates of boundary points
    """
    m, n = mask.shape

    # mask2 is zero padded at the edge
    mask2 = np.zeros((m + 2, n + 2))
    mask2[1:m + 1, 1:n + 1] = mask

    # idx to store the boundary point coordinates
    idx = np.zeros((1, 2))
    for i in range(m):
        for j in range(n):
            # if any of the nearest neighbor is 0 and the point itself is 1, --> boundary point
            if mask[i, j] == value and any([mask2[i + 2, j + 1] == 0, mask2[i + 1, j + 2] == 0, mask2[i, j + 1] == 0, mask2[i + 1, j] == 0]):
                idx = np.append(idx, [[j, i]], axis=0)
    idx = idx[1:, :]

    # Sample the boundary points
    if sample_flag == 1:
        # random_rows = np.random.choice(p, p // sampling, replace=False)
        # idx = idx[random_rows, :]
        idx = idx[::sampling]
    return idx

def getData(fileNo,f,depth,E_inclusion=10e3,E_background= 20e3,display='on',save_dir='/scratch/mdoyley_lab/afaiyaz/GenerateDispData_FEM/Saved_dir/',inc_radius=3e-3,l_x=2,l_y=2):
    #E_background= 5e3 #kPa
    
    global gb_flag
    global saveMesh
    global saveDisp
    v = 0.495 # Poissons Ratio
    height = 30e-3 #mm
    width = 30e-3 #mm
    N=256
    lateral_flag = 1
    BCflag = 1      #Boundary condition by strain= 0 ; force =1
    strain= 0.02    #0.50 here means = 50%
    force = np.array([0,-f])
    
    ###########################
    area ='qa0.000001'     # 'qa0.0000001'
    theta = linspace(0, 2 * pi, 30)[:-1]
    r = inc_radius#3e-3#1e-3
    verta = height
    vertb = width
    loc1=l_x#3
    loc2=l_y#3/2
    loc3=4
    loc4=4
    pts1 = vstack(((cos(theta)) * r + (height/loc1), (sin(theta)) * r + (width/loc2))).T
    pts3 = vstack(((cos(theta)) * r + (height/loc3), (sin(theta)) * r + (width/loc4))).T

    #pts2 = array([[0, 0], [verta, 0], [verta, vertb], [0, vertb]])
    pts2 = np.array([[0, 0], [verta, 0], [0,verta/2],[verta, vertb],[verta,vertb/2], [0, vertb]])
    pts = vstack((pts1, pts3, pts2)) # combined ROI
    roi = dict(vertices=pts2)#vertices=pts
    
    #global mesh_raw_gb
    if gb_flag == 0 :
        print('Working with new mesh')
        
        mesh_raw = triangle.triangulate(roi,area)
        if saveMesh==1:
            save_obj(mesh_raw,'mesh_9')
        #mesh_raw_gb= mesh_raw.copy();
    else :
        mesh_name='mesh_9'
        mesh_raw=load_obj(mesh_name)
        print('Loaded '+mesh_name)
        #mesh_raw = mesh_raw_gb.copy()
        
    vertices = mesh_raw['vertices']
    triangles = mesh_raw['triangles']
    n_node = len(vertices)
    n_ele = len(triangles)
    print('Number of nodes: ',n_node,' Number of elements: ',n_ele)
    
    #vertices_2 = np.zeros((n_node,3))
    E = np.ones(n_node)*E_background 
    df_1='on'
    df_2='on'
    radius=r

    for i in range(n_node):
        if np.sqrt((vertices[i,0]-(height/loc1))**2+(vertices[i,1]-(width/loc2))**2) <= r+ 1e-4:
            E[i] = E_inclusion
            #vertices_2[i,2] = 1
     # Can be changed to accomodate for different number of lesions
    for i in range(n_node):
        #if np.sqrt((vertices[i,0]-center[0]*scale_x)**2+(vertices[i,1]-center[1]*scale_y)**2) <= radius:
        #    E[i] = E_inclusion
        #    vertices_2[i,2] = 1
        if (depth ==1 or df_1=='on') and np.sqrt((vertices[i,0]-(height/loc1))**2+(vertices[i,1]-(width/loc2))**2) <= radius:
            E[i] = E_inclusion
            #vertices_2[i,2] = 1
        if (depth ==2 or df_2=='on') and np.sqrt((vertices[i,0]-(height/loc3))**2+(vertices[i,1]-(width/loc4))**2) <= radius:
            E[i] = E_inclusion
            #vertices_2[i,2] = 1
        '''if (depth ==3 or df_3=='on') and np.sqrt((vertices[i,0]-center3[0]*scale_x)**2+(vertices[i,1]-center3[1]*scale_y)**2) <= radius:
            E[i] = E_inclusion
            vertices_2[i,2] = 1'''
        
    if display=='on':        
        plt.figure(figsize=(13,13))
        plt.scatter(vertices[:,0],vertices[:,1],c=E,s = 19);plt.axis([0,width,0,height])
        plt.colorbar()
        plt.show()
    ##################################################
    

    Ae = calc_Ae(triangles, vertices)
    Be = genBe(triangles, vertices, Ae)
    Ke = local_stiffness(lateral_flag, Ae, Be, E, v, triangles)
    GK, Tens = global_stiffness(Ae,Be,E,v, triangles, len(triangles), len(vertices))


    ############################################## BC
    bcflag=2
    tb=height
    bcy=-0.2
    verindex = np.arange(0, np.shape(vertices)[0])
    nelement = np.shape(triangles)[0]
    t1=0
    t2=-f
    r=0
    mesh = np.zeros((nelement, 3, 4))
    for i in range(nelement):  # loop through each element
        for j in range(3):  # loop through each local nodes of each element
            # Insert the global nodal index
            mesh[i, j, 0] = triangles[i, j]
            # Insert the (x,y) coordinates of the global/local node
            mesh[i, j, 1:3] = vertices[triangles[i, j], :]
            # Check whether each node (global) falls within a prescribed circle with radius r in ROI.
            #if np.sqrt((mesh[i, j, 1] - 10) ** 2 + (mesh[i, j, 2] - 10) ** 2) < (r + 0.05):
            #    mesh[i, j, 3] = 1  # marked for region 1 if falls within the circle. Otherwise marked for region 0.
    [inn2,fxy,GK] = boundaries2(bcflag,mesh,triangles,tb,verindex,t1,t2,vertices,GK,bcy,width) 

    ###################### Solve f=K disp 
    
    GK_s = csr_matrix(GK)
    disp = scipy.sparse.linalg.spsolve(GK_s,fxy)
    print('Shape of Global stiffness matrix - ',np.shape(GK))
    
    ##################################################
    ###### Print result (FEM forward)

    x = vertices[:,0]
    y = vertices[:,1]
    
    X_prev,Y_prev = np.meshgrid(x, y)
    x_new = np.linspace(x.min(),x.max(),N)
    y_new = np.linspace(y.min(),y.max(),N)
    X, Y = np.meshgrid(x_new, y_new)
    disp=np.around(disp,decimals=10)
    disp_fig_x = np.around(griddata(x,y,disp[::2],X,Y,interp=u'linear'),decimals=10)
    disp_fig_y =np.around(griddata(x,y,disp[1::2],X,Y,interp=u'linear'),decimals=10)
    E_mod= np.around(griddata(x,y,E,X,Y,interp=u'linear'),decimals=10)
    print('Max disp',np.min(disp[1::2]))
    if saveDisp==1:
        #directly output to fromFEM folder of fieldII
        #save png for dispY with names fileNo
        import matplotlib
        matplotlib.image.imsave('./Saved_dir/image_inv/'+fileNo+'.png', disp_fig_y,cmap="gray",vmax =0  , vmin = -4.0258e-05)
        matplotlib.image.imsave('./Saved_dir/label_inv/'+fileNo+'.png', E_mod,cmap="gray" ,vmin=1000, vmax=50000)
        '''fig=plt.imshow(disp_fig_y,cmap='jet', aspect='equal');plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(fileNo+'.png', bbox_inches='tight',pad_inches=0)
        '''
        #sio.savemat(save_dir+'true_disps/d_field2_disp'+ str(fileNo)+'.mat', {'dispX': disp_fig_x.T, 'dispY':disp_fig_y.T,'x_new':x_new,'y_new':y_new})
    if display=='on':
        plt.figure(figsize=(10,10))
        plt.subplot(2,2,1);plt.imshow(disp_fig_x,cmap='jet');plt.colorbar();plt.title('dispX/m')
        plt.subplot(2,2,4);plt.imshow(GK,cmap='jet');plt.colorbar();plt.title('Global Stiffness matrix')
        plt.subplot(2,2,2);plt.imshow(disp_fig_y,cmap='jet');plt.colorbar();plt.title('dispY/m')
        plt.subplot(2,2,3);plt.plot(disp_fig_y[0,:]);plt.title('bottom');#plt.ylim(-.03, 0.03)
        #plt.subplot(2,2,4);plt.plot(disp_fig_y[150,:]);plt.title('top');#plt.ylim(-0.03,0.03)
    #plt.subplot(2,2,3);plt.imshow(np.diff(disp_fig_y, axis=0));plt.colorbar();plt.title('strainY')
    #plt.subplot(2,2,4);plt.imshow(np.diff(disp_fig_x, axis=1));plt.colorbar();plt.title('strainX')
    plt.show()

    try:
        os.mkdir(save_dir+'M'+str(fileNo))
    except FileExistsError:
        shutil.rmtree(save_dir+'M'+str(fileNo))
        os.mkdir(save_dir+'M'+str(fileNo))
    '''if saveDisp==1:
        with cd(save_dir+'M'+str(fileNo)+'/'):
            np.savetxt('Final_E.txt', E)
            np.savetxt('vertices.txt', vertices)
            np.savetxt('uxy.txt', disp)
            np.savetxt('triangles.txt', triangles)'''
    
    

import sys

default_dir = '/scratch/mdoyley_lab/afaiyaz/GenerateDispData_FEM/Saved_dir/'
fileNo=str(sys.argv[1])
force=float(sys.argv[2]) #N
E_inclusion = float(sys.argv[3])#17e3 #kPa
E_background = float(sys.argv[4])#17e3 #kPa
inclusion_radius = float(sys.argv[5]) 
x=float(sys.argv[6])
y=float(sys.argv[7])

gb_flag=0  #0 - make new mesh 
           #1 - load mesh      #hardcoded 'mesh_9' saved in "mesh_dir"

saveDisp=0#1
saveMesh=1

getData(fileNo, force, 0, E_inclusion, E_background, display='on', save_dir=default_dir,inc_radius=inclusion_radius,l_x=x,l_y=y)
# python3 -W ignore FE_w_inclusion_savedisps.py 1234 10 20000 10000


#### Consider the following for normalizing criteria
# find min disp for max stiffness and max disp for min stiffness
###    disp; vmax =0  , vmin = -4.0258e-05'''(has to be for minimum background 10kPa for position[2 1.5])'''         #-0.0002264925##-4.5299e-06
###    E; vmin = 1kPa, vmax = 50kPa while saving. White lesion means hard lesion. Comparatively black lesion indicates soft lesion
 
# For U-net, generate data for
#                    - background   10kPa,      20kPa,    30kPa
#					 - inlusion    1 to 50 , interval 3kPa
#                    - # of inclusion (exclude for now)
#                    - size of inclusion 2 3 4 5 6
#                    - positions of inclusion 3*3
#                    - N Force applied 10N.
