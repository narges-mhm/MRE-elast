
# genBe generates the 3D matrix Be according to P.618 of Reddy's, 3rd edition.
# Input:
#   - mesh: mesh matrix containing the global nodal index and nodal coordinates of each element
#   - Ae: array of area of each element.
# Output:
#   - Be: 3D matrix with dimension of 3*6*number of elements
#         3*6 -- (as defined in P.618, 6 = 2 * 3 nodes
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
def genBe(triangles, vertices, Ae):
    Be = np.zeros((3, 6, len(Ae)))  # See page 618 of Reddy's, 3rd edition. 6 = 2 * 3, where 3 = nodal values.
    for i in range(len(Ae)):
        y23 = vertices[triangles[i, 1], 1] - vertices[triangles[i, 2], 1]
        y31 = vertices[triangles[i, 2], 1] - vertices[triangles[i, 0], 1]
        y12 = vertices[triangles[i, 0], 1] - vertices[triangles[i, 1], 1]
        x32 = vertices[triangles[i, 2], 0] - vertices[triangles[i, 1], 0]
        x13 = vertices[triangles[i, 0], 0] - vertices[triangles[i, 2], 0]
        x21 = vertices[triangles[i, 1], 0] - vertices[triangles[i, 0], 0]
        Be[:, :, i] = (1 / (2 * Ae[i])) * np.array([[y23, 0, y31, 0, y12, 0],[0, x32, 0, x13, 0, x21],[x32, y23, x13, y31, x21, y12]])
    return Be

# mesh =zeros((2,3,3))
# mesh[0,:,:] = array([[1,2,3],[2,3,4],[4,5,6]])
# mesh[1,:,:] = array([[1,1,2],[3,3,4],[5,9,6]])
# Ae=ones(2)
# print(mesh[0,:,:])
# print(mesh[1,:,:])
# Be = genBe(mesh,Ae)
# print(2*Be[:,:,0])
# print(2*Be[:,:,1])
