import numpy as np
import os
import scipy.interpolate
import matplotlib.pyplot as plt

def calc_Ae(triangles, vertices):
    """
	:param: triangles, vertices: see meshgen2d.py
	:return:
	Ae: array of element areas.
	"""
    n_element = len(triangles)  # number of elements
    mesh = np.zeros((n_element, 3, 5))
    Ae = np.zeros(n_element)
    # Loop through each element
    for i in range(n_element):  # loop through each element
        # calculates the area of each element
        Ax = vertices[triangles[i, 0], 0]
        Ay = vertices[triangles[i, 0], 1]
        Bx = vertices[triangles[i, 1], 0]
        By = vertices[triangles[i, 1], 1]
        Cx = vertices[triangles[i, 2], 0]
        Cy = vertices[triangles[i, 2], 1]

        Ae[i] = (1 / 2) * np.abs(Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))
        if Ae[i] == 0:
            print('Warning: Ae = 0')
            print(triangles[i])
            print(vertices[triangles[i]])

    return Ae
