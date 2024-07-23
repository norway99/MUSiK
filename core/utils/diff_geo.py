import numpy as np

class Mesh:

    def __init__(self,

                 ):
        

def __inner_product(a, b):

def __cross_product(a, b):

def __norm(a): 

def boundary(mesh): # get polygonal boundary of manifold represented as a mesh

def set_partition(polygon, points): # partition an open or closed polygon by a set of points

def parameterize(mesh): # convert from triangular mesh to parameterization M

def fiber_bundle(M, product_options, P = None): # given a parameterized manifold compute the tangent or normal bundle, optionally at a point P 

def basis(M): #  compute an oriented basis (Euclidean) everywhere

def gradient(M, P = None): # compute the surface gradient, optionally at a point P

def fundamental_form(M, which, P = None): # compute first, second, and/or third fundamental forms, optionally at a point P

def pip(P, polygon): # is point P in the specified poylgon?

def int_with_sphere(M, center, radius): # intersect M with a sphere parameterized by a center and a radius

def march(P_o, d, tmax): # marching algorithm starting at point P_o close to intersection curve J 
    
def curvepoint(P_o, epsilon): # curvepoint algorithm with convergence threshold epsilon 

def quadratic_solver(u, v, d, grad): # solve a system of equations for delta u, delta v (time-varying)
