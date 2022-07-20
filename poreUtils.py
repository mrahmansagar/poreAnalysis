# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:47:49 2022

@author: Mmr Sagar
Phd Student | AG Alves 
MPI for Multidisciplinary Sciences 
=====================================

Utility functions for pore analysis 
"""

# Required libraries 
import numpy as np
import ctypes
import matplotlib.pyplot as plt

def swigObjt2uint8Array(swigPyObj, xdim, ydim, zdim, plot=False):
    """
    SWIGOBJECT2UINT8ARRAY function takes a swig object with dimension specification
    and returns an Numpy array of defined dimension with data type unsigned 8bit
    

    Parameters
    ----------
    swigPyObj : SwigPyObject
        SwigPyObject is an object wrapped by swig .
    xdim : int
        Dimension of array in X.
    ydim : int
        Dimension of array in Y.
    zdim : int
        Dimension of array in Z.
    plot : bool, optional
        If set True then plots the 150th slice of the volume in gray
        scale. The default is False.

    Returns
    -------
    vol_arr : numpy.ndarray ('uint8')
        Numpy array of size specified of data type uint8.

    """
    # types of data to be extracted from memory 
    pt = ctypes.c_ubyte*xdim*ydim*zdim
    
    vol_arr = np.array(list(pt.from_address(int(swigPyObj)))) # int(swigObject) gives the memory address 
    
    if plot:
        plt.figure()
        plt.imshow(vol_arr[150, :, :], cmap='gray')
        plt.show()
    
    return vol_arr 
    




def norm16bit(v):
    """
    NORM16BIT function takes an array and normalized it before converting it into 
    a 16 bit unsigned integer and returns it.

    Parameters
    ----------
    v : numpy.ndarray
        Array of N dimension.

    Returns
    -------
    numpy.ndarray (uint16)
        Numpy Array of same dimension as input with data type as unsigned integer 16 bit

    """
    
    mn = v.min()
    mx = v.max()
      
    mx -= mn
      
    v = ((v - mn)/mx) * 65535
    
    return v.astype(np.uint16)
    
    
    
def norm8bit(v):
    """
    NORM8BIT function takes an array and normalized it before converting it into 
    a 8 bit unsigned integer and returns it.

    Parameters
    ----------
    v : numpy.ndarray
        Array of N dimension.

    Returns
    -------
    numpy.ndarray (uint8)
        Numpy Array of same dimension as input with data type as unsigned integer 8 bit

    """
    
    mn = v.min()
    mx = v.max()
      
    mx -= mn
      
    v = ((v - mn)/mx) * 255
    
    return v.astype(np.uint8)



def formatBasicStats(basic_stats, definition=False, result=True):
    """
    FORMATBASICSTATS function takes a swig object produced py_p3dBasicAnalysis
    and returns a dictionary with formatted values and definition.
    
    Parameters
    ----------
    basic_stats : swigObject
        Swig object produced by py_p3dBasicAnalysis().
    definition : bool, optional
        When Ture definition of the parameters are also added. The default is False.
    result : bool, optional
        When True result are added to the dictionary. The default is True.
    
    Returns
    -------
    basic_analysis : dict
        A dictionary formatted with definition and values.

    """
    
    basic_analysis = {}
    if definition:
        def_basic = {'VV': 'Density (VV). A measure of density based on the number of object voxels with respect to the total number of volume  voxels.',
                     'SV': 'Specific surface area (SV) [mm-1]. A measure of the surface of the object with respect to the total volume. Tipically it is related to the mechanical properties of the object.',
                     'MV': 'Integral of mean curvature (MV)[mm-2]. A positive value implies the dominance of convex structures, while MV < 0 occurs  in the case of predominance of concave structures.',
                     'CV': 'Euler characteristic (CV)[mm-3]. This is an index of connectivity of the object network.'
                     }
    
        basic_analysis['definition'] = def_basic
        
    if result:
        res_basic = {'VV': basic_stats.Vv,
                     'SV': basic_stats.Cv,
                     'MV': basic_stats.Vv,
                     'CV': basic_stats.Cv}
    
        basic_analysis['result'] = res_basic
    
    return basic_analysis



def formatAnisotropyStats(anisotropy_stats, definition=False, result=True):
    """
    FORMATANISOTROPYSTATS function takes a swig object produced py_p3dAnisotropyAnalysis
    and returns a dictionary with formatted values and definition.

    Parameters
    ----------
    anisotropy_stats : swigObject
        Swig object produced by py_p3dAnisotropyAnalysis().
    definition : bool, optional
        When Ture definition of the parameters are also added. The default is False.
    result : bool, optional
        When True result are added to the dictionary. The default is True.
        
    Returns
    -------
    anisotropy_analysis : dict
        A dictionary formatted with definition and values.

    """
    anisotropy_analysis = {}
    if definition:
        def_anisotropy = {'I': 'Isotropy index. It measures the similarity of a fabric to a uniform distribution and varies between 0 (all observation  confined to a single plane or axis) and 1 (perfect isotropy)', 
                          'E': 'Elongation index. It measures the preferred orientation of a fabric in the u1/u2 plane and varies between 0 (no  preferred orientation) and 1 (a perfect preferred orientation with all observations parallel).'}
        
        anisotropy_analysis['definition'] = def_anisotropy
    
    if result:
        res_anisotropy = {'E': anisotropy_stats.E,
                          'I': anisotropy_stats.I}
    
        anisotropy_analysis['result'] = res_anisotropy
    
    return anisotropy_analysis



def formatBlobStats(blob_stats, definition=False, result=True):
    """
    FORMATBLOBSTATS function takes a swig object produced py_p3dBlobAnalysis
    and returns a dictionary with formatted values and definition.

    Parameters
    ---------- 
    blob_stats : swigObject
        Swig object produced by py_p3dBlobAnalysis().
    definition : bool, optional
        When Ture definition of the parameters are also added. The default is False.
    result : bool, optional
        When True result are added to the dictionary. The default is True.

    Returns
    -------
    blob_analysis : dict
        A dictionary formatted with definition and values.

    """
    blob_analysis = {}
    if definition:
        def_blob = { 'COUNT': 'The number of identified blobs.',
                     'VOLUME': '[mm3] An array of length COUNT with the volume of each identified blob computed as the number of voxels rescaled  according to the specified voxel size.', 
                     'MAX_SPHERE': '[mm] An array of length COUNT with the diameter of the maximum inscribed sphere of each identified blob. It is  computed as two times the maximum value of the Euclidean distance transform within the blob.', 
                     'EQ_SPHERE': '[mm] An array of length COUNT with the diameter of the equivalent sphere, i.e. the diameter of a sphere with the  same volume as the blob. It is computed exploiting the inverse formula of the volume of a sphere.', 
                     'MIN_AXIS': '[mm] An array of length COUNT with the minor axis length, i.e. the length of the shortest segment among all the  segments fully included into the blob and passing through its center of mass. The so-called “star” of segments from which  selecting the shortest is generated using random orientations. The "star" image can be optionally returned as output in order  to determine if more random segments have to be computed.', 
                     'MAX_AXIS': '[mm] An array of length COUNT with the major axis length, i.e. the length of the longest segment among all the  segments fully included into the blob and passing through its center of mass.', 
                     'SPHERICITY': 'An array of length COUNT with the ratio of MAX_SPHERE and EQ_SPHERE for each blob.', 
                     'ASPECT_RATIO': 'An array of length COUNT with the ratio of MIN_AXIS and MAX_AXIS for each blob.',
                     'EXTENT': 'An array of length COUNT with the ratio between the volume of the blob and the volume of the minimum bounding box,  i.e. the smallest parallelepiped oriented according to image axis containing the blob.'
                     }
    
        blob_analysis['definition'] = def_blob
    if result:
        mem_pt = ctypes.c_double*blob_stats.blobCount
        res_blob = {'COUNT': blob_stats.blobCount,
                    'VOLUME': list(mem_pt.from_address(int(blob_stats.volume))),
                    'MAX_SPHERE': list(mem_pt.from_address(int(blob_stats.max_sph))),
                    'EQ_SPHERE': list(mem_pt.from_address(int(blob_stats.eq_sph))),
                    'MIN_AXIS': list(mem_pt.from_address(int(blob_stats.l_min))),
                    'MAX_AXIS': list(mem_pt.from_address(int(blob_stats.l_max))),
                    'SPHERICITY': list(mem_pt.from_address(int(blob_stats.sphericity))),
                    'ASPECT_RATIO': list(mem_pt.from_address(int(blob_stats.aspect_ratio))),
                    'EXTENT': list(mem_pt.from_address(int(blob_stats.extent))),
                    }
        blob_analysis['result'] = res_blob
    return blob_analysis


def formatSklStats(skl_stats, definition=False, result=True):
    """
    FORMATSKLSTATS function takes a swig object produced py_p3dSkeltonAnalysis
    and returns a dictionary with formatted values and definition.

    Parameters
    ----------
    skl_stats : swigObject
        Swig object produced by py_p3dSkeltonAnalysis().
    definition : bool, optional
        When Ture definition of the parameters are also added. The default is False.
    result : bool, optional
        When True result are added to the dictionary. The default is True.
        
    Returns
    -------
    skl_analysis : dict
        A dictionary formatted with definition and values.

    """
    skl_analysis = {}
    if definition:
        def_skl = {'CONNECTIVITY_DENSITY': '[mm-3]: A scalar value representing the number of redundant connections normalized to the total volume V.  It is computed as (1 - ΧV )/V where ΧV = (n - b), being n the number of pores and b the number of node-to-node branches.', 
                   'COORDINATION_NUMBER': 'An array of length PORES_COUNT containing the number of branches that spread out from each node.', 
                   'PORES_COUNT': 'An integer value representing the number of pores determined after the application of the merging criterion.  Therefore, it does not necessarly correspond to the number of skeleton nodes.', 
                   'PORES_WIDTH': '[mm]: An array of length PORES_COUNT containing the pore-size distribution computed as diameter of the maximal  inscribed sphere for each pore. The center of the maximal sphere is affected by the merging criterion.', 
                   'ENDPOINTS_COUNT': 'An integer value representing the number of skeleton end points.', 
                   'ENDPOINTS_WIDTH': '[mm]: An array of length ENDPOINTS_COUNT containing the width of each end point computed as the diameter of the  maximal sphere centered on the end point.', 
                   'ENDTOEND_COUNT': 'An integer value representing the number of end-to-end branches.',
                   'ENDTOEND_LENGTH': '[mm]: An array of length ENDTOEND_COUNT containing the length of each end-to-end branch computed from the  surface to the maximal sphere of an end point to the surface of the maximal sphere of the other end point.',
                   'ENDTOEND_MEANWIDTH': '[mm]: An array of length ENDTOEND_COUNT containing the mean width of each endToEndBranches. The width is  computed averaging the diameter of the maximal spheres of each branch voxel.', 
                   'ENDTOEND_MINWIDTH': '[mm]: An array of length ENDTOEND_COUNT containing the minimum width of each end-to-end branch. This value is  the diameter of the smallest maximal spheres among all the maximal spheres centered on each branch voxel.',
                   'ENDTOEND_MAXWIDTH': '[mm]: An array of length ENDTOEND_COUNT containing the maximum width of each end-to-end branch. This value is  the diameter of the largest maximal spheres among all the maximal spheres centered on each branch voxel.',
                   'NODETOEND_COUNT': 'An integer value representing the number of node-to-end branches.',
                   'NODETOEND_LENGTH': '[mm]: An array of length NODETOEND_COUNT containing the length of each node-to-end branch computed from the  surface to the maximal sphere of the node point to the surface of the maximal sphere of the end point.', 
                   'NODETOEND_MEANWIDTH': '[mm]: An array of length NODETOEND_COUNT containing the mean width of each node-to-end branch. The width is  computed averaging the diameter of the maximal spheres of each branch voxel.',
                   'NODETOEND_MINWIDTH': '[mm]: An array of length NODETOEND_COUNT containing the minimum width of each node-to-end branch. This value  is the diameter of the smallest maximal spheres among all the maximal spheres centered on each branch voxel.', 
                   'NODETOEND_MAXWIDTH': '[mm]: An array of length NODETOEND_COUNT containing the maximum width of each node-to-end branch. This value  is the diameter of the largest maximal spheres among all the maximal spheres centered on each branch voxel.',
                   'NODETONODE_COUNT': 'An integer value representing the number of node-to-node branches.', 
                   'NODETONODE_LENGTH': '[mm]: An array of length NODETONODE_COUNT containing the length of each node-to-node branch computed from the  surface of the maximal sphere inscribed within the pore to the surface of the maximal sphere of the other pore.',
                   'NODETONODE_MEANWIDTH': '[mm]: An array of length NODETONODE_COUNT containing the mean width of each node-to-node branch. The width  is computed averaging the diameter of the maximal spheres of each branch voxel.',
                   'NODETONODE_MINWIDTH': '[mm]: An array of length NODETONODE_COUNT containing the minimum width of each node-to-node branch. This  value is the diameter of the smallest maximal spheres among all the maximal spheres centered on each branch voxel. The smallest  thickness along a node-to-node branch is usually defined as throat.',
                   'NODETONODE_MAXWIDTH': '[mm]: An array of length NODETONODE_COUNT containing the maximum width of each node-to-node branch. This  value is the diameter of the largest maximal spheres among all the maximal spheres centered on each branch voxel.'
                   }
    
        skl_analysis['definition'] = def_skl
    if result:
        res_skl = {'CONNECTIVITY_DENSITY': skl_stats.ConnectivityDensity}
        
        if skl_stats.Node_Counter != 0:
            pt = ctypes.c_int*skl_stats.Node_Counter
            res_skl['COORDINATION_NUMBER'] = list(pt.from_address(int(skl_stats.CoordinationNumber)))
        else:
            pass
        
        res_skl['PORES_COUNT'] = skl_stats.Node_Counter
        
        if skl_stats.Node_Counter != 0:
            pt = ctypes.c_double*skl_stats.Node_Counter
            res_skl['PORES_WIDTH'] = list(pt.from_address(int(skl_stats.Node_Width)))
        else:
            pass
        
        res_skl['ENDPOINTS_COUNT'] = skl_stats.End_Counter
        
        if skl_stats.End_Counter !=0:
            pt = ctypes.c_double*skl_stats.End_Counter
            res_skl['ENDPOINTS_WIDTH'] = list(pt.from_address(int(skl_stats.End_Width)))
        else:
            pass
        
        res_skl['ENDTOEND_COUNT'] = skl_stats.EndToEnd_Counter
        
        if skl_stats.EndToEnd_Counter != 0:
            pt = ctypes.c_double*skl_stats.EndToEnd_Counter
            res_skl['ENDTOEND_LENGTH'] = list(pt.from_address(int(skl_stats.EndToEnd_Length)))
            res_skl['ENDTOEND_MEANWIDTH'] = list(pt.from_address(int(skl_stats.EndToEnd_MeanWidth)))
            res_skl['ENDTOEND_MINWIDTH'] = list(pt.from_address(int(skl_stats.EndToEnd_MinWidth)))
            res_skl['ENDTOEND_MAXWIDTH'] = list(pt.from_address(int(skl_stats.EndToEnd_MaxWidth)))
        else:
            pass
        
        res_skl['NODETOEND_COUNT'] = skl_stats.NodeToEnd_Counter
        if skl_stats.NodeToEnd_Counter != 0:
            pt = ctypes.c_double*skl_stats.NodeToEnd_Counter
            res_skl['NODETOEND_LENGTH'] = list(pt.from_address(int(skl_stats.NodeToEnd_Length)))
            res_skl['NODETOEND_MEANWIDTH'] = list(pt.from_address(int(skl_stats.NodeToEnd_MeanWidth)))
            res_skl['NODETOEND_MINWIDTH'] = list(pt.from_address(int(skl_stats.NodeToEnd_MinWidth)))
            res_skl['NODETOEND_MAXWIDTH'] = list(pt.from_address(int(skl_stats.NodeToEnd_MaxWidth)))
            res_skl['NODETOEND_MAXWIDTH'] = list(pt.from_address(int(skl_stats.NodeToEnd_MaxWidth)))
        else:
            pass
        
        res_skl['NODETONODE_COUNT'] = skl_stats.NodeToNode_Counter
        if skl_stats.NodeToNode_Counter != 0:
            pt = ctypes.c_double*skl_stats.NodeToNode_Counter
            res_skl['NODETONODE_LENGTH'] = list(pt.from_address(int(skl_stats.NodeToNode_Length)))
            res_skl['NODETONODE_MEANWIDTH'] = list(pt.from_address(int(skl_stats.NodeToNode_MeanWidth)))
            res_skl['NODETONODE_MINWIDTH'] = list(pt.from_address(int(skl_stats.NodeToNode_MinWidth)))
            res_skl['NODETONODE_MAXWIDTH'] = list(pt.from_address(int(skl_stats.NodeToNode_MaxWidth)))
        else:
            pass
        
        skl_analysis['result'] = res_skl
    
    return skl_analysis
