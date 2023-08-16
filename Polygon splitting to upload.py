# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:47:02 2022

@author: @Chris Matthew @Islands and Coastal Research Lab
"""

from __future__ import division
from shapely.geometry import Polygon, LineString, Point
import shapely.ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import seaborn as sns
from scipy.spatial import Voronoi
from ast import literal_eval

def PointInPolygon(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
        for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def PointsInPoly(coordlist,n=100):

    #to plot polygon use plt.plot(*poly.exterior.xy)
    geometry = coordlist
    poly = Polygon(geometry)
    bound = poly.bounds
    coords = np.array(geometry)
    
    #make grid of points
    xmin = bound[0]
    xmax = bound[2]
    ymin = bound[1]
    ymax = bound[3]

    x = np.linspace(xmin,xmax,n)
    y = np.linspace(ymin,ymax,n)
    
    grid = np.meshgrid(x,y)
    grid = np.array((grid[0].ravel(),grid[1].ravel())).T
    gridbool = np.array([PointInPolygon(x[0],x[1],coords) for x in grid])
    return grid[gridbool]

def SpectralCluster(coords,n_clusters,plot=True):
    """
    https://juanitorduz.github.io/spectral_clustering/
    """
    xrange = round(max([x[0] for x in coords])-min([x[0] for x in coords]))
    yrange = round(max([x[1] for x in coords])-min([x[1] for x in coords]))
    points = xrange
    if yrange>xrange:
        points = yrange

    grid=PointsInPoly(coords,points)        
    data_df = pd.DataFrame(grid,columns=["x","y"])
    spec_cl = SpectralClustering(
        n_clusters=n_clusters, 
        random_state=25, 
        n_neighbors=8, 
        affinity='nearest_neighbors'
        )
    
    data_df['cluster'] = spec_cl.fit_predict(data_df[['x', 'y']])
    data_df['cluster'] = [c for c in data_df['cluster']]
    
    if plot:
        fig, ax = plt.subplots()
        sns.scatterplot(x ='x', y ='y', data=data_df, 
                        hue='cluster', ax=ax)
    return data_df

def VoronoiFinitePolygon2D(vor, radius=None):
    """
    https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def SplitPolygon(coords, n_clusters, plot=False):
    """
    https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango
    https://gis.stackexchange.com/questions/311536/split-polygon-by-multilinestring-shapely
    """
    poly = Polygon(coords)
    d = SpectralCluster(coords, n_clusters,plot)
    centroids = np.array(d.groupby("cluster").mean())
    voron = Voronoi(centroids)
    regions, vertices = VoronoiFinitePolygon2D(voron) 
    out=[]
    
    
    for region in regions:
        #this gets the boundary of the regoin which will be cut
        boundary_coords = vertices[region]
        boundary_coords = np.vstack((boundary_coords,boundary_coords[0]))
        boundaries = LineString(boundary_coords)
        
        merged_lines = shapely.ops.linemerge([boundaries,poly.boundary])
        border_lines = shapely.ops.unary_union(merged_lines)
        decomposition = list(shapely.ops.polygonize(border_lines))
        #then find the centre of the cut region 
        centre = np.array(centroids)[np.array([Polygon(boundaries).contains(Point(np.array(centroids)[i,:])) for i in range(centroids.shape[0])])]
        for x in decomposition:
            if x.contains(Point(centre[0])):
                out.append(x)
        
    return out

#%%

df = pd.read_csv("sample_polygons_to_split.csv",
                 index_col = 0)
# Need to interpret the coordinates as a list
df = df.geometry.apply(literal_eval)

for i in df:
    SplitPolygon(i,10)
        
