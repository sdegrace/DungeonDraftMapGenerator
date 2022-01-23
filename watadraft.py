import io
import json
import re
import random
import itertools
from tkinter import filedialog

from tqdm import tqdm

from numpy.random import default_rng

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import numpy as np
import copy

from shapely.geometry import *
from shapely.geometry.polygon import Polygon
from shapely import ops

from lloyd import Field

import tkinter as tk

rng = default_rng()

num_major_points = 20
num_minor_points = 40
relaxation_amount = 5
exterior_buffer = 0.1
min_building_street_face = 64
mean_building_street_face = 10
max_building_street_face = 512
min_building_depth = 256
mean_building_depth = 7
max_building_depth = 512
min_street_offset = 64
max_street_offset = 128


def gen_node(current):
    current = int(current, 16)
    while True:
        current += 1
        yield hex(current)[2:]


def poly_to_multi_line_string(poly):
    ring = poly.exterior
    if ring.is_ccw:
        ring.coords = ring.coords[::-1]
    return ring_to_mult_line_string(ring)


def ring_to_mult_line_string(ring):
    if ring.is_ccw:
        ring.coords = list(ring.coords)[::-1]
    return MultiLineString(list(itertools.pairwise(list(ring.coords))))


def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


def get_ridges(vor):
    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)
    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])

    return finite_segments, infinite_segments


def construct_streets(poly, n_points):
    min_x, min_y, max_x, max_y = poly.bounds

    x_range = max_x - min_x
    y_range = max_y - min_y

    x_buffer = x_range * exterior_buffer
    y_buffer = y_range * exterior_buffer

    xs = rng.integers(min_x - x_buffer, max_x + x_buffer, n_points)
    ys = rng.integers(min_y - y_buffer, max_y + y_buffer, n_points)
    points = np.vstack([xs, ys]).T
    field = Field(points)

    for i in range(relaxation_amount):
        field.relax()

    mask = []
    for p in field.get_points():
        mask.append(poly.contains(Point(p[0], p[1])))

    culled_field = Field(field.get_points()[mask])

    finite_segments, infinite_segments = get_ridges(culled_field.voronoi)

    streets = MultiLineString(infinite_segments + finite_segments)
    substreets = []
    for line in poly.intersection(streets).geoms:
        substreets.append(list(line.coords))
    split_poly = ops.split(poly_to_multi_line_string(poly), streets)
    return MultiPolygon(ops.polygonize(poly.intersection(streets).union(split_poly))), poly.intersection(streets)


def scatter_buildings(neighborhood):
    last_remainder = None
    boundry = neighborhood.buffer(10, single_sided=True)
    buildings = []
    for side in neighborhood:
        rem_len = side
        first = True
        while rem_len.length > min_building_depth + min_building_street_face:
            max_remain = min(max_building_street_face, rem_len.length - min_building_depth)
            front_len = random.uniform(min_building_street_face, max_remain - min_street_offset)
            if first and last_remainder is not None:
                depth = last_remainder.length
            else:
                depth = random.uniform(min_building_depth, max_building_depth)

            front, rem_len_temp = cut(rem_len, front_len)
            front_offset = random.uniform(min_street_offset, max_street_offset)
            building = front.parallel_offset(front_offset, 'right').union(
                front.parallel_offset(depth, 'right')).minimum_rotated_rectangle
            while not MultiPolygon(buildings + [building] + list(boundry)).is_valid:
                if front_len > min_building_street_face:
                    front_len *= .9
                if depth > min_building_depth:
                    depth *= .9
                front, rem_len_temp = cut(rem_len, front_len)

                if front_len < min_building_street_face and depth < min_building_depth:
                    building = None
                    break
                else:
                    building = front.union(front.parallel_offset(depth, 'right')).minimum_rotated_rectangle

            rem_len = rem_len_temp
            last_remainder = rem_len
            if building:
                buildings.append(building)
            first = False
    return MultiPolygon(buildings)


def parameterize_roof(poly):
    edge = MultiLineString(list(itertools.pairwise(list((poly.boundary).coords))))
    short_ind = np.argmin([ls.length for ls in edge])
    short_cent = edge[short_ind].centroid
    full_cent = poly.centroid
    return np.asarray(short_cent), (np.asarray(full_cent) - np.asarray(short_cent)) * 2 + np.asarray(short_cent), edge[
        short_ind].length / 2


def make_roof(poly, node):
    p1, p2, width = parameterize_roof(poly)
    return {"position": "Vector2( 0, 0 )",
            "rotation": 0,
            "scale": "Vector2(1, 1)",
            "points": f"PoolVector2Array( {p1[0]}, {p1[1]}, {p2[0]}, {p2[1]} )",
            "texture": "res://textures/roofs/flat_clay_red/tiles.png",
            "width": width,
            "type": 1,
            "node_id": node}


def make_path(mls, node):
    coords = np.asarray(l)
    pos = coords[0]
    ed_pts = coords - pos
    return {
        "position": f"Vector2( {pos[0]}, {pos[1]} )",
        "rotation": 0,
        "scale": "Vector2( 1, 1 )",
        "edit_points": f"PoolVector2Array( {', '.join(ed_pts.flatten().astype('str'))} )",
        "smoothness": 1,
        "texture": "res://textures/paths/wagon_trail.png",
        "width": 320,
        "layer": 100,
        "fade_in": False,
        "fade_out": False,
        "grow": False,
        "shrink": False,
        "loop": False,
        "node_id": node
    }


def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

filename = filedialog.askopenfilename()
with open(filename) as f:
    txt = f.read()
m = json.loads(txt)
new_map = copy.deepcopy(m)

# find largest node in file and then create generator for new nodes... I think this is for undo ops?
next_node = gen_node(hex(max([int(s, 16) for s in re.findall('"node_id": "(.*)"', txt)]))[2:])

t = eval(re.findall('\(.*\)', m['world']['levels']['0']['patterns'][0]['points'])[0])
all_lines = list(itertools.pairwise(list(itertools.pairwise(t + t[:2]))[::2]))

p = Polygon([l[0] for l in all_lines])

polys, streets = construct_streets(p, num_major_points)

fig, ax = plt.subplots()

for poly in polys.geoms:
    plot_polygon(ax, poly)

plt.show()

buildings = []
list_neighborhoods = []
for district in tqdm(list(polys)):
    neighborhood = poly_to_multi_line_string(district)
    buildings.extend(scatter_buildings(neighborhood))
for subpoly in tqdm(list_neighborhoods):
    neighborhood = poly_to_multi_line_string(subpoly)
    buildings.extend(scatter_buildings(neighborhood))

fig, ax = plt.subplots()
for poly in tqdm(MultiPolygon(buildings).geoms):
    plot_polygon(ax, poly)

for street in streets:
    s = np.asarray(street)
    plt.plot(s[:, 0], s[:, 1])

plt.show()

filename = filedialog.asksaveasfilename(defaultextension='dungeondraft_map')

dc_roofs = []
for building in buildings:
    r = make_roof(building, next(next_node))
    dc_roofs.append(r)

dc_paths = []
for l in polys.boundary:
    s = make_path(l, next(next_node))
    dc_paths.append(s)

new_map = copy.deepcopy(m)
new_map['world']['levels']['0']['roofs']['roofs'].extend(dc_roofs)
new_map['world']['levels']['0']['paths'].extend(dc_paths)
len(new_map['world']['levels']['0']['roofs']['roofs'])
with open(filename, 'w') as f:
    json.dump(new_map, f, indent='\t')
