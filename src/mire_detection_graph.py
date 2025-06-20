# Author: Vaibhav Ganatra (t-vaganatra at microsoft dot com))

import numpy as np
from utils import process_mires
import pandas as pd
import tqdm
import cv2
import logging
import matplotlib.pyplot as plt
from typing import List

def radial_scanning(image_seg: np.array, image_orig: np.array, center: list, 
    jump: float =2, start_angle: float =0, end_angle: float =360):

    image_cent_list = []
    image_inv = 255 - image_seg

    for angle in np.arange(start_angle, end_angle, jump):
        # edge image processing for angle
        cent, line = process_mires(image_seg.copy(), center, angle, weights=image_orig)
        cent_inv, _ = process_mires(image_inv.copy(), center, angle, weights=255-image_orig)
        cent = cent + cent_inv
        cent.sort(key = lambda x: x[2])
        image_cent_list.append(cent)
    
    return image_cent_list

def find_connected_components(data : pd.DataFrame, center: List, radial_tolerance: float = 1):
    logging.info(f"Number of nodes: {len(data)}")    
    nodes = data.values
    edges = [[] for _ in nodes]
    marked_nodes = np.ones((len(nodes))) * -1
    assert len(marked_nodes) == len(nodes)    

    for i, node in enumerate(nodes):
        radius = node[2]
        curr_node = node[3:5]
        unmatched_nodes = nodes[:,3:5]

        radial_vector = (curr_node - center)
        radial_vector /= np.linalg.norm(radial_vector)

        tangential_vector = np.array([-radial_vector[1], radial_vector[0]])

        tangential_tolerance = max(2 * radius * np.pi / 180, 1.5)

        unmatched_nodes = unmatched_nodes - curr_node
        tangential_distances = np.abs(np.dot(unmatched_nodes, tangential_vector))        
        normal_distances = np.abs(np.dot(unmatched_nodes, radial_vector))

        dist_mask = (tangential_distances <= tangential_tolerance) & (normal_distances <= radial_tolerance) & (nodes[:, 1] != node[1])        
        new_nodes = np.nonzero((dist_mask))

        for new_node in new_nodes[0].tolist():
            if new_node == i:
                continue
            if new_node not in edges[i]:
                edges[i].append(new_node)
            if i not in edges[new_node]:
                edges[new_node].append(i)

    components = np.ones((len(nodes))) * -1
    curr_idx = 1
    for i,node in enumerate(nodes):
        if components[i] != -1:
            continue
        current_component = [i]
        current_component_angles = [node[1]]
        while len(current_component) != 0:
            idx = current_component.pop()
            if components[idx] != -1:
                continue
            
            components[idx] = curr_idx
            new_nodes = [e for e in edges[idx] if components[e] == -1 and e not in current_component and nodes[e][1] not in current_component_angles]
            current_component.extend(new_nodes)
            current_component_angles.extend([nodes[e][1] for e in new_nodes])

        curr_idx += 1
    data["component"] = components    
    return data

def correct_mire_locations(data : pd.DataFrame):
    data.sort_values(by=["radius", "mire_num", "angle"], inplace=True)

    components = data["component"].unique()
    for c in components:
        subset = data[data["component"] == c]
        if len(subset) < 10:
            data.loc[data["component"] == c, "component"] = -1

        mire_counts = dict(subset["mire_num"].value_counts())
        num_points = len(subset)
        max_perc = max(mire_counts.values()) / num_points
        if max_perc != 1:
            mires_in_component = dict(sorted(mire_counts.items(), key=lambda x: x[1], reverse=True))
            for idx, (mire_num, count) in enumerate(mires_in_component.items()):
                if idx == 0:
                    current_mire = mire_num
                    continue
                else:
                    diff = current_mire - mire_num
                    angles = subset[subset["mire_num"] == mire_num]["angle"].values
                    data.loc[(data["mire_num"] >= mire_num) & (data["angle"].isin(angles)), "mire_num"] += diff
    data = data[data["component"] >= 0]
    return data

def remove_mire_locations(data: pd.DataFrame):
    mires = sorted(data["mire_num"].unique())
    for mire_num in mires:
        mire = data[data["mire_num"] == mire_num]
        mire = mire.sort_values(by="angle")
        angles = mire["angle"].value_counts()

        angles = angles[angles > 1].index.tolist()
        data.loc[(data["mire_num"] == mire_num) & (data["angle"].isin(angles)), "component"] = -1

    data = data[data["component"] >= 0]
    return data

def visualize_mire_locations(data : dict, img: np.array, out_dir: str):
    colors = np.random.randint(0, 255, (len(data), 3), )
    for mire_num, color in zip(data.keys(), colors):
        for angle, loc in data[mire_num].items():
            img[int(loc[1]), int(loc[0])] = color
    cv2.imwrite(out_dir + "/mire_locations.jpg", img)

def mire_locations(img :np.array, mire_mask: np.array, center: List, n_mires: int, start_angle: float, end_angle: float, jump: float, out_dir: str):
    logging.info("Finding mire locations")
    radii_list = radial_scanning(mire_mask, img, center, start_angle=start_angle, end_angle=end_angle, jump=jump)
    data = []
    for idx, (angle, r_list) in enumerate(zip(np.arange(start_angle, end_angle, jump), radii_list)):
        for mire_num, radius in enumerate(r_list):
            if mire_num >= n_mires:
                break
            data.append([mire_num, angle, radius[2], radius[0], radius[1], radius[3]])
    
    data = np.asarray(data)
    data = pd.DataFrame(data, columns=["mire_num", "angle", "radius", "x", "y", "intensity"])

    logging.info("Finding connected components")
    data = find_connected_components(data, center)
    logging.info("Correcting mire locations")
    data = correct_mire_locations(data)
    logging.info("Removing mire locations")
    data = remove_mire_locations(data)
    data.to_csv(out_dir + "/mire_locations.csv", index=False)

    mire_locations = dict()
    mire_intensities = dict()
    for mire_num in range(n_mires):
        if mire_num == 0:
            continue
        mire_locations[mire_num] = dict()
        mire_intensities[mire_num] = dict()
        for angle in np.arange(start_angle, end_angle, jump):
            if len(data[(data["mire_num"] == mire_num) & (data["angle"] == angle)]) == 0:
                continue
            mire_locations[mire_num][angle] = data[(data["mire_num"] == mire_num) & (data["angle"] == angle)][["x", "y"]].values[0]
            mire_intensities[mire_num][angle] = data[(data["mire_num"] == mire_num) & (data["angle"] == angle)]["intensity"].values[0]
    img_color = np.dstack((img, np.dstack((img, img))))    
    visualize_mire_locations(mire_locations, img_color, out_dir)
    return mire_locations, mire_intensities