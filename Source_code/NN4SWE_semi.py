#  Copyright (C) 2025
#  
#  Amin Nadimy, Boyang Chen, Zimo Chen, Claire Heaney, Christopher Pain
#  Applied Modelling and Computation Group
#  Department of Earth Science and Engineering
#  Imperial College London
#
#  amin.nadimy19@imperial.ac.uk
#  
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation,
#  version 3.0 of the License.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.

#-- Import general libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import flex_lib as fl

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    num_gpu_devices = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(i) for i in range(num_gpu_devices)]

    print(f"Number of available GPU devices: {num_gpu_devices}")
    device = []
    for i, device_name in enumerate(device_names):
        device.append(torch.device(f"cuda:{i}"))
        print(f"GPU {i}: {device_name}, {device[i]}")
    
    device = device[0]
        
else:
    device = 'cpu'
    print("No GPU devices available. Using CPU.")
print(device)
is_gpu = torch.cuda.is_available()

#######################################################
################# Numerical parameters ################
ntime = 489600              
n_out = 10000               
nrestart = 0              
ctime_old = 0           
mgsolver = True           
nsafe = 0.5              
ctime = 0                   
save_fig = False            
Restart = False          
epsilon_k = 1e-04         
epsilon_eta = 1e-04      
beta = 4                   
real_time = torch.tensor([0.0], device=device)
istep = 0

# # # ################################### # # #
# # # ######   Physical parameters    ### # # #
# # # ################################### # # #
dt = torch.tensor([0.5], device=device)
g_x = 0;g_y = 0;g_z = 9.81  
rho = 1/g_z             

global_ny, global_nx = 612, 952
hal = torch.tensor(4, device=device)

no_domains = 7
x = [0, 800, 0,  634, 0,   150, 700]
y = [0, 0,   90, 90,  300, 300, 300]
nx = [800, global_nx-800, 634, global_nx-634, 150,           550,            global_nx-700]
ny = [90,  90,            210, 210,           global_ny-300, global_ny-300,  global_ny-300]
ratio_list = [0.5, 1, 2, 2, 0.5, 1, 0.5]

dx = torch.tensor([5.0]*no_domains, device=device)
dy = dx

sd = fl.init_subdomains('linear', no_domains, nx, ny, dx, dy, x, y, ratio_list, epsilon_eta, dt)
fl.find_neighbours(sd)
fl.set_physical_boundaries(sd, global_nx, global_ny)

print('Before scaling')
for i in sd:
    print(f'{i.index}: ({i.nx},{i.ny}), dx = {i.dx}')

print(''); print('After scaling')    
fl.resize_subdomains(sd, ratio_list)


for i in sd:
    print(f'{i.index}: ({i.nx},{i.ny}), dx = {i.dx} ratio = {i.ratio}')
    i.nx = torch.tensor(i.nx).to(device)
    i.ny = torch.tensor(i.ny).to(device)


with open('carlisle-5m.dem.raw', 'r') as file:
    # Read the entire content of the file and split it into individual values
    data = file.read().split()
mesh = torch.tensor([float(value) for value in data[12:]])
mesh = mesh.reshape(int(data[3]),int(data[1]))

mesh = torch.nn.functional.interpolate(
    mesh.unsqueeze(0).unsqueeze(0),
    size=(mesh.shape[0] + 1, mesh.shape[1] + 1),
    mode='bilinear',
    align_corners=True
).squeeze(0).squeeze(0)


def extract_subdomains(mesh, x_coords: list, y_coords: list, nx_list: list, ny_list: list, ratio_list: list)->list:
    '''
    Splits the mesh into the tiles with the same size of the subdomains
    x_coods: list of x coordinates of the top left point of all subdomain relative the global domain
    y_coods: list of y coordinates of the top left point of all subdomain relative the global domain
    nx_list: list of all subdomain size in x
    ny_list: list of all subdomain size in y
    '''
    subdomains = []
    for x, y, nx, ny in zip(x_coords, y_coords, nx_list, ny_list):
        subdomain = mesh[y:y+ny, x:x+nx]
        subdomains.append(subdomain)
    
    resampled_subdomains = []
    for subdomain, ratio in zip(subdomains, ratio_list):
        new_size = (int(subdomain.shape[0] * ratio), int(subdomain.shape[1] * ratio))
        resampled_subdomain = F.interpolate(subdomain.unsqueeze(0).unsqueeze(0), size=new_size, mode='bilinear', align_corners=False)
        resampled_subdomain = resampled_subdomain.squeeze(0).squeeze(0)
        resampled_subdomains.append(resampled_subdomain)
    
    del subdomains
    return resampled_subdomains

        
tiles = extract_subdomains(mesh,x, y, nx, ny, ratio_list)

x_origin = 338500 ; y_origin = 554700

df = pd.read_csv(f'carlisle.bci', delim_whitespace=True)

x_upstream1 = [] ; y_upstream1 = []
x_upstream2 = [] ; y_upstream2 = []
x_upstream3 = [] ; y_upstream3 = []
for index, row in df.iterrows():
    # Check if the 'discharge' column in the current row is 'upstream3'
    if row['discharge'] == 'upstream1':
        # Append the value from the second column to the list
        x_upstream1.append((df['x'][index] - x_origin)//5)
        y_upstream1.append((df['y'][index] - y_origin)//5)
    elif row['discharge'] == 'upstream2':
        x_upstream2.append((df['x'][index] - x_origin)//5)
        y_upstream2.append((df['y'][index] - y_origin)//5)
    elif row['discharge'] == 'upstream3':
        x_upstream3.append((df['x'][index] - x_origin)//5)
        y_upstream3.append((df['y'][index] - y_origin)//5)

y_upstream1 = [global_ny-i for i in y_upstream1]
y_upstream2 = [global_ny-i for i in y_upstream2]
y_upstream3 = [global_ny-i for i in y_upstream3]

def find_index(xp, yp):
    for ele in sd:
        if (ele.x2 <= xp) and (xp <= ele.x2 + ele.nx // ele.ratio) and (ele.y2 <= yp) and (yp <= ele.y2 + ele.ny // ele.ratio):
            return ele.index
    return None
source_1_sd_idx = list(map(lambda xy: find_index(xy[0], xy[1]), zip(x_upstream1, y_upstream1)))
source_2_sd_idx = list(map(lambda xy: find_index(xy[0], xy[1]), zip(x_upstream2, y_upstream2)))
source_3_sd_idx = list(map(lambda xy: find_index(xy[0], xy[1]), zip(x_upstream3, y_upstream3)))

df = pd.read_csv(f'flowrates.csv', delim_whitespace=True)
rate1 = torch.tensor(df['upstream1'].values / 5, device=device)
rate2 = torch.tensor(df['upstream2'].values / 5, device=device)
rate3 = torch.tensor(df['upstream3'].values / 5, device=device)        


# update the source coordinates based on the x2 of each subdomain
x_upstream1 = list(map(lambda i: x_upstream1[i], range(len(x_upstream1))))
x_upstream2 = list(map(lambda i: x_upstream2[i], range(len(x_upstream2))))
x_upstream3 = list(map(lambda i: x_upstream3[i], range(len(x_upstream3))))

y_upstream1 = list(map(lambda i: y_upstream1[i], range(len(y_upstream1))))
y_upstream2 = list(map(lambda i: y_upstream2[i], range(len(y_upstream2))))
y_upstream3 = list(map(lambda i: y_upstream3[i], range(len(y_upstream3))))

def scale_no_source_coor(a, rr):
    '''
    It returns a new list of coordinates based on number of source nodes in subdomains with different resolutions
    i.e. if resolution increases the number of source points also increases
    rr: ratio of the subdomain
    a:: list of coordinates
    '''
    result = []
    for i, r in zip(a, rr):
        if r < 1:
            # Use set to remove duplicates and list comprehension for scaling
            result.extend(list({int(i * r)}))
        elif r > 1:
            # Use nested list comprehension for generating range of values
            result.extend([val for val in range(int(i * r), int(i * r + r))])
        elif r==1:
            # Add the original value if r is equal to 1
            result.append(i)
    return result


# lists of ratios of sd where each source point is located
source1_ratio = [sd[idx].ratio for idx in source_1_sd_idx]
source2_ratio = [sd[idx].ratio for idx in source_2_sd_idx]
source3_ratio = [sd[idx].ratio for idx in source_3_sd_idx]

# Apply the scale_no_coor function to all elements of a
x_upstream1 = scale_no_source_coor(x_upstream1, source1_ratio)
x_upstream2 = scale_no_source_coor(x_upstream2, source2_ratio)
x_upstream3 = scale_no_source_coor(x_upstream3, source3_ratio)

y_upstream1 = scale_no_source_coor(y_upstream1, source1_ratio)
y_upstream2 = scale_no_source_coor(y_upstream2, source1_ratio)
y_upstream3 = scale_no_source_coor(y_upstream3, source1_ratio)

global_ny = 844
global_nx = 1317

input_shape = (1,1,global_ny, global_nx)

new_x0 = [639,  1045, 0,    0,   1220, 639,  1191]
new_y0 = [330,  320,  424,  0,   165,    0,    0]


values_H = torch.zeros(input_shape, device=device)
for i in range(len(sd)):
    values_H[0,0, new_y0[i]: new_y0[i] + sd[i].ny,         
                  new_x0[i]: new_x0[i] + sd[i].nx ] = tiles[i]

for i in range(no_domains):
    sd[i].x0 = new_x0[i]
    sd[i].x1 = new_x0[i] + sd[i].nx.item()
    sd[i].x2 = new_x0[i]
    sd[i].x3 = new_x0[i] + sd[i].nx.item()
    
    sd[i].y0 = new_y0[i] + sd[i].ny.item()
    sd[i].y1 = new_y0[i] + sd[i].ny.item()
    sd[i].y2 = new_y0[i]
    sd[i].y3 = new_y0[i]

def transform_middle_point(x0, x1, x2, x3, x5):
    """
    Transforms x1 to x4 based on the transformation of x0 → x3 and x2 → x5.

    Args:
        x0, x1, x2: original reference points (x1 is between x0 and x2)
        x3, x5: new points corresponding to transformed x0 and x2

    Returns:
        x4: transformed version of x1
    """
    r = (x1 - x0) / (x2 - x0)
    x4 = x3 + r * (x5 - x3)

    return int(x4)

for i in range(len(x_upstream1)):
    s_idx= source_1_sd_idx[i]
    x_upstream1[i] = transform_middle_point(x[s_idx], x_upstream1[i], x[s_idx]+sd[s_idx].nx, sd[s_idx].x0, sd[s_idx].x1)

for i in range(len(y_upstream1)):
    s_idx= source_1_sd_idx[i]
    y_upstream1[i] = transform_middle_point(y[s_idx], y_upstream1[i], y[s_idx]+sd[s_idx].ny, sd[s_idx].y3, sd[s_idx].y0)

for i in range(len(x_upstream2)):    
    s_idx= source_2_sd_idx[i]
    x_upstream2[i] = transform_middle_point(x[s_idx], x_upstream2[i], x[s_idx]+sd[s_idx].nx, sd[s_idx].x0, sd[s_idx].x1)
    
for i in range(len(y_upstream2)):
    s_idx= source_2_sd_idx[i]
    y_upstream2[i] = transform_middle_point(y[s_idx], y_upstream2[i], y[s_idx]+sd[s_idx].ny, sd[s_idx].y3, sd[s_idx].y0)

for i in range(len(x_upstream3)):    
    s_idx= source_3_sd_idx[i]
    x_upstream3[i] = transform_middle_point(x[s_idx], x_upstream3[i], x[s_idx]+sd[s_idx].nx, sd[s_idx].x0, sd[s_idx].x1)
    
for i in range(len(y_upstream3)):
    s_idx= source_3_sd_idx[i]
    y_upstream3[i] = transform_middle_point(y[s_idx], y_upstream3[i], y[s_idx]+sd[s_idx].ny, sd[s_idx].y3, sd[s_idx].y0)


# ------------------------------------------------------------------------------- 2 to 0.5
src_2x_half_x  = []
src_2x_half_x += [i for i in range(sd[2].x0+sd[0].shared_indices['bottom'][2]['neig_start'], sd[2].x0+sd[0].shared_indices['bottom'][2]['neig_end'])]     # bottom sd0 part 1
src_2x_half_x += [i for i in range(sd[3].x0+sd[0].shared_indices['bottom'][3]['neig_start'], sd[3].x0+sd[0].shared_indices['bottom'][3]['neig_end'])]     # bottom sd0 part 2
src_2x_half_x += [i for i in range(sd[2].x0+sd[4].shared_indices['top'][2]['neig_start'],    sd[2].x0+sd[4].shared_indices['top'][2]['neig_end'])]        # top sd4
src_2x_half_x += [i for i in range(sd[3].x0+sd[6].shared_indices['top'][3]['neig_start'],    sd[3].x0+sd[6].shared_indices['top'][3]['neig_end'])]        # top sd6

src_2x_half_y = []
src_2x_half_y += [sd[2].y3+2 for i in range(sd[0].shared_indices['bottom'][2]['neig_start'], sd[0].shared_indices['bottom'][2]['neig_end'])]     # bottom sd0 part 1
src_2x_half_y += [sd[3].y3+2 for i in range(sd[0].shared_indices['bottom'][3]['neig_start'], sd[0].shared_indices['bottom'][3]['neig_end'])]     # bottom sd0 part 2
src_2x_half_y += [sd[2].y0-2 for i in range(sd[4].shared_indices['top'][2]['neig_start'],    sd[4].shared_indices['top'][2]['neig_end'])]        # top sd4
src_2x_half_y += [sd[3].y0-2 for i in range(sd[6].shared_indices['top'][3]['neig_start'],    sd[6].shared_indices['top'][3]['neig_end'])]        # top sd6
# ------------------------------------------------------------------------------- 2 to 1
src_2x_1_x  = []
src_2x_1_x += [i for i in range(sd[3].x0+sd[1].shared_indices['bottom'][3]['neig_start'], sd[3].x0+sd[1].shared_indices['bottom'][3]['neig_end'])]        # bottom sd1
src_2x_1_x += [i for i in range(sd[2].x0+sd[5].shared_indices['top'][2]['neig_start'],    sd[2].x0+sd[5].shared_indices['top'][2]['neig_end'])]           # top sd5

src_2x_1_y = []
src_2x_1_y += [sd[3].y3+2 for i in range(sd[1].shared_indices['bottom'][3]['neig_start'], sd[1].shared_indices['bottom'][3]['neig_end'])]        # bottom sd1
src_2x_1_y += [sd[2].y0-2 for i in range(sd[5].shared_indices['top'][2]['neig_start'],    sd[5].shared_indices['top'][2]['neig_end'])]           # top sd5
# ------------------------------------------------------------------------------- 2 to 2
src_2x_2_x = []
src_2x_2_x += [sd[3].x0+2 for i in range(sd[2].shared_indices['right'][3]['neig_start'], sd[2].shared_indices['right'][3]['neig_end'])]          # right sd2
src_2x_2_x += [sd[2].x1-2 for i in range(sd[3].shared_indices['left'][2]['neig_start'],  sd[3].shared_indices['left'][2]['neig_end'])]           # left sd3

src_2x_2_y  = []
src_2x_2_y += [i for i in range(sd[3].y3, sd[3].y0)]         # right sd2
src_2x_2_y += [i for i in range(sd[2].y3, sd[2].y0)]         # left sd3

# ------------------------------------------------------------------------------- 1 to 0.5
src_1x_half_x  = []
src_1x_half_x += [sd[1].x0+2 for i in range(sd[0].shared_indices['right'][1]['neig_start'], sd[0].shared_indices['right'][1]['neig_end'])]       # right sd0
src_1x_half_x += [sd[5].x0+2 for i in range(sd[4].shared_indices['right'][5]['neig_start'], sd[4].shared_indices['right'][5]['neig_end'])]       # right sd4
src_1x_half_x += [sd[5].x1-2 for i in range(sd[6].shared_indices['left'][5]['neig_start'],  sd[6].shared_indices['left'][5]['neig_end'])]        # left sd6

src_1x_half_y  = []
src_1x_half_y += [i for i in range(sd[1].y3, sd[1].y0)]      # right sd0
src_1x_half_y += [i for i in range(sd[5].y3, sd[5].y0)]      # right sd4
src_1x_half_y += [i for i in range(sd[5].y3, sd[5].y0)]      # left sd6

# ------------------------------------------------------------------------------- 1 to 1
# ------------------------------------------------------------------------------- 1 to 2
src_1x_2_x  = []
src_1x_2_x += [i for i in range(sd[1].x0+sd[3].shared_indices['top'][1]['neig_start'],    sd[1].x0+sd[3].shared_indices['top'][1]['neig_end'])]           # top sd3
src_1x_2_x += [i for i in range(sd[5].x0+sd[2].shared_indices['bottom'][5]['neig_start'], sd[5].x0+sd[2].shared_indices['bottom'][5]['neig_end'])]        # bottom sd2
src_1x_2_x += [i for i in range(sd[5].x0+sd[3].shared_indices['bottom'][5]['neig_start'], sd[5].x0+sd[3].shared_indices['bottom'][5]['neig_end'])]        # bottom sd3

src_1x_2_y  = []
src_1x_2_y += [sd[1].y0-2 for i in range(sd[3].shared_indices['top'][1]['neig_start'],    sd[3].shared_indices['top'][1]['neig_end'])]           # top sd3
src_1x_2_y += [sd[5].y3+2 for i in range(sd[2].shared_indices['bottom'][5]['neig_start'], sd[2].shared_indices['bottom'][5]['neig_end'])]        # bottom sd2
src_1x_2_y += [sd[5].y3+2 for i in range(sd[3].shared_indices['bottom'][5]['neig_start'], sd[3].shared_indices['bottom'][5]['neig_end'])]        # bottom sd3

# ------------------------------------------------------------------------------- 0.5 to 0.5
# ------------------------------------------------------------------------------- 0.5 to 1
src_half_1_x  = []
src_half_1_x += [sd[0].x1-2 for i in range(sd[1].shared_indices['left'][0]['neig_start'],  sd[1].shared_indices['left'][0]['neig_end'])]       # left sd1
src_half_1_x += [sd[4].x1-2 for i in range(sd[5].shared_indices['left'][4]['neig_start'],  sd[5].shared_indices['left'][4]['neig_end'])]       # left sd5
src_half_1_x += [sd[6].x0+2 for i in range(sd[5].shared_indices['right'][6]['neig_start'], sd[5].shared_indices['right'][6]['neig_end'])]      # right sd5

src_half_1_y  = []
src_half_1_y += [ i for i in range(sd[0].y3,sd[0].y0)]       # left sd1
src_half_1_y += [ i for i in range(sd[4].y3, sd[4].y0)]      # left sd5
src_half_1_y += [ i for i in range(sd[6].y3, sd[6].y0)]      # right sd5

# ------------------------------------------------------------------------------- 0.5 to 2
src_half_2_x  = []
src_half_2_x += [i for i in range(sd[0].x0+sd[2].shared_indices['top'][0]['neig_start'],    sd[0].x0+sd[2].shared_indices['top'][0]['neig_end'])]        # top sd2
src_half_2_x += [i for i in range(sd[0].x0+sd[3].shared_indices['top'][0]['neig_start'],    sd[0].x0+sd[3].shared_indices['top'][0]['neig_end'])]        # top sd3
src_half_2_x += [i for i in range(sd[4].x0+sd[2].shared_indices['bottom'][4]['neig_start'], sd[4].x0+sd[2].shared_indices['bottom'][4]['neig_end'])]     # bottom sd2
src_half_2_x += [i for i in range(sd[6].x0+sd[3].shared_indices['bottom'][6]['neig_start'], sd[6].x0+sd[3].shared_indices['bottom'][6]['neig_end'])]     # bottom sd3

src_half_2_y  = []
src_half_2_y += [sd[0].y0-2 for i in range(sd[0].x0+sd[2].shared_indices['top'][0]['neig_start'],    sd[0].x0+sd[2].shared_indices['top'][0]['neig_end'])]        # top sd2
src_half_2_y += [sd[0].y0-2 for i in range(sd[0].x0+sd[3].shared_indices['top'][0]['neig_start'],    sd[0].x0+sd[3].shared_indices['top'][0]['neig_end'])]        # top sd3
src_half_2_y += [sd[4].y3+2 for i in range(sd[2].x0+sd[2].shared_indices['bottom'][4]['neig_start'], sd[2].x0+sd[2].shared_indices['bottom'][4]['neig_end'])]     # bottom sd2
src_half_2_y += [sd[6].y3+2 for i in range(sd[6].x0+sd[3].shared_indices['bottom'][6]['neig_start'], sd[6].x0+sd[3].shared_indices['bottom'][6]['neig_end'])]     # bottom sd3


# ------------------------------------------------------------------------------- 2 to 0.5
rec_2x_half_x  = []
rec_2x_half_x += [i for i in range(sd[0].x0+sd[0].shared_indices['bottom'][2]['my_start'], sd[0].x0+sd[0].shared_indices['bottom'][2]['my_end'])]     # bottom sd0 part 1
rec_2x_half_x += [i for i in range(sd[0].x0+sd[0].shared_indices['bottom'][3]['my_start'], sd[0].x0+sd[0].shared_indices['bottom'][3]['my_end'])]     # bottom sd0 part 2
rec_2x_half_x += [i for i in range(sd[4].x0+sd[4].shared_indices['top'][2]['my_start'],    sd[4].x0+sd[4].shared_indices['top'][2]['my_end'])]        # top sd4
rec_2x_half_x += [i for i in range(sd[6].x0+sd[6].shared_indices['top'][3]['my_start'],    sd[6].x0+sd[6].shared_indices['top'][3]['my_end'])]        # top sd6

rec_2x_half_y = []
rec_2x_half_y += [sd[0].y0-1 for i in range(sd[0].shared_indices['bottom'][2]['my_start'], sd[0].shared_indices['bottom'][2]['my_end'])]     # bottom sd0 part 1
rec_2x_half_y += [sd[0].y0-1 for i in range(sd[0].shared_indices['bottom'][3]['my_start'], sd[0].shared_indices['bottom'][3]['my_end'])]     # bottom sd0 part 2
rec_2x_half_y += [sd[4].y3+1 for i in range(sd[4].shared_indices['top'][2]['my_start'],    sd[4].shared_indices['top'][2]['my_end'])]        # top sd4
rec_2x_half_y += [sd[6].y3+1 for i in range(sd[6].shared_indices['top'][3]['my_start'],    sd[6].shared_indices['top'][3]['my_end'])]        # top sd6

# ------------------------------------------------------------------------------- 2 to 1
rec_2x_1_x  = []
rec_2x_1_x += [i for i in range(sd[1].x0+sd[1].shared_indices['bottom'][3]['my_start'], sd[1].x0+sd[1].shared_indices['bottom'][3]['my_end'])]        # bottom sd1
rec_2x_1_x += [i for i in range(sd[5].x0+sd[5].shared_indices['top'][2]['my_start'],    sd[5].x0+sd[5].shared_indices['top'][2]['my_end'])]           # top sd5

rec_2x_1_y = []
rec_2x_1_y += [sd[1].y0-1 for i in range(sd[1].shared_indices['bottom'][3]['my_start'], sd[1].shared_indices['bottom'][3]['my_end'])]        # bottom sd1
rec_2x_1_y += [sd[5].y3+1 for i in range(sd[5].shared_indices['top'][2]['my_start'],    sd[5].shared_indices['top'][2]['my_end'])]           # top sd5

# ------------------------------------------------------------------------------- 2 to 2
rec_2x_2_x = []
rec_2x_2_x += [sd[2].x1-1 for i in range(sd[2].shared_indices['right'][3]['my_start'], sd[2].shared_indices['right'][3]['my_end'])]          # right sd2
rec_2x_2_x += [sd[3].x0+1 for i in range(sd[3].shared_indices['left'][2]['my_start'],  sd[3].shared_indices['left'][2]['my_end'])]           # left sd3

rec_2x_2_y  = []
rec_2x_2_y += [i for i in range(sd[2].y3, sd[2].y0)]          # right sd2
rec_2x_2_y += [i for i in range(sd[3].y3, sd[3].y0)]          # left sd3

# ------------------------------------------------------------------------------- 1 to 0.5
rec_1x_half_x  = []
rec_1x_half_x += [sd[0].x1-1 for i in range(sd[0].shared_indices['right'][1]['my_start'], sd[0].shared_indices['right'][1]['my_end'])]       # right sd0
rec_1x_half_x += [sd[4].x1-1 for i in range(sd[4].shared_indices['right'][5]['my_start'], sd[4].shared_indices['right'][5]['my_end'])]       # right sd4
rec_1x_half_x += [sd[6].x0+1 for i in range(sd[6].shared_indices['left'][5]['my_start'],  sd[6].shared_indices['left'][5]['my_end'])]        # left sd6

rec_1x_half_y  = []
rec_1x_half_y += [i for i in range(sd[0].y3, sd[0].y0)]       # right sd0
rec_1x_half_y += [i for i in range(sd[4].y3, sd[4].y0)]       # right sd4
rec_1x_half_y += [i for i in range(sd[6].y3, sd[6].y0)]       # left sd6

# ------------------------------------------------------------------------------- 1 to 1
# ------------------------------------------------------------------------------- 1 to 2
rec_1x_2_x  = []
rec_1x_2_x += [i for i in range(sd[3].x0+sd[3].shared_indices['top'][1]['my_start'], sd[3].x0+sd[3].shared_indices['top'][1]['my_end'])]           # top sd3
rec_1x_2_x += [i for i in range(sd[2].x0+sd[2].shared_indices['bottom'][5]['my_start'], sd[2].x0+sd[2].shared_indices['bottom'][5]['my_end'])]     # bottom sd2
rec_1x_2_x += [i for i in range(sd[3].shared_indices['bottom'][5]['my_start'], sd[3].shared_indices['bottom'][5]['my_end'])]                       # bottom sd3

rec_1x_2_y  = []
rec_1x_2_y += [sd[3].y3+1 for i in range(sd[3].x0+sd[3].shared_indices['top'][1]['my_start'], sd[3].x0+sd[3].shared_indices['top'][1]['my_end'])]  # top sd3
rec_1x_2_y += [sd[2].y0-1 for i in range(sd[2].shared_indices['bottom'][5]['my_start'], sd[2].shared_indices['bottom'][5]['my_end'])]              # bottom sd2
rec_1x_2_y += [sd[3].y0-1 for i in range(sd[3].shared_indices['bottom'][5]['my_start'], sd[3].shared_indices['bottom'][5]['my_end'])]              # bottom sd3

# ------------------------------------------------------------------------------- 0.5 to 0.5
# ------------------------------------------------------------------------------- 0.5 to 1
rec_half_1_x  = []
rec_half_1_x += [sd[1].x0+1 for i in range(sd[1].shared_indices['left'][0]['my_start'],  sd[1].shared_indices['left'][0]['my_end'])]       # left sd1
rec_half_1_x += [sd[5].x0+1 for i in range(sd[5].shared_indices['left'][4]['my_start'],  sd[5].shared_indices['left'][4]['my_end'])]       # left sd5
rec_half_1_x += [sd[5].x1-1 for i in range(sd[5].shared_indices['right'][6]['my_start'], sd[5].shared_indices['right'][6]['my_end'])]      # right sd5

rec_half_1_y  = []
rec_half_1_y += [ i for i in range(sd[1].y3, sd[1].y0)]       # left sd1
rec_half_1_y += [ i for i in range(sd[5].y3, sd[5].y0)]       # left sd5
rec_half_1_y += [ i for i in range(sd[5].y3, sd[5].y0)]       # right sd5

# ------------------------------------------------------------------------------- 0.5 to 2
rec_half_2_x  = []
rec_half_2_x += [i for i in range(sd[2].x0+sd[2].shared_indices['top'][0]['my_start'],    sd[2].x0+sd[2].shared_indices['top'][0]['my_end'])]        # top sd2
rec_half_2_x += [i for i in range(sd[3].x0+sd[3].shared_indices['top'][0]['my_start'],    sd[3].x0+sd[3].shared_indices['top'][0]['my_end'])]        # top sd3
rec_half_2_x += [i for i in range(sd[2].x0+sd[2].shared_indices['bottom'][4]['my_start'], sd[2].x0+sd[2].shared_indices['bottom'][4]['my_end'])]     # bottom sd2
rec_half_2_x += [i for i in range(sd[3].x0+sd[3].shared_indices['bottom'][6]['my_start'], sd[3].x0+sd[3].shared_indices['bottom'][6]['my_end'])]     # bottom sd3

rec_half_2_y  = []
rec_half_2_y += [sd[2].y3+1 for i in range(sd[2].x0+sd[2].shared_indices['top'][0]['my_start'],    sd[2].x0+sd[2].shared_indices['top'][0]['my_end'])]        # top sd2
rec_half_2_y += [sd[3].y3+1 for i in range(sd[3].x0+sd[3].shared_indices['top'][0]['my_start'],    sd[3].x0+sd[3].shared_indices['top'][0]['my_end'])]        # top sd3
rec_half_2_y += [sd[2].y0-1 for i in range(sd[2].x0+sd[2].shared_indices['bottom'][4]['my_start'], sd[2].x0+sd[2].shared_indices['bottom'][4]['my_end'])]     # bottom sd2
rec_half_2_y += [sd[3].y0-1 for i in range(sd[3].x0+sd[3].shared_indices['bottom'][6]['my_start'], sd[3].x0+sd[3].shared_indices['bottom'][6]['my_end'])]     # bottom sd3


src_2x_half_x = torch.tensor(src_2x_half_x).to(device)
src_2x_1_x    = torch.tensor(src_2x_1_x).to(device)
src_2x_2_x    = torch.tensor(src_2x_2_x).to(device)
src_1x_half_x = torch.tensor(src_1x_half_x).to(device)
src_1x_2_x    = torch.tensor(src_1x_2_x).to(device)
src_half_1_x  = torch.tensor(src_half_1_x).to(device)
src_half_2_x  = torch.tensor(src_half_2_x).to(device)

src_2x_half_y = torch.tensor(src_2x_half_y).to(device)
src_2x_1_y    = torch.tensor(src_2x_1_y).to(device)
src_2x_2_y    = torch.tensor(src_2x_2_y).to(device)
src_1x_half_y = torch.tensor(src_1x_half_y).to(device)
src_1x_2_y    = torch.tensor(src_1x_2_y).to(device)
src_half_1_y  = torch.tensor(src_half_1_y).to(device)
src_half_2_y  = torch.tensor(src_half_2_y).to(device)

rec_2x_half_x = torch.tensor(rec_2x_half_x).to(device)
rec_2x_1_x    = torch.tensor(rec_2x_1_x).to(device)
rec_2x_2_x    = torch.tensor(rec_2x_2_x).to(device)
rec_1x_half_x = torch.tensor(rec_1x_half_x).to(device)
rec_1x_2_x    = torch.tensor(rec_1x_2_x).to(device)
rec_half_1_x  = torch.tensor(rec_half_1_x).to(device)
rec_half_2_x  = torch.tensor(rec_half_2_x).to(device)

rec_2x_half_y = torch.tensor(rec_2x_half_y).to(device)
rec_2x_1_y    = torch.tensor(rec_2x_1_y).to(device)
rec_2x_2_y    = torch.tensor(rec_2x_2_y).to(device)
rec_1x_half_y = torch.tensor(rec_1x_half_y).to(device)
rec_1x_2_y    = torch.tensor(rec_1x_2_y).to(device)
rec_half_1_y  = torch.tensor(rec_half_1_y).to(device)
rec_half_2_y  = torch.tensor(rec_half_2_y).to(device)


''' Physical boundary indices'''
# ------------------------------------------------------------------------------- bottom
bottom_x = []
bottom_x += [i for i in range(sd[4].x0, sd[4].x1)]
bottom_x += [i for i in range(sd[5].x0, sd[5].x1)]
bottom_x += [i for i in range(sd[6].x0, sd[6].x1)]

bottom_y = []
bottom_y += [sd[4].y0 for _ in range(sd[4].x0, sd[4].x1)]
bottom_y += [sd[5].y0 for _ in range(sd[5].x0, sd[5].x1)]
bottom_y += [sd[6].y0 for _ in range(sd[6].x0, sd[6].x1)]

# ------------------------------------------------------------------------------- left
left_x = []
left_x += [sd[0].x0 for _ in range(sd[0].y3, sd[0].y0)]
left_x += [sd[2].x0 for _ in range(sd[2].y3, sd[2].y0)]
left_x += [sd[4].x0 for _ in range(sd[4].y3, sd[4].y0)]

left_y = []
left_y += [i for i in range(sd[0].y3, sd[0].y0)]
left_y += [i for i in range(sd[2].y3, sd[2].y0)]
left_y += [i for i in range(sd[4].y3, sd[4].y0)]

# ------------------------------------------------------------------------------- right
right_x = []
right_x += [sd[1].x1-1 for _ in range(sd[1].y3, sd[1].y0)]
right_x += [sd[3].x1-1 for _ in range(sd[3].y3, sd[3].y0)]
right_x += [sd[6].x1-1 for _ in range(sd[6].y3, sd[6].y0)]

right_y = []
right_y += [i for i in range(sd[1].y3, sd[1].y0)]
right_y += [i for i in range(sd[3].y3, sd[3].y0)]
right_y += [i for i in range(sd[6].y3, sd[6].y0)]

# ------------------------------------------------------------------------------- top
top_x  = []
top_x += [i for i in range(sd[0].x0, sd[0].x1)]
top_x += [i for i in range(sd[1].x0, sd[1].x1)]

top_y = []
top_y += [sd[0].y3-1 for _ in range(sd[0].x0, sd[0].x1)]
top_y += [sd[1].y3-1 for _ in range(sd[1].x0, sd[1].x1)]


bottom_x = torch.tensor(bottom_x).to(device)
bottom_y = torch.tensor(bottom_y).to(device)
left_x   = torch.tensor(left_x).to(device)
left_y   = torch.tensor(left_y).to(device)
right_x  = torch.tensor(right_x).to(device)
right_y  = torch.tensor(right_y).to(device)
top_x    = torch.tensor(top_x).to(device)
top_y    = torch.tensor(top_y).to(device)


values_u = torch.zeros(input_shape, device=device)
values_v = torch.zeros(input_shape, device=device)
values_h = torch.zeros(input_shape, device=device)
source_h = torch.zeros(input_shape, device=device)
eta1 = torch.zeros(input_shape, device=device)
eta2 = torch.zeros(input_shape, device=device)
values_hh = torch.zeros(input_shape, device=device)
dif_values_h = torch.zeros(input_shape, device=device)
values_h_old = torch.zeros(input_shape, device=device)
sigma_q = torch.zeros(input_shape, device=device)
k_u = torch.zeros(input_shape, device=device)
k_v = torch.zeros(input_shape, device=device)
k_x = torch.zeros(input_shape, device=device)
k_y = torch.zeros(input_shape, device=device)
b = torch.zeros(input_shape, device=device)

input_shape_pd = (1,1,global_ny+2, global_nx+2)
values_uu = torch.zeros(input_shape_pd, device=device)
values_vv = torch.zeros(input_shape_pd, device=device)
eta1_p = torch.zeros(input_shape_pd, device=device)
dif_values_hh = torch.zeros(input_shape_pd, device=device)
values_hhp = torch.zeros(input_shape_pd, device=device)
values_hp = torch.zeros(input_shape_pd, device=device)
k_uu = torch.zeros(input_shape_pd, device=device)
k_vv = torch.zeros(input_shape_pd, device=device)
# stablisation factor
k1 = torch.ones(input_shape, device=device)*epsilon_eta
k2 = torch.zeros(input_shape, device=device)    
k3 = torch.ones((global_ny, global_nx), device=device)*10**2*0.5/dt # for transient problem


values_h[0,0,:,:] = values_H[0,0,:,:]
values_H[0,0,:,:] = -values_H[0,0,:,:]

# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
dx=4.0
bias_initializer = torch.tensor([0.0], device=device)

w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)

w2 = torch.tensor([[[[1/(12*dx)], [0.0], [-1/(12*dx)]],
                    [[1/(3*dx)] , [0.0], [-1/(3*dx)]] ,
                    [[1/(12*dx)], [0.0], [-1/(12*dx)]]]], device=device)

w3 = torch.tensor([[[[-1/(12*dx)], [-1/(3*dx)], [-1/(12*dx)]],
                    [[0.0]       , [0.0]      , [0.0]]       ,
                    [[1/(12*dx)] , [1/(3*dx)] , [1/(12*dx)]]]], device=device)

wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                    [[0.11] ,  [0.44], [0.11]],
                    [[0.028], [0.11] , [0.028]]]], device=device)


w1 = torch.reshape(w1,(1,1,3,3))
w2 = torch.reshape(w2,(1,1,3,3))
w3 = torch.reshape(w3,(1,1,3,3))
wm = torch.reshape(wm,(1,1,3,3))

g_x = 0;g_y = 0;g_z = 9.81  # Gravity acceleration (m/s2) 
rho = 1/g_z                 # Resulting density
diag = -w1[0,0,1,1]


dy = torch.tensor(dy).to(device)
x =  torch.tensor(x).to(device)
y =  torch.tensor(y).to(device)

havedef average_4_consecutive_pairs(tensor):
    length = tensor.size(0)
    
    if length % 4 != 0:
        raise ValueError('Length of the array must be divisible by 4')
    
    quads = tensor.view(-1, 4)  # Reshape into groups of 4
    averaged = quads.mean(dim=1)
    return averaged


def average_2_consecutive_pairs(tensor):
    length = tensor.size(0)
    # If the tensor has odd length, drop the last element
    if length % 2 != 0:
        raise ValueError('Length of the array must be even')
    
    paired = tensor.view(-1, 2)   # Reshape to pairs and compute the mean across dimension 1
    averaged = paired.mean(dim=1)
    return averaged

 
def repeat_each_element_twice(tensor):
    if tensor.dim() != 1:
        raise ValueError("Input must be a 1D tensor.")
    
    repeated = tensor.repeat_interleave(2)
    return repeated



def repeat_each_element_four_times(tensor):
    if tensor.dim() != 1:
        raise ValueError("Input must be a 1D tensor.")
    
    repeated = tensor.repeat_interleave(4)
    return repeated



def update_halos_single(tensor, src_2x_half_x=src_2x_half_x, src_2x_half_y=src_2x_half_y,
                 rec_2x_half_x=rec_2x_half_x, rec_2x_half_y=rec_2x_half_y,
                         src_2x_1_x=src_2x_1_x, src_2x_1_y=src_2x_1_y,
                 rec_2x_1_x=rec_2x_1_x, rec_2x_1_y=rec_2x_1_y,
                         src_2x_2_x=src_2x_2_x, src_2x_2_y=src_2x_2_y,
                 rec_2x_2_x=rec_2x_2_x, rec_2x_2_y=rec_2x_2_y,
                         src_1x_half_x=src_1x_half_x, src_1x_half_y=src_1x_half_y,
                 rec_1x_half_x=rec_1x_half_x, rec_1x_half_y=rec_1x_half_y,
                         src_1x_2_x=src_1x_2_x, src_1x_2_y=src_1x_2_y,
                 rec_1x_2_x=rec_1x_2_x, rec_1x_2_y=rec_1x_2_y,
                         src_half_1_x=src_half_1_x, src_half_1_y=src_half_1_y,
                 rec_half_1_x=rec_half_1_x, rec_half_1_y=rec_half_1_y,
                         src_half_2_x=src_half_2_x, src_half_2_y=src_half_2_y,
                 rec_half_2_x=rec_half_2_x, rec_half_2_y=rec_half_2_y):
    
    tensor[0,0,rec_2x_half_y, rec_2x_half_x] = average_4_consecutive_pairs(tensor[0,0,    src_2x_half_y, src_2x_half_x])
    tensor[0,0,rec_2x_1_y   , rec_2x_1_x]    = average_2_consecutive_pairs(tensor[0,0,    src_2x_1_y,    src_2x_1_x])
    tensor[0,0,rec_2x_2_y   , rec_2x_2_x]    = tensor[0,0,                                src_2x_2_y,    src_2x_2_x]
    tensor[0,0,rec_1x_half_y, rec_1x_half_x] = average_2_consecutive_pairs(tensor[0,0,    src_1x_half_y, src_1x_half_x])
    tensor[0,0,rec_1x_2_y   , rec_1x_2_x]    = repeat_each_element_twice(tensor[0,0,      src_1x_2_y,    src_1x_2_x])
    tensor[0,0,rec_half_1_y , rec_half_1_x]  = repeat_each_element_twice(tensor[0,0,      src_half_1_y,  src_half_1_x])
    tensor[0,0,rec_half_2_y , rec_half_2_x]  = repeat_each_element_four_times(tensor[0,0, src_half_2_y,  src_half_2_x])
    
def update_physical_boundary(tensor, bottom_x=bottom_x, bottom_y=bottom_y, left_x=left_x, left_y=left_y, right_x=right_x, right_y=right_y, top_x=top_x, top_y=top_y):
    tensor[0,0,bottom_y, bottom_x].fill_(0.0)
    tensor[0,0,left_y, left_x].fill_(0.0)
    tensor[0,0,right_y, right_x].fill_(0.0)
    tensor[0,0,top_y, top_x].fill_(0.0)
    
    
def update_halos(tensor_list, all_tensors=True):
    # update internal boundaries
    for i in tensor_list:
        update_halos_single(i)
    
    if all_tensors == True:
        # update physical boundaries
        for i in tensor_list[2:]:
            update_physical_boundary(i)

        dum_x = [i.item()+1 for i in left_x]
        tensor_list[0][0,0,left_y, dum_x].fill_(0.0)                               # values_u
        tensor_list[1][0,0,left_y, dum_x] = tensor_list[1][0,0,left_y, left_x]     # values_v

        dum_x = [i.item()-1 for i in right_x]
        tensor_list[0][0,0,right_y, dum_x] = tensor_list[0][0,0,right_y, right_x]  # values_u
        tensor_list[1][0,0,right_y, dum_x].fill_(0.0)                              # values_v
        
    else: # for k_uu and k_vv only
        for i in tensor_list:
            update_physical_boundary(i)

# # # ################################### # # #
# # # #########   AI4SWE MAIN ########### # # #
# # # ################################### # # #
class AI4SWE(nn.Module):
    """docstring for two_step"""
    def __init__(self):
        super(AI4SWE, self).__init__()
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.cmm.weight.data = wm

        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer

    def boundary_condition_u(self, values_u, values_uu):
#         update_halos_single(values_u)
        values_uu[0,0,1:-1,1:-1] = values_u[0,0,:,:]
        values_uu[0,0,:,0].fill_(0) 
        values_uu[0,0,:,-1].fill_(0)
        values_uu[0,0,0,:] = values_uu[0,0,1,:] 
        values_uu[0,0,-1,:] = values_uu[0,0,-2,:]
        return values_uu   

    def boundary_condition_v(self, values_v, values_vv):
#         update_halos_single(values_v)
        values_vv[0,0,1:-1,1:-1] = values_v[0,0,:,:]
        values_vv[0,0,:,0] =  values_vv[0,0,:,1]
        values_vv[0,0,:,-1] = values_vv[0,0,:,-2]
        values_vv[0,0,0,:].fill_(0)
        values_vv[0,0,-1,:].fill_(0)
        return values_vv       

    def boundary_condition_h(self, values_h, values_hp):
#         update_halos_single(values_h)
        values_hp[0,0,1:-1,1:-1] = values_h[0,0,:,:]
        values_hp[0,0,:,-1].fill_(0)     
        values_hp[0,0,:,0].fill_(0)
        values_hp[0,0,-1,:].fill_(0)  
        values_hp[0,0,0,:].fill_(0)
        return values_hp 
    
    def boundary_condition_hh(self, values_hh, values_hhp):
#         update_halos_single(values_hh)
        values_hhp[0,0,1:-1,1:-1] = values_hh[0,0,:,:]
        values_hhp[0,0,:,-1].fill_(0)     
        values_hhp[0,0,:,0].fill_(0)
        values_hhp[0,0,-1,:].fill_(0)  
        values_hhp[0,0,0,:].fill_(0)
        return values_hhp 
    
    def boundary_condition_dif_h(self, dif_values_h, dif_values_hh):
#         update_halos_single(dif_values_h)
        dif_values_hh[0,0,1:-1,1:-1] = dif_values_h[0,0,:,:]
        dif_values_hh[0,0,:,-1].fill_(0)     
        dif_values_hh[0,0,:,0].fill_(0)
        dif_values_hh[0,0,-1,:].fill_(0)  
        dif_values_hh[0,0,0,:].fill_(0)
        return dif_values_hh 
    
    def boundary_condition_eta(self, eta, values_hp):
#         update_halos_single(eta)
        values_hp[0,0,1:-1,1:-1] = eta[0,0,:,:]
        values_hp[0,0,:,-1].fill_(0)     
        values_hp[0,0,:,0].fill_(0)
        values_hp[0,0,-1,:].fill_(0)  
        values_hp[0,0,0,:].fill_(0)
        return values_hp  
    
    def boundary_condition_eta1(self, eta1, eta1_p):
#         update_halos_single(eta1)
        eta1_p[0,0,1:-1,1:-1] = eta1[0,0,:,:]
        eta1_p[0,0,:,-1].fill_(0)     
        eta1_p[0,0,:,0].fill_(0)
        eta1_p[0,0,-1,:].fill_(0)  
        eta1_p[0,0,0,:].fill_(0)
        return eta1_p  
    
    def PG_vector(self, values_uu, values_vv, values_u, values_v, k3):
        k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_uu)) / \
            (1e-03  + (torch.abs(self.xadv(values_uu)) * (dx**-2) + torch.abs(self.yadv(values_uu)) * (dx**-2)) / 2)

        k_v = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_vv)) / \
            (1e-03  + (torch.abs(self.xadv(values_vv)) * (dx**-2) + torch.abs(self.yadv(values_vv)) * (dx**-2)) / 2)

        k_uu = F.pad(torch.minimum(k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        k_vv = F.pad(torch.minimum(k_v, k3) , (1, 1, 1, 1), mode='constant', value=0)
        
        update_halos([k_uu[:,:,1:-1, 1:-1], k_vv[:,:,1:-1, 1:-1]], all_tensors=False)
        
        k_x = 0.5 * (k_u * self.diff(values_uu) + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_y = 0.5 * (k_v * self.diff(values_vv) + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        return k_x, k_y

    def PG_scalar(self, values_hh, values_h, values_u, values_v, k3):
        k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_hh)) / \
            (1e-03 + (torch.abs(self.xadv(values_hh)) * (dx**-2) + torch.abs(self.yadv(values_hh)) * (dx**-2)) / 2)  
    
        k_uu = F.pad(torch.minimum(k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        update_halos([k_uu[:,:,1:-1, 1:-1]], all_tensors=False)
        return 0.5 * (k_u * self.diff(values_hh) + self.diff(values_hh * k_uu) - values_h * self.diff(k_uu))        

    def forward(self, values_u, values_uu, values_v, values_vv, values_H, values_h, values_hp, dt, rho, k1, k2, k3, eta1_p, source_h, dif_values_h, dif_values_hh, values_hh, values_hhp, eta, eta1):
        values_uu = self.boundary_condition_u(values_u,values_uu)
        values_vv = self.boundary_condition_v(values_v,values_vv)

        [k_x,k_y] = self.PG_vector(values_uu, values_vv, values_u, values_v, k3)

        values_u = values_u + k_x * dt - values_u * self.xadv(values_uu) * dt - values_v * self.yadv(values_uu) * dt   
        values_v = values_v + k_y * dt - values_u * self.xadv(values_vv) * dt - values_v * self.yadv(values_vv) * dt 
        values_u = values_u - self.xadv(self.boundary_condition_h(values_h,values_hp)) * dt
        values_v = values_v - self.yadv(self.boundary_condition_h(values_h,values_hp)) * dt     

        sigma_q = (values_u**2 + values_v**2)**0.5 * 0.055**2 / (torch.maximum( k1,
            dx*self.cmm(self.boundary_condition_eta(eta,values_hp))*0.01+(values_H+values_h)*0.99 )**(4/3))

        values_u = values_u / (1 + sigma_q * dt / rho)
        values_v = values_v / (1 + sigma_q * dt / rho)

        values_uu = self.boundary_condition_u(values_u,values_uu)
        values_vv = self.boundary_condition_v(values_v,values_vv)
#         eta1 = torch.maximum(k2,(values_H+values_h))
        eta2 = torch.maximum(k1,(values_H+values_h))
        # dbug = eta2 -> eta1
        b = beta * rho * (-self.xadv(self.boundary_condition_eta1(eta1,eta1_p)) * values_u - \
                           self.yadv(self.boundary_condition_eta1(eta1,eta1_p)) * values_v - \
                           eta1 * self.xadv(values_uu) - eta1 * self.yadv(values_vv) + \
                           self.PG_scalar(self.boundary_condition_eta1(eta1,eta1_p), eta1, values_u, values_v, k3) - \
                           self.cmm(self.boundary_condition_dif_h(dif_values_h,dif_values_hh)) / dt + source_h) / (dt * eta2)   
        values_h_old = values_h.clone()
        for i in range(2):
            values_hh = values_hh - (-self.diff(self.boundary_condition_hh(values_hh,values_hhp)) + beta * rho / (dt**2 * eta2) * values_hh) / \
                    (diag + beta * rho / (dt**2 * eta2)) + b / (diag + beta * rho / (dt**2 * eta2))
        values_h = values_h + values_hh
        dif_values_h = values_h - values_h_old 
        values_u = values_u - self.xadv(self.boundary_condition_hh(values_hh,values_hhp)) * dt / rho
        values_v = values_v - self.yadv(self.boundary_condition_hh(values_hh,values_hhp)) * dt / rho 

        return values_u, values_v, values_h, values_hh, b, dif_values_h, sigma_q

model = torch.compile(AI4SWE().to(device))


def get_source(time, rate1, rate2, rate3):
    '''
    sets the source based on variable time compatible with the real data
    '''
    indx = int(time.item() // 900) # 900 is the time interval in the given time series
    
    for i, s_idx in enumerate(source_1_sd_idx):
        source_h[0,0,y_upstream1[i],x_upstream1[i]]     = ((rate1[indx+1] - rate1[indx])/900 * (time%900) + rate1[indx])#/(4**(source1_ratio[i]-1))
    
    for i, s_idx in enumerate(source_2_sd_idx):
        source_h[0,0,y_upstream2[i]-11,x_upstream2[i]]  = ((rate2[indx+1] - rate2[indx])/900 * (time%900) + rate2[indx])#/(4**(source2_ratio[i]-1))
        
    for i, s_idx in enumerate(source_3_sd_idx):
        source_h[0,0,y_upstream3[i]-4,x_upstream3[i]]   = ((rate3[indx+1] - rate3[indx])/900 * (time%900) + rate3[indx])#/(4**(source3_ratio[i]-1))
        


with torch.no_grad():
    for itime in range(1,5400+1):
        get_source(real_time, rate1, rate2, rate3)
        for t in range(2):
            eta1 = torch.maximum(k2,(values_H+values_h))
            eta = values_H+values_h
            update_halos([values_u, values_v, values_h, values_hh, dif_values_h, eta, eta1])
            [values_u, values_v, values_h, values_hh, b, dif_values_h, sigma_q] = model(values_u, values_uu, values_v, values_vv, values_H, values_h, 
                    values_hp, dt, rho, k1, k2, k3, eta1_p, source_h, dif_values_h, dif_values_hh, values_hh, values_hhp, eta, eta1)










