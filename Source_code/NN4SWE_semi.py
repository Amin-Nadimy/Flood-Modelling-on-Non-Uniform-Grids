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

################# Numerical parameters ################
ntime = 489600              # Time steps
n_out = 100                 # Results output
nrestart = 0                # Last time step for restart
ctime_old = 0               # Last ctime for restart
mgsolver = True             # Multigrid solver for non-hydrostatic pressure
nsafe = 0.5                 # Continuty equation residuals
ctime = 0                   # Initialise ctime
save_fig = True             # Save results
Restart = False             # Restart
eplsion_k = 1e-04          # Stablisatin factor in Petrov-Galerkin for velocity
epsilon_eta = 1e-04         # Stablisatin factor in Petrov-Galerkin for height
beta = 4                    # diagonal factor in mass term
real_time = 0
istep = 0
manning = 0.055

dt = 0.5
################# Physical parameters #################
g_x = 0;g_y = 0;g_z = 9.81  # Gravity acceleration (m/s2)
rho = 1/g_z                 # Resulting density

global_nx = 952
global_ny = 612

# -------------------------------------------------- 21 subdomains
nx =  [250,100,200,100,200,global_nx-850, 150, 150, 200, 100, 250, 270, 60, 60, 270, global_nx - 600, 50, 520, 130, 100, global_nx-750]
ny =  [global_ny-430, global_ny-430, global_ny-430, global_ny-430, global_ny-430, global_ny-350, 100, 100, 100, 100, 80, 230, 150, 80, 230, 170, 80, 100, 100, 180, 180]

x = [0, 250, 350, 550, 650, 850, 0, 150, 300, 500, 600, 0, 270, 270, 330, 600, 600, 0, 520, 650, 750]
y = [430, 430, 430, 430, 430, 350, 330, 330, 330, 330, 350, 100, 180, 100, 100, 180, 100, 0, 0, 0, 0]
ratio_list = [0.5,1,0.5,1,0.5, 0.5,0.5,1,0.5,2, 2,2,2,0.5,2, 2,0.5,0.5,1,0.5,0.5]
# ------------------------------------------------------------------
no_domains = len(x)
dx = [5]*no_domains
dy = dx

# initialise all subdomains without applying ratio
sd = fl.init_subdomains('linear', no_domains, nx, ny, dx, dy, x, y, ratio_list, epsilon_eta, dt)

# Find neighbors
fl.find_neighbours(sd)

# Set physical boundaries
fl.set_physical_boundaries(sd, global_nx, global_ny)

# resize al sd variables based on ration
fl.resize_subdomains(sd, ratio_list)

def create_halos(sd):
    for ele in range(no_domains):
        # Get the size of the block along the x and y axes
        ny, nx = sd[ele].values_u.shape[2], sd[ele].values_u.shape[3]

        # Create a list where each element is a tensor corresponding to a face of the block
        sd[ele].halo_u = [torch.zeros((nx,)),  # Bottom face
                        torch.zeros((ny,)),  # Left face
                        torch.zeros((ny,)),  # Right face
                        torch.zeros((nx,))]  # Top face

        sd[ele].halo_v = [torch.zeros((nx,)),  # Bottom face
                        torch.zeros((ny,)),  # Left face
                        torch.zeros((ny,)),  # Right face
                        torch.zeros((nx,))]  # Top face

        sd[ele].halo_b_u = [torch.zeros((nx,)),  # Bottom face
                            torch.zeros((ny,)),  # Left face
                            torch.zeros((ny,)),  # Right face
                            torch.zeros((nx,))]  # Top face

        sd[ele].halo_b_v = [torch.zeros((nx,)),  # Bottom face
                            torch.zeros((ny,)),  # Left face
                            torch.zeros((ny,)),  # Right face
                            torch.zeros((nx,))]  # Top face

        sd[ele].halo_h = [torch.zeros((nx,)),  # Bottom face
                        torch.zeros((ny,)),  # Left face
                        torch.zeros((ny,)),  # Right face
                        torch.zeros((nx,))]  # Top face

        sd[ele].halo_hh = [torch.zeros((nx,)),  # Bottom face
                        torch.zeros((ny,)),  # Left face
                        torch.zeros((ny,)),  # Right face
                        torch.zeros((nx,))]  # Top face

        sd[ele].halo_dif_h = [torch.zeros((nx,)),  # Bottom face
                            torch.zeros((ny,)),  # Left face
                            torch.zeros((ny,)),  # Right face
                            torch.zeros((nx,))]  # Top face

        sd[ele].halo_eta = [torch.zeros((nx,)),  # Bottom face
                            torch.zeros((ny,)),  # Left face
                            torch.zeros((ny,)),  # Right face
                            torch.zeros((nx,))]  # Top face

        sd[ele].halo_eta1 = [torch.zeros((nx,)),  # Bottom face
                            torch.zeros((ny,)),  # Left face
                            torch.zeros((ny,)),  # Right face
                            torch.zeros((nx,))]  # Top face

    return

# Create halos for each block :: sd[ele].halo_u[iface]
create_halos(sd)

# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0], device=device)
# Isotropic Laplacian
w1 = torch.tensor([[[[1/3], [ 1/3] , [1/3]],
                    [[1/3], [-8/3] , [1/3]],
                    [[1/3], [ 1/3] , [1/3]]]])
# Gradient in x
w2 = torch.tensor([[[[1/12], [0.0], [-1/12]],
                    [[1/3] , [0.0], [-1/3]] ,
                    [[1/12], [0.0], [-1/12]]]])
# Gradient in y
w3 = torch.tensor([[[[-1/12], [-1/3], [-1/12]],
                    [[0.0]  , [0.0] , [0.0]]  ,
                    [[1/12] , [1/3] , [1/12]]]])
# Consistant mass matrix
wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                    [[0.11] , [0.44] , [0.11]],
                    [[0.028], [0.11] , [0.028]]]])
w1 = torch.reshape(w1,(1,1,3,3))
w2 = torch.reshape(w2,(1,1,3,3))
w3 = torch.reshape(w3,(1,1,3,3))
wm = torch.reshape(wm,(1,1,3,3))

path = './Documents/'
# # # ################################### # # #
# # # #######   Initialisation ########## # # #
# # # ################################### # # #
# Open the .raw file for reading
with open(f'{path}carlisle-5m.dem.raw', 'r') as file:
    # Read the entire content of the file and split it into individual values
    data = file.read().split()

# Convert the string values to floats
mesh = torch.tensor([float(value) for value in data[12:]])

# Now, float_values contains a list of floating-point numbers from the .raw file

mesh = mesh.reshape(int(data[3]),int(data[1]))
# make the size even numbers (left, right, top, bottom) padding
mesh = F.pad(mesh, (0,1,0,1), mode='constant', value=0)
mesh[:,-1] = mesh[:,-2]
mesh[-1,:] = mesh[-2,:]


# Plot the mesh
fig, ax = plt.subplots(figsize=(9.52,6.11))

plt.imshow(mesh)

# Draw the boundaries of the subdomains
for ele in sd:
    r = 1# ele.ratio
    x_coords = [ele.x0/r, ele.x1/r, ele.x3/r, ele.x2, ele.x0/r]
    y_coords = [ele.y0/r, ele.y1/r, ele.y3/r, ele.y2, ele.y0/r]
    ax.plot(x_coords, y_coords, marker='o', label=f'Subdomain {ele.index}')
    ax.text((ele.x0 + ele.x1) / 2/r, (ele.y0/r + ele.y2) / 2, f'{ele.index}', fontsize=18, ha='center', color='white')

# Splits the mesh into the tiles with the same size of the subdomains          
tiles = fl.extract_subdomains(mesh,x, y, nx, ny, ratio_list)

# copy tiles into sd.values_H
for i in range(no_domains):
    sd[i].values_H[0,0,:,:] = tiles[i]

for ele in range(no_domains):
    sd[ele].values_h = sd[ele].values_H
    sd[ele].values_hp[0,0,1:-1,1:-1] = sd[ele].values_H

'''
It returns the coordinates of the sources based  on the original size of the domain
'''
x_origin = 338500 ; y_origin = 554700

df = pd.read_csv(f'{path}carlisle.bci', delim_whitespace=True)

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


plt.scatter(x_upstream1, y_upstream1, label='upstream 1', color='red', marker='o')
plt.scatter(x_upstream2, y_upstream2, label='upstream 2', color='blue', marker='x')
plt.scatter(x_upstream3, y_upstream3, label='upstream 3', color='green', marker='s')
plt.legend()
plt.xlim(0,len(mesh[0,:]))


# Here the y coordinates must be reversed as the origin is at the top left of the domain not bottom left
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

# update the source coordinates based on the x2 of each subdomain
x_upstream1 = list(map(lambda i: int((x_upstream1[i] - sd[source_1_sd_idx[i]].x2) ), range(len(x_upstream1))))
x_upstream2 = list(map(lambda i: int((x_upstream2[i] - sd[source_2_sd_idx[i]].x2) ), range(len(x_upstream2))))
x_upstream3 = list(map(lambda i: int((x_upstream3[i] - sd[source_3_sd_idx[i]].x2) ), range(len(x_upstream3))))

y_upstream1 = list(map(lambda i: int((y_upstream1[i] - sd[source_1_sd_idx[i]].y2) ), range(len(y_upstream1))))
y_upstream2 = list(map(lambda i: int((y_upstream2[i] - sd[source_2_sd_idx[i]].y2-1) ), range(len(y_upstream2))))
y_upstream3 = list(map(lambda i: int((y_upstream3[i] - sd[source_3_sd_idx[i]].y2-1) ), range(len(y_upstream3))))

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


df = pd.read_csv(f'{path}flowrates.csv', delim_whitespace=True)
rate1 = df['upstream1']/5
rate2 = df['upstream2']/5
rate3 = df['upstream3']/5


def get_source(time):
    '''
    sets the source based on variable time compatible with the real data
    '''
    global rate1, rate2, rate3
    
    indx = time // 900 # 900 is the time interval in the given time series
    
    for i, s_idx in enumerate(source_1_sd_idx):
        sd[s_idx].source_h[0,0,y_upstream1[i],x_upstream1[i]]     = ((rate1[indx+1] - rate1[indx])/900 * (time%900) + rate1[indx])/(4**(source1_ratio[i]-1))
    
    for i, s_idx in enumerate(source_2_sd_idx):
        sd[s_idx].source_h[0,0,y_upstream2[i]-11,x_upstream2[i]]     = ((rate2[indx+1] - rate2[indx])/900 * (time%900) + rate2[indx])/(4**(source2_ratio[i]-1))
        
    for i, s_idx in enumerate(source_3_sd_idx):
        sd[s_idx].source_h[0,0,y_upstream3[i]-4,x_upstream3[i]]     = ((rate3[indx+1] - rate3[indx])/900 * (time%900) + rate3[indx])/(4**(source3_ratio[i]-1))
        
#     sd[3].source_h[0,0,:,1] = rate2[index]
#     sd[1].source_h[0,0,:,1] = rate2[index]

for ele in range(no_domains):
    sd[ele].values_H = -sd[ele].values_H                


# # # ################################### # # #
# # # #########   AI4SWE MAIN ########### # # #
# # # ################################### # # #
class AI4SWE(nn.Module):
    """docstring for two_step"""
    def __init__(self):
        super(AI4SWE, self).__init__()
        self.cmm = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm.weight.data = wm
        
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1 / dx[0]**2
        self.xadv.weight.data = w2 / dx[0]
        self.yadv.weight.data = w3 / dx[0]

        self.diag = -w1[0, 0, 1, 1].item() / (dx[0])**2
            
        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer


    def scale_it(self, val1, val2):
        '''
        val1 :: info comming from
        val2 :: info going to
        ratio = val1/val2 size
        '''
        size_val2 = val2.size(0)
        ratio = val1.size(0) / size_val2

        if ratio == 1.0:
            return val1
        elif ratio == 0.5:
            # Use repeat_interleave to increase the size of val1 to match the size of val2
            repeat_factor = size_val2 // val1.size(0)
            val2 = val1.repeat_interleave(repeat_factor)
            return val2
        elif ratio == 2:
            # Calculate the number of elements in each group
            group_size = val1.size(0) // size_val2
            # Reshape val1 into a 2D tensor with size_val2 rows
            val1_reshaped = val1.view(size_val2, -1)
            # Calculate the mean of each row
            val2 = val1_reshaped.mean(dim=1)
            return val2
        elif ratio == 0.25:
            # Repeat each element of val1 4 times
            val2 = val1.repeat_interleave(4)
            return val2
        elif ratio == 4:
            # Reshape val1 into a 2D tensor with size_val2 rows and 4 columns
            val1_reshaped = val1.view(size_val2, -1)
            # Calculate the mean of each row
            val2 = val1_reshaped.mean(dim=1)
            return val2


        
    def update_halos(self, sd):  # AMIN:: to optimise pass in as local vbls instead of global
        for ele in range(no_domains):
            for key, val in sd[ele].neig.items():
                for i in val:
                    if not isinstance(i, str):
                        my_x1 = sd[ele].shared_indices[key][i]['my_start']
                        my_x2 = sd[ele].shared_indices[key][i]['my_end']
                        neig_x1 =  sd[ele].shared_indices[key][i]['neig_start']
                        neig_x2 =  sd[ele].shared_indices[key][i]['neig_end']

                        if key == 'bottom':
                            sd[ele].halo_u[0][my_x1:my_x2]     = self.scale_it(sd[i].values_u[0,0,0              , neig_x1:neig_x2], sd[ele].halo_u[0][my_x1:my_x2])
                            sd[ele].halo_v[0][my_x1:my_x2]     = self.scale_it(sd[i].values_v[0,0,0              , neig_x1:neig_x2], sd[ele].halo_v[0][my_x1:my_x2])
                            sd[ele].halo_b_u[0][my_x1:my_x2]   = self.scale_it(sd[i].b_u[0,0,0                   , neig_x1:neig_x2], sd[ele].halo_b_u[0][my_x1:my_x2])
                            sd[ele].halo_b_v[0][my_x1:my_x2]   = self.scale_it(sd[i].b_v[0,0,0                   , neig_x1:neig_x2], sd[ele].halo_b_v[0][my_x1:my_x2])
                            sd[ele].halo_h[0][my_x1:my_x2]     = self.scale_it(sd[i].values_h[0,0,0              , neig_x1:neig_x2], sd[ele].halo_h[0][my_x1:my_x2])
                            sd[ele].halo_hh[0][my_x1:my_x2]    = self.scale_it(sd[i].values_hh[0,0,0             , neig_x1:neig_x2], sd[ele].halo_hh[0][my_x1:my_x2])
                            sd[ele].halo_dif_h[0][my_x1:my_x2] = self.scale_it(sd[i].dif_values_h[0,0,0          , neig_x1:neig_x2], sd[ele].halo_dif_h[0][my_x1:my_x2])
                            sd[ele].halo_eta[0][my_x1:my_x2]   = self.scale_it((sd[i].values_h+sd[i].values_H)[0,0,0, neig_x1:neig_x2], sd[ele].halo_eta[0][my_x1:my_x2])
                            sd[ele].halo_eta1[0][my_x1:my_x2]  = self.scale_it(sd[i].eta1[0,0,0, neig_x1:neig_x2], sd[ele].halo_eta1[0][my_x1:my_x2])
                            sd[ele].halo_k_uu[0][my_x1:my_x2]  = self.scale_it(torch.minimum(sd[i].k_u[0,0,0,neig_x1:neig_x2]  , sd[i].k3[0,neig_x1:neig_x2]), sd[ele].halo_k_uu[0][my_x1:my_x2])
                            sd[ele].halo_k_vv[0][my_x1:my_x2]  = self.scale_it(torch.minimum(sd[i].k_v[0,0,0,neig_x1:neig_x2]  , sd[i].k3[0,neig_x1:neig_x2]), sd[ele].halo_k_vv[0][my_x1:my_x2])
                        if key == 'left':
                            sd[ele].halo_u[1][my_x1:my_x2]     = self.scale_it(sd[i].values_u[0,0                , neig_x1:neig_x2, -1], sd[ele].halo_u[1][my_x1:my_x2])
                            sd[ele].halo_v[1][my_x1:my_x2]     = self.scale_it(sd[i].values_v[0,0                , neig_x1:neig_x2, -1], sd[ele].halo_v[1][my_x1:my_x2])
                            sd[ele].halo_b_u[1][my_x1:my_x2]   = self.scale_it(sd[i].b_u[0,0                     , neig_x1:neig_x2, -1], sd[ele].halo_b_u[1][my_x1:my_x2])
                            sd[ele].halo_b_v[1][my_x1:my_x2]   = self.scale_it(sd[i].b_v[0,0                     , neig_x1:neig_x2, -1], sd[ele].halo_b_v[1][my_x1:my_x2])
                            sd[ele].halo_h[1][my_x1:my_x2]     = self.scale_it(sd[i].values_h[0,0                , neig_x1:neig_x2, -1], sd[ele].halo_h[1][my_x1:my_x2])
                            sd[ele].halo_hh[1][my_x1:my_x2]    = self.scale_it(sd[i].values_hh[0,0               , neig_x1:neig_x2, -1], sd[ele].halo_hh[1][my_x1:my_x2])
                            sd[ele].halo_dif_h[1][my_x1:my_x2] = self.scale_it(sd[i].dif_values_h[0,0            , neig_x1:neig_x2, -1], sd[ele].halo_dif_h[1][my_x1:my_x2])
                            sd[ele].halo_eta[1][my_x1:my_x2]   = self.scale_it((sd[i].values_h+sd[i].values_H)[0,0, neig_x1:neig_x2,-1], sd[ele].halo_eta[1][my_x1:my_x2])
                            sd[ele].halo_eta1[1][my_x1:my_x2]  = self.scale_it(sd[i].eta1[0,0                    , neig_x1:neig_x2, -1], sd[ele].halo_eta1[1][my_x1:my_x2])
                            sd[ele].halo_k_uu[1][my_x1:my_x2]  = self.scale_it(torch.minimum(sd[i].k_u[0,0,neig_x1:neig_x2,-1] , sd[i].k3[neig_x1:neig_x2,-1]), sd[ele].halo_k_uu[1][my_x1:my_x2])
                            sd[ele].halo_k_vv[1][my_x1:my_x2]  = self.scale_it(torch.minimum(sd[i].k_v[0,0,neig_x1:neig_x2,-1] , sd[i].k3[neig_x1:neig_x2,-1]), sd[ele].halo_k_vv[1][my_x1:my_x2])
                        if key == 'right':
                            sd[ele].halo_u[2][my_x1:my_x2]     = self.scale_it(sd[i].values_u[0,0                , neig_x1:neig_x2, 0], sd[ele].halo_u[2][my_x1:my_x2])
                            sd[ele].halo_v[2][my_x1:my_x2]     = self.scale_it(sd[i].values_v[0,0                , neig_x1:neig_x2, 0], sd[ele].halo_v[2][my_x1:my_x2])
                            sd[ele].halo_b_u[2][my_x1:my_x2]   = self.scale_it(sd[i].b_u[0,0                     , neig_x1:neig_x2, 0], sd[ele].halo_b_u[2][my_x1:my_x2])
                            sd[ele].halo_b_v[2][my_x1:my_x2]   = self.scale_it(sd[i].b_v[0,0                     , neig_x1:neig_x2, 0], sd[ele].halo_b_v[2][my_x1:my_x2])
                            sd[ele].halo_h[2][my_x1:my_x2]     = self.scale_it(sd[i].values_h[0,0                , neig_x1:neig_x2, 0], sd[ele].halo_h[2][my_x1:my_x2])
                            sd[ele].halo_hh[2][my_x1:my_x2]    = self.scale_it(sd[i].values_hh[0,0               , neig_x1:neig_x2, 0], sd[ele].halo_hh[2][my_x1:my_x2])
                            sd[ele].halo_dif_h[2][my_x1:my_x2] = self.scale_it(sd[i].dif_values_h[0,0            , neig_x1:neig_x2, 0], sd[ele].halo_dif_h[2][my_x1:my_x2])
                            sd[ele].halo_eta[2][my_x1:my_x2]   = self.scale_it((sd[i].values_h+sd[i].values_H)[0,0, neig_x1:neig_x2,0], sd[ele].halo_eta[2][my_x1:my_x2])
                            sd[ele].halo_eta1[2][my_x1:my_x2]  = self.scale_it(sd[i].eta1[0,0                    , neig_x1:neig_x2, 0], sd[ele].halo_eta1[2][my_x1:my_x2])
                            sd[ele].halo_k_uu[2][my_x1:my_x2]  = self.scale_it(torch.minimum(sd[i].k_u[0,0,neig_x1:neig_x2,0]  , sd[i].k3[neig_x1:neig_x2,0]), sd[ele].halo_k_uu[2][my_x1:my_x2])
                            sd[ele].halo_k_vv[2][my_x1:my_x2]  = self.scale_it(torch.minimum(sd[i].k_v[0,0,neig_x1:neig_x2,0]  , sd[i].k3[neig_x1:neig_x2,0]), sd[ele].halo_k_vv[2][my_x1:my_x2])
                        if key == 'top':
                            sd[ele].halo_u[3][my_x1:my_x2]     = self.scale_it(sd[i].values_u[0,0,-1             , neig_x1:neig_x2], sd[ele].halo_u[3][my_x1:my_x2])
                            sd[ele].halo_v[3][my_x1:my_x2]     = self.scale_it(sd[i].values_v[0,0,-1             , neig_x1:neig_x2], sd[ele].halo_v[3][my_x1:my_x2])
                            sd[ele].halo_b_u[3][my_x1:my_x2]   = self.scale_it(sd[i].b_u[0,0,-1                  , neig_x1:neig_x2], sd[ele].halo_b_u[3][my_x1:my_x2])
                            sd[ele].halo_b_v[3][my_x1:my_x2]   = self.scale_it(sd[i].b_v[0,0,-1                  , neig_x1:neig_x2], sd[ele].halo_b_v[3][my_x1:my_x2])
                            sd[ele].halo_h[3][my_x1:my_x2]     = self.scale_it(sd[i].values_h[0,0,-1             , neig_x1:neig_x2], sd[ele].halo_h[3][my_x1:my_x2])
                            sd[ele].halo_hh[3][my_x1:my_x2]    = self.scale_it(sd[i].values_hh[0,0,-1            , neig_x1:neig_x2], sd[ele].halo_hh[3][my_x1:my_x2])
                            sd[ele].halo_dif_h[3][my_x1:my_x2] = self.scale_it(sd[i].dif_values_h[0,0,-1         , neig_x1:neig_x2], sd[ele].halo_dif_h[3][my_x1:my_x2])
                            sd[ele].halo_eta[3][my_x1:my_x2]   = self.scale_it((sd[i].values_h+sd[i].values_H)[0,0,-1, neig_x1:neig_x2], sd[ele].halo_eta[3][my_x1:my_x2])
                            sd[ele].halo_eta1[3][my_x1:my_x2]  = self.scale_it(sd[i].eta1[0,0,-1, neig_x1:neig_x2], sd[ele].halo_eta1[3][my_x1:my_x2])
                            sd[ele].halo_k_uu[3][my_x1:my_x2]  = self.scale_it(torch.minimum(sd[i].k_u[0,0,-1,neig_x1:neig_x2] , sd[i].k3[-1,neig_x1:neig_x2]), sd[ele].halo_k_uu[3][my_x1:my_x2])
                            sd[ele].halo_k_vv[3][my_x1:my_x2]  = self.scale_it(torch.minimum(sd[i].k_v[0,0,-1,neig_x1:neig_x2] , sd[i].k3[-1,neig_x1:neig_x2]), sd[ele].halo_k_vv[3][my_x1:my_x2])
        return
                
            
        
    def boundary_condition_u(self, values_u, values_uu, sd):
        values_uu[0,0,1:-1,1:-1] = values_u[0,0, :, :]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig['bottom'], int):
            values_uu[0,0, -1, :] = values_uu[0,0, -2, :]
        else:
            values_uu[0,0, -1, 1:-1] = sd.halo_u[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig['left'], int):
            values_uu[0,0,    :, 0].fill_(0) 
        else:
            values_uu[0,0, 1:-1, 0] = sd.halo_u[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig['right'], int):
            values_uu[0,0,    :, -1].fill_(0)
        else:
            values_uu[0,0, 1:-1, -1] = sd.halo_u[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig['top'], int):
            values_uu[0,0, 0, :] = values_uu[0,0, 1, :]
        else:
            values_uu[0,0, 0, 1:-1] = sd.halo_u[3]
            
        return


    def boundary_condition_v(self, values_v, values_vv, sd):
        values_vv[0,0,1:-1,1:-1] = values_v[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig['bottom'], int):
            values_vv[0,0, -1,    :].fill_(0)
        else:
            values_vv[0,0, -1, 1:-1] = sd.halo_v[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig['left'], int):
            values_vv[0,0,      :,   0] = values_vv[0,0, :, 1]
        else:
            values_vv[0,0,   1:-1,   0] = sd.halo_v[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig['right'], int):   
            values_vv[0,0,      :, -1] = values_vv[0,0, :,-2] 
        else:
            values_vv[0,0,   1:-1, -1] = sd.halo_v[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig['top'], int):
            values_vv[0,0, 0,    :].fill_(0)
        else:
            values_vv[0,0, 0, 1:-1] = sd.halo_v[3]

        return
    
    
    def boundary_condition_b_u(self, b_u, b_uu, sd):
        b_uu[0,0,1:-1,1:-1] = b_u[0,0, :, :]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig['bottom'], int):
            b_uu[0,0, -1,    :] = b_uu[0,0,-2, :]
        else:
            b_uu[0,0, -1, 1:-1] = sd.halo_b_u[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig['left'], int):
            b_uu[0,0,   :, 0].fill_(0)
        else:
            b_uu[0,0,1:-1, 0] = sd.halo_b_u[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig['right'], int):
            b_uu[0,0,    :, -1].fill_(0) 
        else:
            b_uu[0,0, 1:-1, -1] = sd.halo_b_u[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig['top'], int):
            b_uu[0,0, 0,    :] = b_uu[0,0, 1, :]
        else:
            b_uu[0,0, 0, 1:-1] = sd.halo_b_u[3]

        return


    def boundary_condition_b_v(self, b_v, b_vv, sd):
        b_vv[0,0,1:-1,1:-1] = b_v[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig['bottom'], int):
            b_vv[0,0, -1,    :].fill_(0)
        else:
            b_vv[0,0, -1, 1:-1] = sd.halo_b_v[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig['left'], int):
            b_vv[0,0,      :,   0] = b_vv[0,0, :, 1]
        else:
            b_vv[0,0,   1:-1,   0] = sd.halo_b_v[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig['right'], int):   
            b_vv[0,0,      :, -1] = b_vv[0,0, :,-2]
        else:
            b_vv[0,0,   1:-1, -1] = sd.halo_b_v[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig['top'], int):
            b_vv[0,0, 0,    :].fill_(0)         # top
        else:
            b_vv[0,0, 0, 1:-1] = sd.halo_b_v[3]

        return


    def boundary_condition_h(self, values_h, values_hp, sd): # Amin:: delete nny etc instead use -1 # use tensor.add()
        values_hp[0,0,1:-1,1:-1] = values_h[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig['bottom'], int):
            values_hp[0,0,-1,:].fill_(0)
        else:
            values_hp[0,0,-1,1:-1] = sd.halo_h[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig['left'], int):
            values_hp[0,0,   :,0].fill_(0)
        else:
            values_hp[0,0,1:-1,0] = sd.halo_h[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig['right'], int):
            values_hp[0,0,   :,-1].fill_(0)
        else:
            values_hp[0,0,1:-1,-1] = sd.halo_h[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig['top'], int):
            values_hp[0,0,0,    :].fill_(0)
        else:
            values_hp[0,0,0, 1:-1] = sd.halo_h[3]
                
        return values_hp
        
    
    def boundary_condition_hh(self, values_hh, values_hhp, sd): # Amin:: delete nny etc instead use -1 # use tensor.add()
        values_hhp[0,0,1:-1,1:-1] = (values_hh)[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig['bottom'], int):
            values_hhp[0,0,-1,:].fill_(0)
        else:
            values_hhp[0,0,-1, 1:-1] = sd.halo_hh[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig['left'], int):
            values_hhp[0,0,   :,0].fill_(0)
        else:
            values_hhp[0,0,1:-1,0] = sd.halo_hh[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig['right'], int): # on the physical boundary
            values_hhp[0,0,   :,-1].fill_(0)
        else:
            values_hhp[0,0,1:-1,-1] = sd.halo_hh[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig['top'], int):
            values_hhp[0,0,0,:].fill_(0)
        else:
            values_hhp[0,0,0,1:-1] = sd.halo_hh[3]
            
        return values_hhp
    
    
    def boundary_condition_dif_h(self, dif_values_h, dif_values_hh, sd):
        dif_values_hh[0,0,1:-1,1:-1] = (dif_values_h)[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig['bottom'], int):
            dif_values_hh[0,0,-1,:].fill_(0)
        else:
            dif_values_hh[0,0,-1,1:-1] = sd.halo_dif_h[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig['left'], int):
            dif_values_hh[0,0,   :,0].fill_(0)
        else:
            dif_values_hh[0,0,1:-1,0] = sd.halo_dif_h[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig['right'], int): # on the physical boundary
            dif_values_hh[0,0,   :,-1].fill_(0)
        else:
            dif_values_hh[0,0,1:-1,-1] = sd.halo_dif_h[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig['top'], int):
            dif_values_hh[0,0,0,:].fill_(0)
        else:
            dif_values_hh[0,0,0,1:-1] = sd.halo_dif_h[3]
        
            
        return dif_values_hh
                                                     
        
    def boundary_condition_eta(self, eta, values_hp, sd):
        values_hp[0,0,1:-1,1:-1] = eta[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig['bottom'], int):
            values_hp[0,0,-1,    :].fill_(0)
        else:
            values_hp[0,0,-1, 1:-1] = sd.halo_eta[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig['left'], int):
            values_hp[0,0,    :,   0].fill_(0)             
        else:
            values_hp[0,0, 1:-1,   0] = sd.halo_eta[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig['right'], int):
            values_hp[0,0,    :, -1].fill_(0)              
        else:
            values_hp[0,0, 1:-1, -1] = sd.halo_eta[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig['top'], int):
            values_hp[0,0,0,    :].fill_(0)
        else:
            values_hp[0,0,0, 1:-1] = sd.halo_eta[3]

        return values_hp

    
    def boundary_condition_eta1(self, eta1, eat1_p, sd): # Amin:: delete nny etc instead use -1 # use tensor.add()# check is dimension applys inside the ()
        eat1_p[0,0,1:-1,1:-1] = eta1[0,0,:,:]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig['bottom'], int):
            eat1_p[0,0,-1,    :].fill_(0)
        else:
            eat1_p[0,0,-1, 1:-1] = sd.halo_eta1[0]
        # --------------------------------------------------------------------------------- left
        if isinstance(sd.neig['left'], int): # on the physical boundary
            eat1_p[0,0,    :,0].fill_(0)
        else:
            eat1_p[0,0, 1:-1,0] = sd.halo_eta1[1]
        # --------------------------------------------------------------------------------- right
        if isinstance(sd.neig['right'], int): # on the physical boundary
            eat1_p[0,0,   :,-1].fill_(0)
        else:
            eat1_p[0,0,1:-1,-1] = sd.halo_eta1[2]
        # --------------------------------------------------------------------------------- top
        if isinstance(sd.neig['top'], int):
            eat1_p[0,0,0,    :].fill_(0)
        else:
            eat1_p[0,0,0, 1:-1] = sd.halo_eta1[3]
        
        return eat1_p
    
        
    def update_halo_k_uu(self, sd):
        # --------------------------------------------------------------------------------- bottom
        neig0 = sd.neig['bottom']
        if not isinstance(neig0, int):
            sd.k_uu[0,0, -1, 1:-1] = sd.halo_k_uu[0]
        # --------------------------------------------------------------------------------- left
        neig1 = sd.neig['left']
        if not isinstance(neig1, int): # has neighbour
            sd.k_uu[0,0, 1:-1, 0] = sd.halo_k_uu[1]
        # --------------------------------------------------------------------------------- right
        neig2 = sd.neig['right']
        if not isinstance(neig2, int):
            sd.k_uu[0,0, 1:-1, -1] = sd.halo_k_uu[2]
        # --------------------------------------------------------------------------------- top
        neig3 = sd.neig['top']
        if not isinstance(neig3, int):
            sd.k_uu[0,0, 0, 1:-1] = sd.halo_k_uu[3]
        return
 

    def update_halo_k_vv(self, sd):
        # --------------------------------------------------------------------------------- bottom
        neig0 = sd.neig['bottom']
        if not isinstance(neig0, int):
            sd.k_vv[0,0, -1, 1:-1] = sd.halo_k_vv[0]
        # --------------------------------------------------------------------------------- left
        neig1 = sd.neig['left']
        if not isinstance(neig1, int): # has neighbour
            sd.k_vv[0,0,1:-1,0] = sd.halo_k_vv[1]
        # --------------------------------------------------------------------------------- right
        neig2 = sd.neig['right']
        if not isinstance(neig2, int): # has neighbour
            sd.k_vv[0,0,1:-1,-1] = sd.halo_k_vv[2]
        # --------------------------------------------------------------------------------- top
        neig3 = sd.neig['top']
        if not isinstance(neig3, int):
            sd.k_vv[0,0, 0, 1:-1] = sd.halo_k_vv[3]
        return
    
    
    def PG_vector(self, values_uu, values_vv, values_u, values_v, k3, dx, sd):
        sd.k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_uu)) / \
            (1e-03  + (torch.abs(self.xadv(values_uu)) * (dx**-2) + torch.abs(self.yadv(values_uu)) * (dx**-2)) / 2)

        sd.k_v = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(values_vv)) / \
            (1e-03  + (torch.abs(self.xadv(values_vv)) * (dx**-2) + torch.abs(self.yadv(values_vv)) * (dx**-2)) / 2)

        sd.k_uu = F.pad(torch.minimum(sd.k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        sd.k_vv = F.pad(torch.minimum(sd.k_v, k3) , (1, 1, 1, 1), mode='constant', value=0)
        
        self.update_halo_k_uu(sd)
        self.update_halo_k_vv(sd)
        
        sd.k_x = 0.5 * (sd.k_u * self.diff(values_uu) + self.diff(values_uu * sd.k_uu) - values_u * self.diff(sd.k_uu))
        sd.k_y = 0.5 * (sd.k_v * self.diff(values_vv) + self.diff(values_vv * sd.k_vv) - values_v * self.diff(sd.k_vv))
        return


    def PG_scalar(self, sd, eta1_p, eta1, values_u, values_v, k3, dx):
        sd.k_u = 0.25 * dx * torch.abs(1/2 * (dx**-2) * (torch.abs(values_u) * dx + torch.abs(values_v) * dx) * self.diff(eta1_p)) / \
            (1e-03 + (torch.abs(self.xadv(eta1_p)) * (dx**-2) + torch.abs(self.yadv(eta1_p)) * (dx**-2)) / 2)
        sd.k_uu = F.pad(torch.minimum(sd.k_u, k3) , (1, 1, 1, 1), mode='constant', value=0)
        self.update_halo_k_uu(sd)
        return 0.5 * (sd.k_u * self.diff(eta1_p) + self.diff(eta1_p * sd.k_uu) - eta1 * self.diff(sd.k_uu))


    def forward(self, sd, dt, rho):
        self.update_halos(sd)
        for ele in range(no_domains):
            self.boundary_condition_u(sd[ele].values_u,sd[ele].values_uu, sd[ele])
            self.boundary_condition_v(sd[ele].values_v,sd[ele].values_vv, sd[ele])
    # -------------------------------------------------------------------------------------------------------------------
            self.PG_vector(sd[ele].values_uu, sd[ele].values_vv, sd[ele].values_u, sd[ele].values_v, sd[ele].k3, sd[ele].dx, sd[ele])
            # ===================================================================================================================================== 1-step velocity solver
            sd[ele].values_u = sd[ele].values_u + sd[ele].k_x * dt - sd[ele].values_u * self.xadv(sd[ele].values_uu) * dt - sd[ele].values_v * self.yadv(sd[ele].values_uu) * dt
            sd[ele].values_v = sd[ele].values_v + sd[ele].k_y * dt - sd[ele].values_u * self.xadv(sd[ele].values_vv) * dt - sd[ele].values_v * self.yadv(sd[ele].values_vv) * dt
            # ============================================================================================================================================================
            sd[ele].values_u = sd[ele].values_u - self.xadv(self.boundary_condition_h(sd[ele].values_h,sd[ele].values_hp, sd[ele])) * dt
            sd[ele].values_v = sd[ele].values_v - self.yadv(self.boundary_condition_h(sd[ele].values_h,sd[ele].values_hp, sd[ele])) * dt

            sd[ele].sigma_q = torch.pow(torch.pow(sd[ele].values_u,2) + torch.pow(sd[ele].values_v,2),0.5) * 0.055**2 / (torch.maximum( sd[ele].k1,
               sd[ele].dx*self.cmm(self.boundary_condition_eta(sd[ele].values_H+sd[ele].values_h,sd[ele].values_hp, sd[ele]))*0.01+(sd[ele].values_H+sd[ele].values_h)*0.99 )**(4/3))

            sd[ele].values_u = sd[ele].values_u / (1 + sd[ele].sigma_q * dt / rho)
            sd[ele].values_v = sd[ele].values_v / (1 + sd[ele].sigma_q * dt / rho)

    # -------------------------------------------------------------------------------------------------------------------
            self.boundary_condition_u(sd[ele].values_u,sd[ele].values_uu, sd[ele])
            self.boundary_condition_v(sd[ele].values_v,sd[ele].values_vv, sd[ele])
            sd[ele].eta1 = torch.maximum(sd[ele].k2,(sd[ele].values_H+sd[ele].values_h))
            sd[ele].eta2 = torch.maximum(sd[ele].k1,(sd[ele].values_H+sd[ele].values_h))

    # -------------------------------------------------------------------------------------------------------------------
            sd[ele].b = beta * rho * (-self.xadv(self.boundary_condition_eta1(sd[ele].eta1,sd[ele].eta1_p, sd[ele])) * sd[ele].values_u - \
                               self.yadv(self.boundary_condition_eta1(sd[ele].eta1,sd[ele].eta1_p, sd[ele])) * sd[ele].values_v - \
                               sd[ele].eta1 * self.xadv(sd[ele].values_uu) - sd[ele].eta1 * self.yadv(sd[ele].values_vv) + \
                               self.PG_scalar(sd[ele], self.boundary_condition_eta1(sd[ele].eta1,sd[ele].eta1_p, sd[ele]), sd[ele].eta1, sd[ele].values_u, sd[ele].values_v, sd[ele].k3, sd[ele].dx) - \
                               self.cmm(self.boundary_condition_dif_h(sd[ele].dif_values_h,sd[ele].dif_values_hh, sd[ele])) / dt + sd[ele].source_h) / (dt * sd[ele].eta2)
            sd[ele].values_h_old = sd[ele].values_h.clone()

    # -------------------------------------------------------------------------------------------------------------------
            for i in range(2):
                sd[ele].values_hh = sd[ele].values_hh - (-self.diff(self.boundary_condition_hh(sd[ele].values_hh,sd[ele].values_hhp, sd[ele])) + beta * rho / (dt**2 * sd[ele].eta2) * sd[ele].values_hh) / \
                            (self.diag + beta * rho / (dt**2 * sd[ele].eta2)) + sd[ele].b / (self.diag + beta * rho / (dt**2 * sd[ele].eta2))
            sd[ele].values_h = sd[ele].values_h + sd[ele].values_hh
            sd[ele].dif_values_h = sd[ele].values_h - sd[ele].values_h_old

    # -------------------------------------------------------------------------------------------------------------------
            sd[ele].values_u = sd[ele].values_u - self.xadv(self.boundary_condition_hh(sd[ele].values_hh,sd[ele].values_hhp, sd[ele])) * dt / rho
            sd[ele].values_v = sd[ele].values_v - self.yadv(self.boundary_condition_hh(sd[ele].values_hh,sd[ele].values_hhp, sd[ele])) * dt / rho
            
        return sd[0].values_hh#, sd[1].values_hh,sd[2].values_hh,sd[3].values_hh

model = AI4SWE().to(device)
# model = torch.compile(AI4SWE().to(device))
for ele in range(no_domains):
    sd[ele].to(device)

start = time.time()
istep=0

image_folder = '/home/an619/Desktop/git/AI/RMS/semi/2D/Linear_results/images/'
def plot_subdomains(no_domains, no_domains_x, no_domains_y, sd, img):
    # Calculate the size of each subplot
    subplot_size_x = 3.11  # You can adjust this value as needed
    subplot_size_y = 2  # You can adjust this value as needed

    # Calculate the total figure size
    fig_size_x = subplot_size_x * no_domains_x
    fig_size_y = subplot_size_y * no_domains_y
    
    fig, axs = plt.subplots(no_domains_y, no_domains_x, figsize=(2*fig_size_x, 2*fig_size_y))

    
    for i in range(no_domains):
        row = no_domains_y - 1 - i // no_domains_x
        col = i % no_domains_x
        axs[row, col].imshow(sd[i].values_h[0,0,:,:].cpu()+sd[i].values_H[0,0,:,:].cpu(),vmin=0, vmax=8 )#, cmap='Blues')
        axs[row, col].axis('off')  # Set axis off
    

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)  # Remove space between subplots and padding
    plt.savefig(f'{image_folder}{img}.png', dpi=200, bbox_inches='tight')
    plt.close()
    
import matplotlib.patches as patches

def plot_subdomains_with_borders(no_domains, no_domains_x, no_domains_y, sd, img):
    subplot_size_x = 3.11
    subplot_size_y = 2

    fig_size_x = subplot_size_x * no_domains_x
    fig_size_y = subplot_size_y * no_domains_y
    
    fig, axs = plt.subplots(no_domains_y, no_domains_x, figsize=(2*fig_size_x, 2*fig_size_y))

    for i in range(no_domains):
        row = no_domains_y - 1 - i // no_domains_x
        col = i % no_domains_x
        image = sd[i].values_h[0,0,:,:].cpu()+sd[i].values_H[0,0,:,:].cpu()
        axs[row, col].imshow(image)
        axs[row, col].axis('off')

        # Display the size of the image on top of the image
        axs[row, col].text(0.5, 0.7, f"Size: {image.shape[0],image.shape[1]}", ha='center', va='top', transform=axs[row, col].transAxes, color='white', fontsize=14)
        axs[row, col].text(0.5, 0.75, f"Resolution: {sd[i].nx//sd[1].nx}x", ha='center', va='top', transform=axs[row, col].transAxes, color='white', fontsize=14)
        axs[row, col].text(0.5, 0.8, f"Subdomain {i}", ha='center', va='top', transform=axs[row, col].transAxes, color='white', fontsize=14)

        # Add border to the image
        rect = patches.Rectangle((0,0),1,1,linewidth=1,edgecolor='white',facecolor='none', transform=axs[row, col].transAxes)
        axs[row, col].add_patch(rect)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(f'{image_folder}{img}_bc.png', dpi=200, bbox_inches='tight')
    plt.close()


# with open(file_name, 'a', newline='') as csvfile:
with torch.no_grad():
    while istep <= 1:
        img = int(real_time/900)
        get_source(real_time)
        # note:: I can take sd out to here which increases the speed as halo update is not needed twice
        for t in range(2):
            conv0  = model( sd, dt, rho )

        real_time = real_time + dt
        istep +=1

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
        if np.max(np.abs(conv0.cpu().detach().numpy())) > 10.0:
            print('sd0, Not converged !!!!!!')
            break

        if istep == 2 or istep==100800 or istep==201600 or istep==302400 or istep==403200:
            img = int(real_time/900)
            print(f'Time step:, {istep}, img = {int(real_time/900)} , time in seconds = {real_time:.0f}', 'wall clock:', (time.time()-start)/60)
            for element in sd:
#                     plt.imshow((element.values_h+element.values_H)[0,0,:,:].cpu().detach().numpy(),vmin=0, vmax=6, cmap='Blues' )
#                     plt.axis('off')
#                     plt.savefig(f'/home/an619/Desktop/git/AI/RMS/semi/2D/Linear_results/{img}_{element.index}_bc.png', dpi=200, bbox_inches='tight')
                np.save(f'/home/an619/Desktop/git/AI/RMS/semi/2D/Linear_results/flex_vel/L/{istep}_{element.index}', arr=((element.values_u**2+element.values_v**2)**0.5).cpu().detach().numpy()[0,0,:,:])
                plt.imshow((((element.values_u**2+element.values_v**2))**0.5)[0,0,:,:].cpu().detach().numpy(),vmin=0, vmax=2.5, cmap='turbo' )
                plt.axis('off')
                plt.savefig(f'/home/an619/Desktop/git/AI/RMS/semi/2D/Linear_results/flex_vel/L/{img}_{element.index}_bc.png', dpi=200, bbox_inches='tight')
    end = time.time()
    print('time',(end-start),istep)