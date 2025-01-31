from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F

@dataclass
class Subdomain():
    '''
    __annotation__ :: tells the type of the variable
    __doc__ :: tells the description of the variable
    '''
    element_order: str
    index: int
    nx: int
    ny: int
    dx: float
    dy: float
    x2: int
    y2: int
    x0: int
    y0: int
    x1: int
    y1: int
    x3: int
    y3: int
    ratio: float
    epsilon_eta: float
    dt: float
    neig: dict = None
    tensor: torch.Tensor = None
    shared_indices: dict = None

    def __post_init__(self):
        '''
        The __post_init__ method in the Subdomain class is a special method provided by the dataclasses module in Python. 
        It is called automatically after the __init__ method of the dataclass has been called. 
        This method is used to perform additional initialization tasks that are not handled by the __init__ method.
        '''
        self.neig = {
            'top': [],
            'bottom': [],
            'left': [],
            'right': []
        }
        self.shared_indices = {
            'top': {},
            'bottom': {},
            'left': {},
            'right': {}
        }
        new_ny, new_nx = int(self.ny*self.ratio), int(self.nx*self.ratio)
        input_shape = (1, 1, new_ny, new_nx )
        self.corner_neig = None
        self.corner_node_neig = None

        self.values_h = torch.zeros(input_shape)
        self.values_H = torch.zeros(input_shape)
        self.values_u = torch.zeros(input_shape)
        self.values_v = torch.zeros(input_shape)
        self.a_u = torch.zeros(input_shape)
        self.a_v = torch.zeros(input_shape)
        self.b_u = torch.zeros(input_shape)
        self.b_v = torch.zeros(input_shape)
        self.eta1 = torch.zeros(input_shape)
        self.eta2 = torch.zeros(input_shape)
        self.values_hh = torch.zeros(input_shape)
        self.dif_values_h = torch.zeros(input_shape)
        self.values_h_old = torch.zeros(input_shape)
        self.sigma_q = torch.zeros(input_shape)
        self.k_u = torch.zeros(input_shape)
        self.k_v = torch.zeros(input_shape)
        self.k_x = torch.zeros(input_shape)
        self.k_y = torch.zeros(input_shape)
        self.b = torch.zeros(input_shape)
        self.source_h = torch.zeros(input_shape)
        self.b = torch.zeros(input_shape)
        self.H = torch.zeros(input_shape)

        # stablisation factor
        self.k1 = torch.ones(input_shape)*self.epsilon_eta
        self.k2 = torch.zeros(input_shape)
        self.k3 = torch.ones((new_ny,new_nx))*self.dx**2*0.25/self.dt
        self.kmax = None
        self.m_i = None
        self.pg_cst = None

        # Padding
        if self.element_order == 'linear':
                input_shape_pd = (1, 1, new_ny + 2, new_nx + 2)
        elif self.element_order == 'quadratic':
            input_shape_pd = (1, 1, new_ny + 4, new_nx + 4)
            self.values_hhp_L = torch.zeros((1, 1, int(self.ny*self.ratio) + 2, int(self.nx*self.ratio) + 2))
                 
        self.values_uu = torch.zeros(input_shape_pd)
        self.values_vv = torch.zeros(input_shape_pd)
        self.b_uu = torch.zeros(input_shape_pd)
        self.b_vv = torch.zeros(input_shape_pd)
        self.eta1_p = torch.zeros(input_shape_pd)
        self.dif_values_hh = torch.zeros(input_shape_pd)
        self.values_hhp = torch.zeros(input_shape_pd)
        self.values_hp = torch.zeros(input_shape_pd)
        self.values_Hp = torch.zeros(input_shape_pd)
        self.k_uu = torch.zeros(input_shape_pd)
        self.k_vv = torch.zeros(input_shape_pd)
        self.pad_H = torch.zeros(input_shape_pd)
        
        # Halos
        self.halo_u = [
            torch.zeros((new_nx+4)),   # bottom
            torch.zeros((new_ny+4)),   # left
            torch.zeros((new_ny+4)),   # right
            torch.zeros((new_nx+4)),   # top
        ]
        self.halo_v = copy.deepcopy(self.halo_u)
        self.halo_h = copy.deepcopy(self.halo_u)
        self.halo_hh = copy.deepcopy(self.halo_u)
        self.halo_dif_h = copy.deepcopy(self.halo_u)
        self.halo_eta = copy.deepcopy(self.halo_u)
        self.halo_eta1 = copy.deepcopy(self.halo_u)
        self.halo_k_uu = [
            torch.zeros((new_nx)),   # bottom
            torch.zeros((new_ny)),   # left
            torch.zeros((new_ny)),   # right
            torch.zeros((new_nx)),   # top
        ]
        self.halo_k_vv = copy.deepcopy(self.halo_k_uu)
        
        # CNN models
        self.dif = None
        self.xadv = None
        self.yadv = None
        self.CNN3D_Su = None
        self.CNN3D_Sv = None
        self.CNN3D_pu = None
        self.CNN3D_pv = None
        self.CNN3D_A_padd = None
        self.CNN3D_A = None

    def to(self, device):
        self.device = device
        for attr in vars(self):
            value = getattr(self, attr)
            if torch.is_tensor(value):
                setattr(self, attr, value.to(device))


    def add_neighbor(self, direction, neighbor_index, my_start=None, my_end=None, neighbor_start=None, neighbor_end=None):
        if neighbor_index not in self.neig[direction]:
            self.neig[direction].append(neighbor_index)
            if my_start is not None and my_end is not None and neighbor_start is not None and neighbor_end is not None:
                self.shared_indices[direction][neighbor_index] = {
                    'my_start': my_start,
                    'my_end': my_end,
                    'neig_start': neighbor_start,
                    'neig_end': neighbor_end
                }
    def describe(self):
        print(f'------------------------------------- description of subdomain {self.index} ---------------------------------------------------')
        print('nx, ny:         ', self.nx,self.ny)
        print('dx, dy:         ', self.dx, self.dy)
        # print('nlevel:           ', self.nlevel)
        # print('sd.w shape:       ', np.shape(self.values_w))
        print('neighbours:     ', self.neig)
        print('shared indices: ', self.shared_indices)
        # print('corner neig:      ', self.corner_neig)
        # print('corner node neig: ', self.corner_node_neig)
        print(f'--------------------------------------------------------------------------------------------------------------------')


def init_subdomains(element_order, no_subdomains, nx, ny, dx, dy, x, y, ratio_list,epsilon_eta, dt):
    '''
    initialise all subdomains without applying ratio
    '''
    sd_list = []
    for i in range(no_subdomains):
        sd_index = i
        ratio = ratio_list[i]

        x0 = x[i]
        y0 = y[i] + ny[i]

        x1 = x[i] + nx[i]
        y1 = y[i] + ny[i]

        x2 = x[i]
        y2 = y[i]

        x3 = x[i] + nx[i]
        y3 = y[i]

        sd = Subdomain(element_order, sd_index, nx[i], ny[i], dx[i], dy[i], x2, y2, x0, y0, x1, y1, x3, y3, ratio, epsilon_eta, dt)
        sd_list.append(sd)
    return sd_list


def calculate_shared_indices(sd, other_sd, axis):
    '''
    Find the shared indices between two subdomains
    my_start , my_end, neig_start, neig_end
    '''
    if axis == 'x':
        my_start = max(min(sd.x2, sd.x3), min(other_sd.x0, other_sd.x1)) - min(sd.x2, sd.x3)
        my_end = min(max(sd.x2, sd.x3), max(other_sd.x0, other_sd.x1)) - min(sd.x2, sd.x3)
        neighbor_start = max(min(sd.x2, sd.x3), min(other_sd.x0, other_sd.x1)) - min(other_sd.x0, other_sd.x1)
        neighbor_end = min(max(sd.x2, sd.x3), max(other_sd.x0, other_sd.x1)) - min(other_sd.x0, other_sd.x1)
    elif axis == 'y':
        my_start = max(min(sd.y0, sd.y2), min(other_sd.y1, other_sd.y3)) - min(sd.y0, sd.y2)
        my_end = min(max(sd.y2, sd.y0), max(other_sd.y1, other_sd.y3)) - min(sd.y0, sd.y2)
        neighbor_start = max(min(sd.y0, sd.y2), min(other_sd.y1, other_sd.y3)) - min(other_sd.y1, other_sd.y3)
        neighbor_end = min(max(sd.y2, sd.y0), max(other_sd.y1, other_sd.y3)) - min(other_sd.y1, other_sd.y3)
    return my_start, my_end, neighbor_start, neighbor_end


def find_neighbours(sd_list):
    '''
    Find the neighbors of each subdomain
    '''
    for sd in sd_list:
        for other_sd in sd_list:
            if sd.index == other_sd.index:
                continue
            # Check top neighbor
            if (sd.y2 == other_sd.y0) and (max(min(sd.x2, sd.x3), min(other_sd.x0, other_sd.x1)) < min(max(sd.x2, sd.x3), max(other_sd.x0, other_sd.x1))):
                my_start, my_end, neighbor_start, neighbor_end = calculate_shared_indices(sd, other_sd, 'x')
                sd.add_neighbor('top', other_sd.index, my_start, my_end, neighbor_start, neighbor_end)
            # Check bottom neighbor
            elif (sd.y0 == other_sd.y2) and (max(min(sd.x2, sd.x3), min(other_sd.x0, other_sd.x1)) < min(max(sd.x2, sd.x3), max(other_sd.x0, other_sd.x1))):
                my_start, my_end, neighbor_start, neighbor_end = calculate_shared_indices(sd, other_sd, 'x')
                sd.add_neighbor('bottom', other_sd.index, my_start, my_end, neighbor_start, neighbor_end)
            # Check left neighbor
            elif (sd.x0 == other_sd.x1) and (max(min(sd.y0, sd.y2), min(other_sd.y1, other_sd.y3)) < min(max(sd.y2, sd.y0), max(other_sd.y1, other_sd.y3))):
                my_start, my_end, neighbor_start, neighbor_end = calculate_shared_indices(sd, other_sd, 'y')
                sd.add_neighbor('left', other_sd.index, my_start, my_end, neighbor_start, neighbor_end)
            # Check right neighbor
            elif (sd.x1 == other_sd.x0) and (max(min(sd.y1, sd.y3), min(other_sd.y0, other_sd.y2)) < min(max(sd.y1, sd.y3), max(other_sd.y0, other_sd.y2))):
                my_start, my_end, neighbor_start, neighbor_end = calculate_shared_indices(sd, other_sd, 'y')
                sd.add_neighbor('right', other_sd.index, my_start, my_end, neighbor_start, neighbor_end)


def set_physical_boundaries(sd_list, domain_width, domain_height):
    for sd in sd_list:
        if sd.y0 == domain_height:
            sd.add_neighbor('bottom', 'bottom', 0, sd.x1 - sd.x0, 0, sd.x1 - sd.x0)
        if sd.y2 == 0:
            sd.add_neighbor('top', 'top', 0, sd.x3 - sd.x2, 0, sd.x3 - sd.x2)
        if sd.x2 == 0:
            sd.add_neighbor('left', 'left', 0, sd.y0 - sd.y2, 0, sd.y0 - sd.y2)
        if sd.x3 == domain_width:
            sd.add_neighbor('right', 'righ', 0, sd.y1 - sd.y3, 0, sd.y1 - sd.y3)


def resize_subdomains(sd_list, ratio_list):
    '''
    transform all shared indices based on the ratio
    '''
    for i in range(len(ratio_list)):
        r = ratio_list[i]

        sd_list[i].nx *= r
        sd_list[i].ny *= r

        sd_list[i].dx /= r
        sd_list[i].dy /= r

        for key, neig_idx in sd_list[i].neig.items():
            if not isinstance(neig_idx[0], str):
                for nidx in neig_idx:
                    sd_list[i].shared_indices[key][nidx]['my_start'] = int(sd_list[i].shared_indices[key][nidx]['my_start'] * r)
                    sd_list[i].shared_indices[key][nidx]['my_end']   =int(sd_list[i].shared_indices[key][nidx]['my_end'] * r)
                    sd_list[i].shared_indices[key][nidx]['neig_start'] = int(sd_list[i].shared_indices[key][nidx]['neig_start'] * sd_list[nidx].ratio)
                    sd_list[i].shared_indices[key][nidx]['neig_end'] = int(sd_list[i].shared_indices[key][nidx]['neig_end'] * sd_list[nidx].ratio) 



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
    
    # now scale the sub_mesh
    resampled_subdomains = []
    for subdomain, ratio in zip(subdomains, ratio_list):
        # Calculate the new size
        new_size = (int(subdomain.shape[0] * ratio), int(subdomain.shape[1] * ratio))
        # Resample the subdomain
        resampled_subdomain = F.interpolate(subdomain.unsqueeze(0).unsqueeze(0), size=new_size, mode='bilinear', align_corners=False)
        # Remove the extra dimensions added by unsqueeze
        resampled_subdomain = resampled_subdomain.squeeze(0).squeeze(0)
        resampled_subdomains.append(resampled_subdomain)
    
    del subdomains
    return resampled_subdomains

                    
def plot_subdomains(sd_list, background_image=None, image_width=None, image_height=None, save_image=None):
    '''
    plots subdomains based on top left coordinate of each subdomain and nx ny
    It does not consider the ratio of the subdomains
    '''
    fig, ax = plt.subplots(figsize=(9.52,6.11))
    
    # If background image is provided, display it
    if background_image:
        if image_width and image_height:
            extent = [0, image_width, image_height, 0]
        else:
            extent = [0, background_image.width, background_image.height, 0]
        ax.imshow(background_image, extent=extent, aspect='auto')
        # invert y axis to match the tensor indices
        plt.gca().invert_yaxis()
    
    for sd in sd_list:
        r = 1  # sd.ratio
        x_coords = [sd.x0/r, sd.x1/r, sd.x3/r, sd.x2, sd.x0/r]
        y_coords = [sd.y0/r, sd.y1/r, sd.y3/r, sd.y2, sd.y0/r]
        ax.plot(x_coords, y_coords, label=f'Subdomain {sd.index}', color='black', linestyle='-', linewidth=1)
        ax.text((sd.x0 + sd.x1) / 2/r, (sd.y0/r + sd.y2) / 2, f'{sd.index}', fontsize=12, ha='center')

    ax.plot([0,image_width], [image_height,image_height], color='black', linestyle='-', linewidth=1)
    ax.plot([image_width,image_width], [0,image_height], color='black', linestyle='-', linewidth=1)
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    ax.axis('off')
    plt.gca().invert_yaxis()  # Invert y-axis to match tensor indices
    if save_image:
        plt.savefig(f'./flexible_bc.png', dpi=200, bbox_inches='tight')
    plt.show()