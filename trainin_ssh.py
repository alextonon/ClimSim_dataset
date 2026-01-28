# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

import xarray as xr
import os
import torch
from functools import reduce 

torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOW_RES_SAMPLE_PATH = "data/ClimSim_low-res/train/"
LOW_RES_GRID_PATH = "data/ClimSim_low-res/ClimSim_low-res_grid-info.nc"
ZARR_PATH = "data/ClimSim_low-res.zarr"
NORM_PATH = "ClimSim/preprocessing/normalizations/"

print("Imports done.")
# %%
class ClimSimMLP(nn.Module):
    def __init__(self, input_dim=556, output_tendancies_dim=120, output_surface_dim=8):
        super(ClimSimMLP, self).__init__()
        
        # Hidden Layers: [768, 640, 512, 640, 640]
        self.layer1 = nn.Linear(input_dim, 768)
        self.layer2 = nn.Linear(768, 640)
        self.layer3 = nn.Linear(640, 512)
        self.layer4 = nn.Linear(512, 640)
        self.layer5 = nn.Linear(640, 640)
        

        self.last_hidden = nn.Linear(640, 128)
        
        # --- Output Heads ---
        self.head_tendencies = nn.Linear(128, output_tendancies_dim)
        self.head_surface = nn.Linear(128, output_surface_dim)
        
        # LeakyReLU alpha=0.15
        self.activation = nn.LeakyReLU(0.15)

    def forward(self, x):
        # Pass through the 5 main hidden layers
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.activation(self.layer4(x))
        x = self.activation(self.layer5(x))
        
        # Pass through the fixed 128 layer
        x = self.activation(self.last_hidden(x))
        
        # Output 1: Tendencies (Linear activation)
        out_linear = self.head_tendencies(x)
        
        # Output 2: Surface variables (ReLU activation)
        out_relu = F.relu(self.head_surface(x))
        
        # Concatenate along the feature dimension (dim=1)
        return torch.cat([out_linear, out_relu], dim=1)

        return out_linear

@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device, input_dim, output_dim):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs.view(-1, input_dim)  # Allow flattening for MLP
        targets = targets.view(-1, output_dim) 

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples
    return average_loss

def train_one_epoch(model, dataloader, optimizer, criterion, device, input_dim, output_dim):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training", unit="batch")

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs.view(-1, input_dim)  # Allow flattening for MLP
        targets = targets.view(-1, output_dim) 
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar description with current loss
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return total_loss / total_samples

# Pour reproduire le gagnant :
n_layers = 5
units = [768, 640, 512, 640, 640]
hp_act = 'leakyrelu'
hp_optimizer = 'RAdam'
hp_batch_size = 3072

vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC',
            'cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

# %%

import json
from scipy import stats
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset


class ClimSimZarrDataset(Dataset):
    def __init__(self, 
                zarr_path, 
                grid_path,
                norm_path,
                features,
                normalize=True,
                transform=None):
        self.zarr_path = zarr_path

        self.features = features
        self.features_list = self.__get_features__()

        self.ds = xr.open_zarr(zarr_path, chunks="auto")
        self.setup_tendencies()
        self.ds = self.ds[self.features_list]

        self.grid = xr.open_dataset(grid_path)
        self.transform = transform
        
        self.input_mean = xr.open_dataset(norm_path + "inputs/input_mean.nc")
        self.input_std = xr.open_dataset(norm_path + "inputs/input_std.nc")
        self.output_scale = xr.open_dataset(norm_path + "outputs/output_scale.nc")
        
        self.length = self.ds.dims['sample']
        
        self.input_vars = [v for v in self.ds.data_vars if 'in' in v]
        self.output_vars = [v for v in self.ds.data_vars if 'out' in v]

        if normalize:
            self.normalize()

    def setup_tendencies(self):    
        timestep = 1200 # secondes
        
        self.ds['out_ptend_t'] = (self.ds['out_state_t'] - self.ds['in_state_t']) / timestep
        self.ds['out_ptend_q0001'] = (self.ds['out_state_q0001'] - self.ds['in_state_q0001']) / timestep
        self.ds['out_ptend_u'] = (self.ds['out_state_u'] - self.ds['in_state_u']) / timestep # U tendency [m/s/s]
        self.ds['out_ptend_v'] = (self.ds['out_state_v'] - self.ds['in_state_v']) / timestep # V tendency [m/s/s]

        self.target_list = ["out_ptend_t", "out_ptend_q0001", "out_ptend_u", "out_ptend_v"]

    def __len__(self):
        return self.length
    
    def __get_features__(self):
        feat = np.concat([self.features["features"]["tendancies"], self.features["features"]["surface"]])
        target = np.concat([self.features["target"]["tendancies"], self.features["target"]["surface"]])
        return np.concat([feat, target])

    def __getitem__(self, idx, normalize=True):
        def prepare_data(vars_list):
            output_list = []
            for var in vars_list:
                data = self.ds[var][idx].values # Faster than isel ?
                
                if data.ndim == 1: # Variable de surface (ncol,) -> (ncol, 1)
                    data = data[:, np.newaxis]
                else: # Variable 3D (nlev, ncol) -> (ncol, nlev)
                    data = data.T
                
                output_list.append(data)
            
            return np.concatenate(output_list, axis=1).astype(np.float32)

        x_np = prepare_data(self.input_vars) # (384, 246)
        y_np = prepare_data(self.output_vars) # (384, 61)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        return x, y
    
    def get_models_dims(self, variables_dict):
        features_tend = variables_dict["features"]["tendancies"]
        features_surf = variables_dict["features"]["surface"]
        
        target_tend = variables_dict["target"]["tendancies"]
        target_surf = variables_dict["target"]["surface"]

        def get_var_dim(var):
            if 'lev' in self.ds[var].dims:
                return self.ds[var].sizes['lev']
            return 1

        in_tend_dim = sum([get_var_dim(var) for var in features_tend])
        in_surf_dim = len(features_surf)
        
        out_tend_dim = sum([get_var_dim(var) for var in target_tend])
        out_surf_dim = len(target_surf)

        return {
            "input_total": in_tend_dim + in_surf_dim,
            "output_tendancies": out_tend_dim,
            "output_surface": out_surf_dim
        }
        
    def train_test_split(self, test_size=0.2, seed=42, shuffle=True):
        """
        Split the dataset into train and test subsets without loading any data.
        """
        n = len(self)
        indices = np.arange(n)

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        split = int((1 - test_size) * n)

        train_indices = indices[:split]
        test_indices  = indices[split:]

        return (
            Subset(self, train_indices),
            Subset(self, test_indices)
        )

        
    def normalize(self):
        for var in self.input_vars:
            var_name = var.replace("in_", "", 1)
            self.ds[var_name] = (self.ds[var] - self.input_mean[var_name]) / self.input_std[var_name]
        
        for var in self.output_vars:
            var_name = var.replace("out_", "", 1)
            self.ds[var_name] = self.ds[var] * self.output_scale[var_name]

# %%
BATCH_SIZE = 3072
N_EPOCHS = 10

FEATURES = {
    "features" :{
        "tendancies" : ["in_state_t", "in_state_q0001", "in_state_ps"],
        "surface" : ["in_pbuf_LHFLX", "in_pbuf_SHFLX", "in_pbuf_SOLIN"],
    },  
    "target" :{
        "tendancies" : ["out_ptend_t", "out_ptend_q0001"],
        "surface" : ["out_cam_out_NETSW", "out_cam_out_FLWDS", "out_cam_out_PRECSC", "out_cam_out_PRECC", "out_cam_out_SOLS", "out_cam_out_SOLL", "out_cam_out_SOLSD", "out_cam_out_SOLLD"]
    }
}

print("Setting up dataset...")

dataset = ClimSimZarrDataset(ZARR_PATH, LOW_RES_GRID_PATH, NORM_PATH, FEATURES, normalize=True)
model_dims = dataset.get_models_dims(FEATURES)

print("Dataset ready.")
print(f"Model dimensions: {model_dims}")
model = ClimSimMLP(input_dim=model_dims["input_total"], output_tendancies_dim=model_dims["output_tendancies"], output_surface_dim=model_dims["output_surface"])
optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# %%
train, test = dataset.train_test_split(test_size=0.2, seed=42)

train_loader = torch.utils.data.DataLoader(
    train, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=4,
)

test_loader = torch.utils.data.DataLoader(
    test, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=4,
)

# %%
for epoch in range(N_EPOCHS):
    train_loss = train_one_epoch(
        model, 
        train_loader, 
        optimizer, 
        criterion, 
        device="cpu",
        input_dim=model_dims["input_total"],
        output_dim=model_dims["output_tendancies"] + model_dims["output_surface"],
        )
    val_loss = evaluate_model(
        model, 
        test_loader, 
        criterion, 
        device="cpu",
        input_dim=model_dims["input_total"],
        output_dim=model_dims["output_tendancies"] + model_dims["output_surface"],
        )
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    model_path = f"climsim_mlp_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), model_path)
        

# %%
input_nc = xr.open_dataset("ClimSim/preprocessing/normalizations/inputs/input_mean.nc")


