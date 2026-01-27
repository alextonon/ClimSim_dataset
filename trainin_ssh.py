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

# %%

import json
from scipy import stats
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset


class ClimSimZarrDataset(Dataset):
    def __init__(self, zarr_path, grid_path, features, transform=None, stats=None):
        self.zarr_path = zarr_path

        self.features = features
        self.features_list = self.__get_features__()

        self.ds = xr.open_zarr(zarr_path)[self.features_list]# for testing purpose
        self.grid = xr.open_dataset(grid_path)
        self.transform = transform
        
        if stats is not None and os.path.exists(stats):
            self.stats = torch.load(stats)
        else:
            self.stats = None
        
        self.length = self.ds.dims['sample']
        
        self.input_vars = [v for v in self.ds.data_vars if 'in' in v]
        self.output_vars = [v for v in self.ds.data_vars if 'out' in v]

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

        if self.stats is not None and normalize:
            x = (x - self.stats["input_mean"]) / self.stats["input_std"]
            y = (y - self.stats["target_mean"]) / self.stats["target_std"]

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
        
        
    def __get_raw_item__(self, idx):
        """Lit les données brutes sans aucune normalisation."""
        sample = self.ds.isel(sample=idx)

        def prepare_data(vars_list): # to put all data in (ncol, nfeatures) format to concatenate them
            output_list = []
            for var in vars_list:
                data = sample[var].values # Peut être (60, 384) ou (384,)            
                if data.ndim == 2:
                    data = data.T # To make sure all start by (ncol,)
                else:
                    # C'est une variable de surface (ncol,) -> on veut (ncol, 1)
                    data = data[:, np.newaxis]
                
                output_list.append(data)
        
            # Now all data is (ncol, nfeatures) format
            return np.concatenate(output_list, axis=1).astype(np.float32)

        x = prepare_data(self.input_vars)  # Résultat: (384, 246)
        y = prepare_data(self.output_vars) # Résultat: (384, 61)
            
        return torch.from_numpy(x), torch.from_numpy(y)
        
    def compute_norm_stats(self, n_samples=1000, save=True):
        print(f"Estimation des stats sur {n_samples} échantillons aléatoires...")
        
        indices = np.random.choice(len(self), n_samples, replace=False)
        
        inputs, targets = [], []
        
        for idx in indices:
            x, y = self.__get_raw_item__(idx) 
            
            inputs.append(x)
            targets.append(y)

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        self.stats = {
            "input_mean": inputs.mean(0),
            "input_std": inputs.std(0) + 1e-6,
            "target_mean": targets.mean(0),
            "target_std": targets.std(0) + 1e-6
        }
        if save:
            stats_path = "climsim_norm_stats.pt" 
            torch.save(self.stats, stats_path)
            print(f"Stats sauvegardées dans {stats_path}")
        
        return self.stats

# %%
BATCH_SIZE = 100
N_EPOCHS = 10

FEATURES = {
    "features" :{
        "tendancies" : ["in_state_t", "in_state_q0001", "in_state_u", "in_state_v"],
        "surface" : ["in_pbuf_COSZRS", "in_pbuf_LHFLX", "in_pbuf_SHFLX", "in_pbuf_TAUX", "in_pbuf_TAUY", "in_pbuf_SOLIN"],
    },  
    "target" :{
        "tendancies" : ["out_state_t"],
        "surface" : ["out_cam_out_SOLL"]
    }
}

dataset = ClimSimZarrDataset(ZARR_PATH, LOW_RES_GRID_PATH, FEATURES, stats="climsim_norm_stats.pt")
# dataset_stats = dataset.compute_norm_stats(n_samples=2000)

model_dims = dataset.get_models_dims(FEATURES)

model = ClimSimMLP(input_dim=model_dims["input_total"], output_tendancies_dim=model_dims["output_tendancies"], output_surface_dim=model_dims["output_surface"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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



