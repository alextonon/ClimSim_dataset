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
        # self.head_surface = nn.Linear(128, output_surface_dim)
        
        # LeakyReLU alpha=0.15
        self.activation = nn.LeakyReLU(0.15)
        
        for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

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
        # out_relu = F.relu(self.head_surface(x))
        
        # Concatenate along the feature dimension (dim=1)
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

def train_one_epoch(model, dataloader, optimizer,scheduler, criterion, device, input_dim, output_dim):
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
    
    scheduler.step()

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
import xarray as xr
import numpy as np
import torch 
from torch.utils.data import Dataset, Subset
import re


class ClimSimBase:
    def __init__(self, zarr_path, grid_path, norm_path, features, normalize=True):
        self.ds = xr.open_zarr(zarr_path, chunks=None)
        self.features = features
        self.features_list = self.__get_features__()
        self.normalize_flag = normalize

        self.grid = xr.open_dataset(grid_path)
        
        self.input_mean = xr.open_dataset(f"{norm_path}inputs/input_mean.nc")
        self.input_std = xr.open_dataset(f"{norm_path}inputs/input_std.nc")
        self.output_scale = xr.open_dataset(f"{norm_path}outputs/output_scale.nc")

        self.input_vars = [v for v in self.features_list if 'in' in v]
        self.output_vars = [v for v in self.features_list if 'out' in v]

    def __get_features__(self):
        feat = np.concatenate([self.features["features"]["tendancies"], self.features["features"]["surface"]])
        target = np.concatenate([self.features["target"]["tendancies"], self.features["target"]["surface"]])
        return np.concatenate([feat, target])

    def _prepare_data(self, idx):
        # On passe idx explicitement à process_list
        x = self.process_list(self.input_vars, idx, is_input=True)
        y = self.process_list(self.output_vars, idx, is_input=False)
        return x, y

    def process_list(self, vars_list, idx, is_input=True):
        out_list = []
        for var in vars_list:
            # Récupération
            if "ptend" in var:
                data = self._calculate_tendency_on_fly(var, idx)
            else:
                # Utiliser .isel() est plus "xarray-style" et sécurisé
                data = self.ds[var].isel(sample=idx).values
            
            # Normalisation
            data = self._normalize_var(data, var, is_input=is_input)

            # Gestion des dimensions : (ncol, nlev) -> ici ncol est implicitement 1 par idx
            # On veut un vecteur plat pour concaténer à la fin
            if data.ndim == 0: # Scalaire
                data = np.array([data])
            
            out_list.append(data.flatten())
            
        return np.concatenate(out_list).astype(np.float32)

    def __len__(self):
        return self.ds.dims['sample']

    def _calculate_tendency_on_fly(self, var, idx):
        """Calcule la tendance uniquement pour l'échantillon demandé"""
        dt = 1200
        mapping = {
            'out_ptend_t': ('out_state_t', 'in_state_t'),
            'out_ptend_q0001': ('out_state_q0001', 'in_state_q0001'),
            'out_ptend_u': ('out_state_u', 'in_state_u'),
            'out_ptend_v': ('out_state_v', 'in_state_v'),
        }
        out_v, in_v = mapping[var]
        return (self.ds[out_v][idx].values - self.ds[in_v][idx].values) / dt

    def _normalize_var(self, data, var_name, is_input=True):
        """Applique la normalisation selon que la variable est 3D ou de surface."""
        if not self.normalize_flag:
            return data

        short_name = re.sub(r'^(in_|out_)', '', var_name)

        if is_input:
            m = self.input_mean[short_name].values
            s = self.input_std[short_name].values
            
            m_norm = m[:, np.newaxis] if m.ndim > 0 else m
            s_norm = s[:, np.newaxis] if s.ndim > 0 else s
            
            return (data - m_norm) / (s_norm + 1e-8)
        else:
            scale = self.output_scale[short_name].values
            scale_norm = scale[:, np.newaxis] if scale.ndim > 0 else scale
            return data * scale_norm
    

class ClimSimPyTorch(ClimSimBase, Dataset):
    def __getitem__(self, idx):
        x_np, y_np = self._prepare_data(idx)
        return torch.from_numpy(x_np), torch.from_numpy(y_np)

    # On peut remettre ta méthode de split ici
    def train_test_split(self, test_size=0.2, seed=42, shuffle=True):
        n = len(self)
        indices = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        split = int((1 - test_size) * n)
        return Subset(self, indices[:split]), Subset(self, indices[split:])
    
    def get_models_dims(self, variables_dict):
        features_tend = variables_dict["features"]["tendancies"]
        features_surf = variables_dict["features"]["surface"]
        
        target_tend = variables_dict["target"]["tendancies"]
        target_surf = variables_dict["target"]["surface"]

        def get_var_dim(var):
            # 1. Gérer les variables virtuelles (tendances calculées)
            if 'ptend' in var:
                # On mappe vers la variable d'état pour connaître la dimension 'lev'
                # ex: out_ptend_t -> out_state_t
                source_var = var.replace('ptend', 'state')
                return self.ds[source_var].sizes['lev']
            
            # 2. Gérer les variables réelles présentes dans le Zarr
            if 'lev' in self.ds[var].dims:
                return self.ds[var].sizes['lev']
            
            # 3. Variables de surface (scalaires)
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

    

# %%
BATCH_SIZE = 200
N_EPOCHS = 10

FEATURES = {
    "features" :{
        "tendancies" : ["in_state_t", "in_state_q0001", "in_state_ps"],
        "surface" : ["in_pbuf_LHFLX", "in_pbuf_SHFLX", "in_pbuf_SOLIN"],
    },  
    "target" :{
        "tendancies" : ["out_ptend_t", "out_ptend_q0001"],
        "surface" : []
    }
}

dataset = ClimSimPyTorch(ZARR_PATH, LOW_RES_GRID_PATH, NORM_PATH, FEATURES, normalize=True)
model_dims = dataset.get_models_dims(FEATURES)

model = ClimSimMLP(input_dim=model_dims["input_total"], output_tendancies_dim=model_dims["output_tendancies"], output_surface_dim=model_dims["output_surface"])
optimizer = torch.optim.RAdam(model.parameters(), lr=2.5e-4)
criterion = nn.MSELoss()
from torch.optim.lr_scheduler import CyclicLR

# Paramètres extraits de votre code Keras
INIT_LR = 2.5e-4
MAX_LR = 2.5e-3
step_size = 2 * (len(dataset) // BATCH_SIZE) 

scheduler = CyclicLR(
    optimizer, 
    base_lr=INIT_LR, 
    max_lr=MAX_LR,
    step_size_up=step_size,
    mode='exp_range',
    gamma=0.5, # Réduit l'amplitude de moitié à chaque cycle
    cycle_momentum=False # RAdam ne gère pas toujours bien le momentum cyclique
)
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

def check_normalization(loader, name="Train"):
    inputs, targets = next(iter(train_loader))

    # On redimensionne pour isoler les features (50 batches * 384 colonnes = 19200 lignes)
    in_features = inputs.view(-1, model_dims["input_total"]) 
    out_features = targets.view(-1, model_dims["output_tendancies"] + model_dims["output_surface"])

    print(f"{'Variable Index':<15} | {'Mean':<10} | {'Std':<10}")
    print("-" * 40)

    # On regarde les 5 premières et 5 dernières variables
    indices_to_check = list(range(5)) + list(range(in_features.shape[1]-5, in_features.shape[1]))

    for i in indices_to_check:
        m = in_features[:, i].mean().item()
        s = in_features[:, i].std().item()
        print(f"Input Feature {i:<3} | {m:>10.4f} | {s:>10.4f}")

    # Alerte si le Std est trop loin de 1.0
    if any(in_features.std(dim=0) > 2.0) or any(in_features.std(dim=0) < 0.5):
        print("\n⚠️ KRITIQUE : Certaines variables d'entrée n'ont pas un Std de 1.0 !")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
# On récupère UN SEUL batch et on s'arrête
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.to(device), targets.to(device)

inputs = inputs.view(-1, 124) 
targets = targets.view(-1, 120)

print(f"Nouvelle forme pour le modèle : {inputs.shape}")

# Vérification cruciale
print(f"Nouvelle forme inputs : {inputs.shape}") 
# Si ici ce n'est pas [200, 124], ton modèle ClimSimMLP ne marchera jamais.

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

print(f"Test sur un batch de taille : {inputs.size(0)}")

for epoch in range(100):  # On monte à 100 époques pour voir la courbe descendre
    model.train()
    optimizer.zero_grad()
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.8f}")

print("fin du test sur un batch unique.")

check_normalization(train_loader, "Train")
# %%
for epoch in range(N_EPOCHS):
    train_loss = train_one_epoch(
        model, 
        train_loader, 
        optimizer,
	scheduler, 
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


