import xarray as xr
import numpy as np
import torch 
from torch.utils.data import Dataset, Subset
import re
import os

class ClimSimBase:
    def __init__(self, zarr_path, grid_path, norm_path, features, normalize=True):
        self.ds = xr.open_zarr(zarr_path, chunks=None)
        self.features = features
        self.features_list = self.__get_features__()
        self.normalize_flag = normalize

        self.grid = xr.open_dataset(grid_path, engine="netcdf4")
        
        self.input_mean = xr.open_dataset(os.path.join(norm_path, "inputs/input_mean.nc"), engine="h5netcdf")
        self.input_std = xr.open_dataset(os.path.join(norm_path, "inputs/input_std.nc"), engine="h5netcdf")
        self.output_scale = xr.open_dataset(os.path.join(norm_path, "outputs/output_scale.nc"), engine="h5netcdf")

        self.input_vars = [v for v in self.features_list if 'in' in v]
        self.output_vars = [v for v in self.features_list if 'out' in v]

    def __get_features__(self):
        feat = np.concatenate([self.features["features"]["multilevel"], self.features["features"]["surface"]])
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
                data = self.ds[var].isel(sample=idx).values
            
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
    
    
def denormalize_output(self, y_pred):
    """Dénormalise les prédictions en gardant la structure de batch."""
    
    # Sécurité pour les dimensions
    if y_pred.ndim == 1:
        y_pred = y_pred[np.newaxis, :]

    full_scale_vector = []
    for var in self.output_vars:
        short_name = re.sub(r'^(in_|out_)', '', var)
        scale = self.output_scale[short_name].values
        
        # Détermination de la taille (60 ou 1)
        if 'ptend' in var:
            dim_size = 60
        elif 'lev' in self.ds[var].dims:
            dim_size = self.ds[var].sizes['lev']
        else:
            dim_size = 1
        
        if np.isscalar(scale) or scale.size == 1:
            scale_expanded = np.full(dim_size, scale)
        else:
            scale_expanded = scale
            
        full_scale_vector.append(scale_expanded)
    
    full_scale_vector = np.concatenate(full_scale_vector)

    y_denorm = y_pred / (full_scale_vector + 1e-15) 

    return y_denorm.astype(np.float32) # On garde la 2D (Batch, 384)
    

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
    
class ClimSimKeras(ClimSimBase):
    def get_batch_for_keras(self, idx, input_dim, output_dim):
        x_np, y_np = self._prepare_data(idx)

        x_np = x_np.reshape(-1, input_dim)
        y_np = y_np.reshape(-1, output_dim)
        return x_np, y_np
    
    def get_sample_number(self):
        return self.ds.dims['sample']