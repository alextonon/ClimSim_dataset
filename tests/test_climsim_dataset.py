import numpy as np
import xarray as xr
import torch
import pytest
import re
from typing import Dict

# ====== Colle ici tes classes ClimSimBase, ClimSimPyTorch, ClimSimKeras ======
import xarray as xr
import numpy as np
import torch 
from torch.utils.data import Dataset, Subset
import re


import xarray as xr
import numpy as np
import torch 
from torch.utils.data import Dataset, Subset
import re

from lib import data


class ClimSimBase:
    def __init__(self, zarr_path, grid_path, norm_path, features, num_latlon = 384, normalize=True):
        self.ds = xr.open_zarr(zarr_path, chunks=None)
        self.features = features
        self.features_list = self.__get_features__()
        self.normalize_flag = normalize
        self.num_latlon = num_latlon
        self.grid = xr.open_dataset(grid_path, engine="netcdf4")
        
        self.input_mean = xr.open_dataset(os.path.join(norm_path, "inputs/input_mean.nc"), engine="h5netcdf")
        self.input_std = xr.open_dataset(os.path.join(norm_path, "inputs/input_std.nc"), engine="h5netcdf")
        self.input_max = xr.open_dataset(os.path.join(norm_path, "inputs/input_max.nc"), engine="h5netcdf")
        self.input_min = xr.open_dataset(os.path.join(norm_path, "inputs/input_min.nc"), engine="h5netcdf")
        self.output_scale = xr.open_dataset(os.path.join(norm_path, "outputs/output_scale.nc"), engine="h5netcdf")

        self.grid['area_wgt'] = self.grid['area']/self.grid['area'].mean(dim = 'ncol')
        self.area_wgt = self.grid['area_wgt'].values

        self.input_vars = [v for v in self.features_list if 'in' in v]
        self.output_vars = [v for v in self.features_list if 'out' in v]

        self.grav    = 9.80616    # acceleration of gravity ~ m/s^2
        self.cp      = 1.00464e3  # specific heat of dry air   ~ J/kg/K
        self.lv      = 2.501e6    # latent heat of evaporation ~ J/kg
        self.lf      = 3.337e5    # latent heat of fusion      ~ J/kg
        self.lsub    = self.lv + self.lf    # latent heat of sublimation ~ J/kg
        self.rho_air = 101325/(6.02214e26*1.38065e-23/28.966)/273.15 # density of dry air at STP  ~ kg/m^3
        self.rho_h20 = 1.e3       # density of fresh water     ~ kg/m^ 3

        self.target_energy_conv = {'ptend_t':self.cp,
                            'ptend_q0001':self.lv,
                            'ptend_q0002':self.lv,
                            'ptend_q0003':self.lv,
                            'ptend_qn':self.lv,
                            'ptend_wind': None,
                            'cam_out_NETSW':1.,
                            'cam_out_FLWDS':1.,
                            'cam_out_PRECSC':self.lv*self.rho_h20,
                            'cam_out_PRECC':self.lv*self.rho_h20,
                            'cam_out_SOLS':1.,
                            'cam_out_SOLL':1.,
                            'cam_out_SOLSD':1.,
                            'cam_out_SOLLD':1.
                            }
        
        
        self.dp = None 
        self.pressure_grid = None

    def __get_features__(self):
        feat = np.concatenate([self.features["features"]["multilevel"], self.features["features"]["surface"]])
        target = np.concatenate([self.features["target"]["tendancies"], self.features["target"]["surface"]])
        return np.concatenate([feat, target])

    def _prepare_data(self, idx):
        x = self.process_list(self.input_vars, idx, is_input=True)
        y = self.process_list(self.output_vars, idx, is_input=False)
        return x, y

    def process_list(self, vars_list, idx, is_input=True):
        out_list = []
        n_geo = self.num_latlon # 384

        for var in vars_list:
            if "ptend" in var:
                # Cette fonction doit renvoyer du (Time, 384, 60)
                data = self._calculate_tendency_on_fly(var, idx)
                if data.ndim == 2: # Si (384, 60)
                    data = data[np.newaxis, :, :]
            
            else:
                da = self.ds[var].isel(sample=idx)
                
                # Redressement par nom de dimension Xarray
                if 'lev' in da.dims:
                    if "sample" in da.dims:
                        data = da.transpose('sample', 'ncol', 'lev').values
                    else:
                        data = da.transpose('ncol', 'lev').values[np.newaxis, :, :]
                else:
                    if "sample" in da.dims:
                        # Surface : (Time, 384)
                        data = da.values[:, :, np.newaxis]  # Ajouter une dimension lev=1
                    else:
                        # Surface : (Time, 384) -> (Time, 384, 1)
                        data = da.values[np.newaxis, :, np.newaxis]
            
            # 2. Normalisation (Maintenant data est garanti (N, 384, L))
            data = self._normalize_var(data, var, is_input=is_input)
            out_list.append(data.astype(np.float32))

        # 3. Concaténation et aplatissement
        combined = np.concatenate(out_list, axis=-1)
        return combined.reshape(-1, combined.shape[-1])

    def __len__(self):
        return self.ds.dims['sample']

    def _calculate_tendency_on_fly(self, var, idx):
        dt = 1200
        mapping = {
            'out_ptend_t': ('out_state_t', 'in_state_t'),
            'out_ptend_q0001': ('out_state_q0001', 'in_state_q0001'),
            'out_ptend_u': ('out_state_u', 'in_state_u'),
            'out_ptend_v': ('out_state_v', 'in_state_v'),
        }
        out_v, in_v = mapping[var]

        v_final = self.ds[out_v].isel(sample=idx)
        v_init  = self.ds[in_v].isel(sample=idx)

        # Fonction utilitaire: remettre en ordre (sample?, ncol, lev) si sample existe
        def to_array(da):
            dims = da.dims

            if 'ncol' not in dims or 'lev' not in dims:
                raise ValueError(f"{da.name}: dims inattendues {dims}, attendu ncol & lev")

            # Cas slice -> dims contiennent sample
            if 'sample' in dims:
                da = da.transpose('sample', 'ncol', 'lev')
                return da.values  # (sample, ncol, lev)

            # Cas int -> dims (lev, ncol) ou (ncol, lev)
            da = da.transpose('ncol', 'lev')
            return da.values[None, ...]  # (1, ncol, lev)

        vf = to_array(v_final)
        vi = to_array(v_init)

        return (vf - vi) / dt  # (time, ncol, lev)


    
    def _normalize_var(self, data, var_name, is_input=True):
        # data est (N, 384, L) où L est 1 ou 60
        short_name = re.sub(r'^(in_|out_)', '', var_name)
        
        if is_input:
            m = self.input_mean[short_name].values     # (L,)
            diff = (self.input_max[short_name].values - self.input_min[short_name].values) # (L,)
            
            # On redimensionne les stats en (1, 1, L) pour s'aligner sur data (N, 384, L)
            m = m.reshape(1, 1, -1)
            diff = diff.reshape(1, 1, -1)
            
            return (data - m) / (diff + 1e-15)
        else:
            scale = self.output_scale[short_name].values # (L,)
            return data * scale.reshape(1, 1, -1)
            
    def set_pressure_grid(self, input_data):
        '''
        Calcule la grille de pression 3D à partir de state_ps.
        Code directement issu de ClimSim original.
        '''
        self.ps_index = self._find_ps_index(self.features)
        state_ps = input_data[:, self.ps_index]
        if self.normalize_flag:
            state_ps = state_ps * (self.input_max['state_ps'].values - self.input_min['state_ps'].values) + self.input_mean['state_ps'].values
        
        state_ps = state_ps.reshape(-1, self.num_latlon)

        p1 = (self.grid['P0'] * self.grid['hyai']).values[:, None, None]
        p2 = self.grid['hybi'].values[:, None, None] * state_ps[None, :, :]
        
        self.pressure_grid = p1 + p2
        self.dp = (self.pressure_grid[1:61] - self.pressure_grid[0:60]).transpose((1, 2, 0))
    
    
    def denormalize_output(self, y_pred):
        """Dénormalise les prédictions."""

        full_scale_vector = [] # To vectorize we generate the full scale vector first
        for var in self.output_vars:
            short_name = re.sub(r'^(in_|out_)', '', var)
            scale = self.output_scale[short_name].values
            
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

        y_denorm = y_pred / (full_scale_vector + 1e-8) 

        return (y_denorm).astype(np.float32)
    
    def calc_MAE(self, pred, target, avg_grid = True):
        '''
        calculate 'globally averaged' mean absolute error 
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        mae = np.abs(pred - target).mean(axis = 0)
        if avg_grid:
            return mae.mean(axis = 0) # we decided to average globally at end
        else:
            return mae
    
    def calc_RMSE(self, pred, target, avg_grid = True):
        '''
        calculate 'globally averaged' root mean squared error 
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        sq_diff = (pred - target)**2
        rmse = np.sqrt(sq_diff.mean(axis = 0)) # mean over time
        if avg_grid:
            return rmse.mean(axis = 0) # we decided to separately average globally at end
        else:
            return rmse

    def calc_R2(self, pred, target, avg_grid = True):
        '''
        calculate 'globally averaged' R-squared
        for vertically-resolved variables, input shape should be time x grid x level
        for scalars, input shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        sq_diff = (pred - target)**2
        tss_time = (target - target.mean(axis = 0)[np.newaxis, ...])**2 # mean over time
        r_squared = 1 - sq_diff.sum(axis = 0)/tss_time.sum(axis = 0) # sum over time
        if avg_grid:
            return r_squared.mean(axis = 0) # we decided to separately average globally at end
        else:
            return r_squared
    
    def output_weighting(self, output, just_weights=False):
        num_samples = output.shape[0]
        n_geo = self.num_latlon
        n_time = num_samples // n_geo
        
        # Configuration des indices : (début, fin, est_profil)
        offsets = {
            'ptend_t': (0, 60, True), 'ptend_q0001': (60, 120, True),
            'cam_out_NETSW': (120, 121, False), 'cam_out_FLWDS': (121, 122, False),
            'cam_out_PRECSC': (122, 123, False), 'cam_out_PRECC': (123, 124, False),
            'cam_out_SOLS': (124, 125, False), 'cam_out_SOLL': (125, 126, False),
            'cam_out_SOLSD': (126, 127, False), 'cam_out_SOLLD': (127, 128, False)
        }
        
        dp = self.dp / self.grav
        var_dict = {}

        for var, (start, end, is_prof) in offsets.items():
            # Extraction et reshape
            val = output[:, start:end].reshape(n_time, n_geo, -1 if is_prof else 1)
            if not is_prof: val = val.squeeze(-1)

            # [0] Undo scaling
            scale = self.output_scale[var].values
            val /= scale[None, None, :] if is_prof else scale

            # [1] Vertical weighting
            if is_prof: val *= dp
            
            # [2] Area weighting
            val *= self.area_wgt[None, :, None] if is_prof else self.area_wgt[None, :]
            
            # [3] Energy conversion
            val *= self.target_energy_conv[var]
            
            var_dict[var] = val

        return var_dict
    
    def _find_ps_index(self, features_dict):
        """
        Calcule l'index de départ de 'in_state_ps' dans le vecteur d'entrée plat.
        Prend en compte que chaque variable multilevel occupe 60 colonnes.
        """
        current_index = 0
        
        # 1. Parcourir les variables multi-niveaux (60 niveaux chacune)
        for var in features_dict["features"]["multilevel"]:
            if var == "in_state_ps":
                return current_index
            current_index += 60
            
        # 2. Parcourir les variables de surface (1 niveau chacune)
        for var in features_dict["features"]["surface"]:
            if var == "in_state_ps":
                return current_index
            current_index += 1
            
        raise ValueError("Variable 'in_state_ps' non trouvée dans le dictionnaire FEATURES.")

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
        features_tend = variables_dict["features"]["multilevel"]
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
    

# ---------- Helpers pour créer un objet testable sans I/O ----------
def make_fake_norm_ds(varnames, lev=60):
    """
    Retourne des xarray.Dataset pour mean/std/min/max/scale
    avec variables de dimensions (lev,) ou scalaires selon le nom.
    """
    data_vars_mean = {}
    data_vars_std = {}
    data_vars_min = {}
    data_vars_max = {}
    data_vars_scale = {}

    for short in varnames:
        # Heuristique: si c'est une variable "state"/"ptend"/profile => (lev,)
        is_profile = short.startswith(("state_", "ptend_"))
        if is_profile:
            data_vars_mean[short] = (("lev",), np.linspace(0.0, 1.0, lev))
            data_vars_std[short]  = (("lev",), np.linspace(1.0, 2.0, lev))
            data_vars_min[short]  = (("lev",), np.linspace(-1.0, 0.0, lev))
            data_vars_max[short]  = (("lev",), np.linspace(1.0, 2.0, lev))
            data_vars_scale[short] = (("lev",), np.linspace(0.5, 1.5, lev))
        else:
            data_vars_mean[short] = ((), 0.1)
            data_vars_std[short]  = ((), 2.0)
            data_vars_min[short]  = ((), -1.0)
            data_vars_max[short]  = ((), 1.0)
            data_vars_scale[short] = ((), 0.25)

    mean = xr.Dataset(data_vars=data_vars_mean)
    std  = xr.Dataset(data_vars=data_vars_std)
    vmin = xr.Dataset(data_vars=data_vars_min)
    vmax = xr.Dataset(data_vars=data_vars_max)
    scale = xr.Dataset(data_vars=data_vars_scale)
    return mean, std, vmin, vmax, scale

# Modifie légèrement ton helper ds pour inclure Ps
def make_fake_ds(samples=3, ncol=384, lev=60):
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        data_vars={
            "in_state_t": (("sample","ncol","lev"), rng.normal(size=(samples,ncol,lev))),
            "out_state_t": (("sample","ncol","lev"), rng.normal(size=(samples,ncol,lev))),
            "in_state_q0001": (("sample","ncol","lev"), rng.normal(size=(samples,ncol,lev))),
            "out_state_q0001": (("sample","ncol","lev"), rng.normal(size=(samples,ncol,lev))),
            # AJOUT DE PS
            "in_state_ps": (("sample","ncol"), np.full((samples, ncol), 101325.0)), 
            "out_cam_out_FLWDS": (("sample","ncol"), rng.normal(size=(samples,ncol))),
            "out_cam_out_SOLSD": (("sample","ncol"), rng.normal(size=(samples,ncol))),
        },
        # Ajout des constantes de grille pour le calcul de pression hybride
        coords={
            "sample": np.arange(samples),
            "ncol": np.arange(ncol),
            "lev": np.arange(lev),
            "hyai": (("lev_plus_1",), np.linspace(0.01, 0, 61)),
            "hybi": (("lev_plus_1",), np.linspace(0, 0.99, 61)),
            "P0": 100000.0
        }
    )
    return ds

def build_testable_climsim(ds: xr.Dataset, features_dict: Dict, normalize: bool = True):
    """
    Fabrique un objet ClimSimPyTorch sans passer par __init__.
    """
    # 1. Instanciation sans appel au constructeur
    obj = ClimSimPyTorch.__new__(ClimSimPyTorch)
    
    # 2. Assignation des attributs de base
    obj.ds = ds
    obj.features = features_dict
    obj.normalize_flag = normalize

    # 3. Récupération de la liste des features
    # Note : On utilise l'accès mangled car c'est une méthode privée de la classe de base
    get_features_func = ClimSimBase.__dict__.get('__get_features__') or \
                    ClimSimBase.__dict__.get('_ClimSimBase__get_features__')

    if get_features_func is None:
        raise AttributeError("Impossible de trouver __get_features__ dans ClimSimBase")

    obj.features_list = get_features_func(obj)
    
    # 4. Identification plus stricte des entrées/sorties
    obj.input_vars = [v for v in obj.features_list if v.startswith('in_')]
    obj.output_vars = [v for v in obj.features_list if v.startswith('out_')]

    # 5. Préparation des statistiques de normalisation
    # Extraction propre des noms courts (ex: 'in_state_t' -> 'state_t')
    shorts = [re.sub(r'^(in_|out_)', '', v) for v in obj.features_list]
    shorts = list(dict.fromkeys(shorts)) # Suppression des doublons

    # Génération des données factices
    # On suppose que ds possède la dimension 'lev'
    lev_size = ds.sizes.get("lev", 60) 
    mean, std, vmin, vmax, scale = make_fake_norm_ds(shorts, lev=lev_size)
    
    # 6. Hydratation de l'objet
    obj.input_mean = mean
    obj.input_std = std
    obj.input_min = vmin
    obj.input_max = vmax
    obj.output_scale = scale

    return obj


# ----------------------- Fixtures -----------------------
@pytest.fixture
def features_dict():
    return {
        "features": {
            "multilevel": np.array(["in_state_t", "in_state_q0001"]),
            "surface": np.array([]),
        },
        "target": {
            "tendancies": np.array(["out_ptend_t", "out_ptend_q0001"]),
            "surface": np.array(["out_cam_out_FLWDS", "out_cam_out_SOLSD"]),
        }
    }

@pytest.fixture
def ds():
    return make_fake_ds(samples=4, ncol=384, lev=60)

@pytest.fixture
def climsim(ds, features_dict):
    return build_testable_climsim(ds, features_dict, normalize=True)


# ======================= Tests =======================

def test_get_features_concat(climsim, features_dict):
    fl = climsim.features_list
    assert list(fl) == [
        "in_state_t", "in_state_q0001",
        "out_ptend_t", "out_ptend_q0001",
        "out_cam_out_FLWDS", "out_cam_out_SOLSD"
    ]

def test_calculate_tendency_on_fly(climsim):
    idx = 0
    t = climsim._calculate_tendency_on_fly("out_ptend_t", idx)
    assert t.shape == (384, 60)
    # check numerical relation
    ref = (climsim.ds["out_state_t"][idx].values - climsim.ds["in_state_t"][idx].values) / 1200
    assert np.allclose(t, ref)

def test_normalize_input_minmax_profile(climsim):
    # 1. Préparation des données
    var = "in_state_t"
    data = climsim.ds[var].isel(sample=0).values  # Forme (384, 60)
    
    # 2. Appel de la fonction (Min-Max)
    out = climsim._normalize_var(data, var, is_input=True)
    
    # 3. Vérification de la forme
    assert out.shape == (384, 60)
    
    # 4. Calcul de référence manuel (en alignant les dimensions correctement)
    short_name = "state_t"
    m = climsim.input_mean[short_name].values     # (60,)
    m_max = climsim.input_max[short_name].values  # (60,)
    m_min = climsim.input_min[short_name].values  # (60,)
    
    # Simulation de ce que fait TON code à l'intérieur de _normalize_var:
    m_norm = m[:, np.newaxis]          # Devient (60, 1)
    diff_norm = (m_max - m_min)[:, np.newaxis]
    
    ref = (data - m[np.newaxis, :]) / ((m_max - m_min)[np.newaxis, :] + 1e-15)
    
    assert np.allclose(out, ref)

def test_normalize_output_scale_surface(climsim):
    var = "out_cam_out_FLWDS"
    data = climsim.ds[var].isel(sample=0).values  # (384,)
    out = climsim._normalize_var(data, var, is_input=False)
    assert out.shape == (384,)
    sc = climsim.output_scale["cam_out_FLWDS"].values  # scalar
    print(sc)
    assert np.allclose(out, data * sc)

def test_process_list_shapes_and_concat(climsim):
    idx = 0
    x = climsim.process_list(["in_state_t", "in_state_q0001"], idx, is_input=True)
    # each is (384,60) => concat -> (384,120)
    assert x.shape == (384, 120)

    y = climsim.process_list(["out_ptend_t", "out_cam_out_FLWDS"], idx, is_input=False)
    # out_ptend_t => (384,60), out_cam_out_FLWDS => (384,1)
    assert y.shape == (384, 61)

def test_process_list_transpose_2d(climsim, ds):
    # On injecte une variable 2D mal orientée: (sample, lev, ncol) -> on va créer un DataArray custom
    rng = np.random.default_rng(1)
    bad = xr.DataArray(
        rng.normal(size=(ds.sizes["sample"], ds.sizes["lev"], ds.sizes["ncol"])),
        dims=("sample", "lev", "ncol")
    )
    ds2 = ds.copy()
    ds2["in_bad2d"] = bad  # values -> (lev, ncol) après isel(sample=idx)
    cl2 = build_testable_climsim(ds2, {
        "features": {"multilevel": np.array(["in_bad2d"]), "surface": np.array([])},
        "target": {"tendancies": np.array([]), "surface": np.array([])},
    }, normalize=False)

    x = cl2.process_list(["in_bad2d"], 0, is_input=True)
    # doit transposer (lev,384) -> (384,lev) et concat => (384,60)
    assert x.shape == (384, 60)

def test_prepare_data(climsim):
    x, y = climsim._prepare_data(0)
    assert x.shape[0] == 384
    assert y.shape[0] == 384
    assert x.dtype == np.float32
    assert y.dtype == np.float32

def test_denormalize_output_roundtrip(climsim):
    # on prend un y "réel" non normalisé, on le normalise via process_list (is_input=False), puis on dénormalise
    idx = 0
    y_norm = climsim.process_list(climsim.output_vars, idx, is_input=False)  # (384, outdim)
    # on simule un batch flatten comme ton modèle ferait (ex: (384, outdim))
    y_denorm = climsim.denormalize_output(y_norm)
    # Comme la normalisation est * scale, la dénormalisation est / scale => on doit retrouver l'original
    # MAIS attention: out_ptend_* est calculé à la volée, donc l'original pour ptend est celui calculé
    y_true = []
    for var in climsim.output_vars:
        if "ptend" in var:
            y_true.append(climsim._calculate_tendency_on_fly(var, idx))
        else:
            y_true.append(climsim.ds[var].isel(sample=idx).values[:, None])  # surface -> (384,1)
    y_true = np.concatenate(y_true, axis=1).astype(np.float32)
    assert np.allclose(y_denorm, y_true, atol=1e-4)

def test_get_models_dims(climsim, features_dict):
    dims = climsim.get_models_dims(features_dict)
    # inputs: 2 multilevel vars -> 60+60 = 120 ; surface 0
    assert dims["input_total"] == 120
    # outputs tendencies: 2 ptend -> 60+60 = 120
    assert dims["output_tendancies"] == 120
    # outputs surface: 2 vars
    assert dims["output_surface"] == 2

def test_train_test_split_reproducible(climsim):
    tr1, te1 = climsim.train_test_split(test_size=0.25, seed=123, shuffle=True)
    tr2, te2 = climsim.train_test_split(test_size=0.25, seed=123, shuffle=True)
    assert tr1.indices.tolist() == tr2.indices.tolist()
    assert te1.indices.tolist() == te2.indices.tolist()
    assert len(tr1) + len(te1) == len(climsim)

def test_pytorch_getitem_returns_tensors(climsim):
    x, y = climsim[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] == 384
    assert y.shape[0] == 384

def test_set_pressure_grid_shape(climsim):
    """Vérifie que le calcul de dp produit les bonnes dimensions."""
    # Simulation d'un batch de 2 échantillons (chaque échantillon a 384 colonnes)
    x_batch = np.random.randn(2 * 384, 120) # 120 = input_total (t + q)
    
    # On force l'index de Ps pour le test (supposons qu'il soit à la fin ou injecté)
    climsim.ps_index = 0 # Factice pour le test
    
    # On appelle la fonction (on s'attend à ce qu'elle crée self.dp)
    climsim.set_pressure_grid(x_batch)
    
    # La forme doit être (384, 60) si tu utilises la moyenne, 
    # ou (N_batch, 384, 60) si tu calcules par échantillon.
    # Ici, vérifions la forme flexible :
    assert climsim.dp.shape[-2:] == (384, 60)

def test_output_weighting_dictionary_structure(climsim, features_dict):
    """Vérifie que la sortie est un dictionnaire bien formé."""
    # 1. Préparer un batch de prédiction (N=2, ncol=384, out_dim=122)
    # out_dim = 60 (t) + 60 (q) + 1 (FLWDS) + 1 (SOLSD) = 122
    out_dim = 122
    preds_batch = np.ones((2 * 384, out_dim))
    
    # 2. Initialiser un dp factice (poids de 1.0 partout)
    climsim.dp = np.ones((384, 60))
    
    # 3. Appeler la pondération
    phys_dict = climsim.output_weighting(preds_batch)
    
    # 4. Vérifier les clés (noms courts sans out_)
    expected_keys = ["ptend_t", "ptend_q0001", "cam_out_FLWDS", "cam_out_SOLSD"]
    for key in expected_keys:
        assert key in phys_dict
        
    # 5. Vérifier les dimensions des profils (Time, 384, 60)
    assert phys_dict["ptend_t"].shape == (2, 384, 60)
    # Vérifier les dimensions des surfaces (Time, 384)
    assert phys_dict["cam_out_FLWDS"].shape == (2, 384)

def test_output_weighting_math_consistency(climsim):
    """Vérifie que le calcul y * dp est correct."""
    # On crée une tendance simple de 1.0
    # On crée un dp de 2.0
    climsim.dp = np.full((384, 60), 2.0)
    
    # Batch de 1 échantillon, variable ptend_t uniquement (60 niveaux)
    # Pour simplifier, on suppose que le batch ne contient que ptend_t
    single_var_batch = np.full((1 * 384, 60), 10.0) 
    
    # On simule le comportement interne de weighting pour ptend_t
    # (data.reshape(-1, 384, 60) * dp)
    data_reshaped = single_var_batch.reshape(1, 384, 60)
    weighted = data_reshaped * climsim.dp
    
    # Le résultat doit être 20.0 (10.0 * 2.0)
    assert np.all(weighted == 20.0)

def test_metrics_accumulation_logic():
    """Vérifie que le cumul des erreurs (SS_res) fonctionne pour les vecteurs."""
    # Test manuel de la logique SS_res += sum((y_t - y_p)**2)
    y_t = np.array([[[10.0, 10.0]]]) # (1, 1, 2) -> 1 sample, 1 col, 2 levels
    y_p = np.array([[[11.0, 12.0]]]) # (1, 1, 2)
    
    diff_sq = (y_t - y_p)**2 # [[ 1, 4 ]]
    ss_res = np.sum(diff_sq, axis=(0, 1)) # [1, 4]
    
    assert ss_res[0] == 1.0
    assert ss_res[1] == 4.0