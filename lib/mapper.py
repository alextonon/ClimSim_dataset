import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

from scipy.interpolate import griddata

class ClimSimMapper:
    def __init__(self, grid_ds):
        self.grid_ds = grid_ds
        # On récupère les coordonnées de chaque colonne
        self.lats = grid_ds.lat.values
        self.lons = grid_ds.lon.values
        
    def plot_smooth_map(self, data_vector, title=""):
        grid_lon = np.linspace(0, 360, 360)
        grid_lat = np.linspace(-90, 90, 180)
        lon_2d, lat_2d = np.meshgrid(grid_lon, grid_lat)

        # Interpolation to a regular grid to have proper colormesh
        grid_z = griddata((self.lons, self.lats), data_vector, 
                        (lon_2d, lat_2d), method='nearest')

        plt.figure(figsize=(15, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        img = ax.pcolormesh(grid_lon, grid_lat, grid_z, transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
        plt.colorbar(img)
        plt.title(title)

    def return_closer_gridpoint_index(self, lat, lon):
        """
        Retourne l'index (flattened) du point de grille le plus proche 
        en utilisant la distance de Haversine.
        """

        lon1, lat1, lon2, lat2 = map(np.radians, [lon, lat, self.lons, self.lats])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # Formule de Haversine
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        
        c = 2 * np.arcsin(np.sqrt(a))

        return np.argmin(c)