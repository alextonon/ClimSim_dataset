import re
import torch
import numpy as np

class MetricsEvaluator:
    def __init__(self, dataset, target_vars, features_config, eps=1e-15):
        self.dataset = dataset
        self.target_vars = target_vars
        self.features_config = features_config
        self.eps = eps
        self.stats = self._init_stats()

    def _short_name(self, var):
        return re.sub(r'^(in_|out_)', '', var)

    def _get_var_size(self, var):
        return 60 if var in self.features_config["target"]["tendancies"] else 1

    def _init_stats(self):
        stats = {}
        for var in self.target_vars:
            sname = self._short_name(var)
            size = self._get_var_size(var)
            stats[sname] = {
                "ss_res": np.zeros(size),
                "sum_abs_err": np.zeros(size),
                "sum_y": np.zeros(size),
                "sum_y_sq": np.zeros(size),
                "count": 0
            }
        return stats

    def update(self, y_true_flat, y_pred_flat, x_flat):
        """
        Calcule et accumule les métriques.
        Force la mise à jour de la grille de pression pour correspondre au batch.
        """
        # 1. Conversion 
        if torch.is_tensor(y_pred_flat):
            y_pred_flat = y_pred_flat.detach().cpu().numpy()
        if torch.is_tensor(y_true_flat):
            y_true_flat = y_true_flat.detach().cpu().numpy()

        # 2. Mise à jour impérative de la pression pour ce batch précis
        self.dataset.set_pressure_grid(x_flat)

        # 3. Pondération physique
        true_dict = self.dataset.output_weighting(y_true_flat)
        pred_dict = self.dataset.output_weighting(y_pred_flat)

        for var in self.target_vars:
            sname = self._short_name(var)
            yt = true_dict[sname]
            yp = pred_dict[sname]
            
            if torch.is_tensor(yt): yt = yt.detach().cpu().numpy()
            if torch.is_tensor(yp): yp = yp.detach().cpu().numpy()

            diff = yt - yp
            
            # Somme sur Time (0) et Geo (1), on garde Level (2)
            self.stats[sname]["ss_res"] += np.sum(diff**2, axis=(0, 1))
            self.stats[sname]["sum_abs_err"] += np.sum(np.abs(diff), axis=(0, 1))
            self.stats[sname]["sum_y"] += np.sum(yt, axis=(0, 1))
            self.stats[sname]["sum_y_sq"] += np.sum(yt**2, axis=(0, 1))
            self.stats[sname]["count"] += yt.shape[0] * yt.shape[1]

    def finalize(self, return_profiles=True):
            results = {}
            for var in self.target_vars:
                sname = self._short_name(var)
                s = self.stats[sname]
                n = s["count"]
                if n == 0: continue

                mae_levels = s["sum_abs_err"] / n
                
                ss_tot = s["sum_y_sq"] - (s["sum_y"]**2 / n)
                r2_levels = 1 - (s["ss_res"] / (ss_tot + self.eps))

                results[var] = {
                    "mae": np.mean(mae_levels),   
                    "r2": np.mean(r2_levels)         
                }
                
                if return_profiles:
                    results[var]["mae_levels"] = mae_levels
                    results[var]["r2_levels"] = r2_levels
                    
            return results