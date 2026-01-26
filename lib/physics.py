import numpy as np
import pandas as pd

# Constantes physiques
R = 287.058      # Constante des gaz parfaits pour l'air sec (J/kg·K)
G0 = 9.80665     # Gravité standard (m/s²)

ISA_LAYERS = [
    (0,      -0.0065, 288.15, 101325.0),
    (11000,   0.0,    216.65, 22632.1),
    (20000,   0.001,  216.65, 5474.89),
    (32000,   0.0028, 228.65, 868.02),
    (47000,   0.0,    270.65, 110.91),
    (51000,  -0.0028, 270.65, 66.94),
    (71000,  -0.002,  214.65, 3.96),
    (84852,   0.0,    186.95, 0.52) # Mesopause approx
]

def get_layer_from_altitude(altitude):
    """Retourne l'index de la couche ISA correspondant à l'altitude donnée."""
    for i in range(len(ISA_LAYERS) - 1):
        h_base = ISA_LAYERS[i][0]
        h_next = ISA_LAYERS[i+1][0]
        if h_base <= altitude < h_next:
            return i
    return len(ISA_LAYERS) - 1  # Dernière couche si au-dessus de toutes

def get_isa_conditions(altitudes):
    """Calcule les conditions ISA à partir d'une liste ou d'un array d'altitudes."""
    if isinstance(altitudes, (int, float)):
        altitudes = [altitudes]

    res_T, res_P, res_rho = [], [], []
    
    for h in altitudes:
        # On cherche la couche correspondante
        layer_index = get_layer_from_altitude(h)

        h_base, L, T_base, P_base = ISA_LAYERS[layer_index]
        h_next = ISA_LAYERS[layer_index+1][0] if layer_index < len(ISA_LAYERS) - 1 else None
        
        if h_next is None or h_base <= h <= h_next:
            dh = h - h_base
            
            # 1. Calcul Température
            current_T = T_base + L * dh
            
            # 2. Calcul Pression
            if L == 0:
                current_P = P_base * np.exp(-G0 * dh / (R * T_base))
            else:
                current_P = P_base * (current_T / T_base) ** (-G0 / (L * R))
            
            # 3. Calcul Densité
            current_rho = current_P / (R * current_T)
            
            res_T.append(current_T)
            res_P.append(current_P)
            res_rho.append(current_rho)
            
                
    return np.array(res_T), np.array(res_P), np.array(res_rho)