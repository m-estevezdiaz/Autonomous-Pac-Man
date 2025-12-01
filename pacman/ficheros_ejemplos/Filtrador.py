import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff as scipy_arff

def calcular_deltas(px, py, ghost_coords):
    diffs = [(gx - px, gy - py) for gx, gy in ghost_coords]
    dists = [np.linalg.norm([dx, dy]) for dx, dy in diffs]
    min_idx = np.argmin(dists)
    dx, dy = diffs[min_idx]
    return dx, dy, dists[min_idx]

def guardar_arff(nombre_archivo, df, relation_name, incluir_accion=False):
    with open(nombre_archivo, 'w') as f:
        f.write(f"@RELATION {relation_name}\n\n")
        for col in df.columns[:-1] if incluir_accion else df.columns:
            f.write(f"@ATTRIBUTE {col} REAL\n")
        if incluir_accion:
            clases = sorted(df[df.columns[-1]].unique())
            f.write(f"@ATTRIBUTE action {{{','.join(clases)}}}\n")
        f.write("\n@DATA\n")
        for _, row in df.iterrows():
            f.write(','.join(map(str, row.tolist())) + "\n")

def procesar_fichero(i):
    input_path = f"Prueva_{i}_Inicial.arff"
    output_path = f"Prueva_{i}_Filtrados.arff"
    output_con_accion = f"Prueva_{i}_Filtrados_Clase.arff"

    # Cargar archivo original
    data, meta = scipy_arff.loadarff(input_path)
    df = pd.DataFrame(data)
    df["action"] = df["action"].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    # Extraer coordenadas
    px = df['pacman_x'].astype(float)
    py = df['pacman_y'].astype(float)
    ghost_coords = [
        (df[f'ghost{g}_x'].astype(float), df[f'ghost{g}_y'].astype(float))
        for g in range(1, 5)
    ]

    dxs, dys = [], []
    for idx in range(len(df)):
        ghosts = [(gx[idx], gy[idx]) for gx, gy in ghost_coords]
        dx, dy, _ = calcular_deltas(px[idx], py[idx], ghosts)
        dxs.append(dx)
        dys.append(dy)

    result_df = pd.DataFrame({
        'dist_nearest_ghost_x': dxs,
        'dist_nearest_ghost_y': dys
    })

    # Normalizar
    scaler = MinMaxScaler()
    norm_array = scaler.fit_transform(result_df)
    norm_df = pd.DataFrame(norm_array, columns=result_df.columns)

    # Guardar sin acción
    guardar_arff(output_path, norm_df, f"Prueva_{i}_Filtrados")

    # Guardar con acción (sin "Stop")
    df_accion = df[df["action"] != "Stop"].reset_index(drop=True)
    norm_df_accion = norm_df.loc[df_accion.index].reset_index(drop=True)
    norm_df_accion["action"] = df_accion["action"]
    guardar_arff(output_con_accion, norm_df_accion, f"Prueva_{i}_Filtrados_Clase", incluir_accion=True)

    print(f"Guardado: {output_path}")
    print(f"Guardado (sin 'Stop'): {output_con_accion}")

# Procesar los 4 archivos
for i in range(1, 5):
    procesar_fichero(i)
