#!/usr/bin/env python3
import os
import csv
import json
import glob
from collections import Counter

def check_map_dimensions(data_dir="pacman_data"):
    """Verifica las dimensiones de todos los mapas en los archivos CSV"""
    dimensions = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print(f"Revisando {len(csv_files)} archivos CSV...")
    
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if int(row.get('agent_index', 0)) == 0:  # Solo agente Pacman
                    map_matrix_str = row.get('map_matrix', '[]')
                    try:
                        map_matrix = json.loads(map_matrix_str)
                        if map_matrix and isinstance(map_matrix, list) and isinstance(map_matrix[0], list):
                            height = len(map_matrix)
                            width = len(map_matrix[0])
                            dimensions.append((height, width))
                    except (json.JSONDecodeError, IndexError):
                        print(f"Error en archivo {csv_file}, fila {i}: {map_matrix_str[:50]}...")
    
    # Contar dimensiones únicas
    dim_counts = Counter(dimensions)
    print(f"\nDimensiones encontradas:")
    for dim, count in dim_counts.items():
        print(f"  {dim[0]}x{dim[1]}: {count} mapas")
    
    if len(dim_counts) > 1:
        print(f"\n¡PROBLEMA! Se encontraron {len(dim_counts)} dimensiones diferentes.")
        print("Para el entrenamiento necesitamos mapas de dimensiones consistentes.")
    else:
        print(f"\n✓ Todos los mapas tienen las mismas dimensiones: {list(dim_counts.keys())[0]}")
    
    return list(dim_counts.keys())

if __name__ == "__main__":
    unique_dims = check_map_dimensions()
