import os
import glob
import json
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Fijamos todas las semillas para reproducibilidad
torch.manual_seed(22)
random.seed(22)
np.random.seed(22) # Añadido para numpy

# Constantes
INPUT_SIZE_SHAPE = None # Se determinará en tiempo de ejecución como (height, width)
HIDDEN_SIZE_FC = 128  # Tamaño para las capas densas después de la convolución
NUM_ACTIONS = 5      # Stop, North, South, East, West
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100 # Podría necesitar más épocas para una CNN
MODELS_DIR = "models"

# Mapeo de acciones a índices
ACTION_TO_IDX = {
    'Stop': 0,
    'North': 1,
    'South': 2,
    'East': 3,
    'West': 4
}

# Mapeo de índices a acciones
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}

class PacmanDataset(Dataset):
    def __init__(self, maps, actions):
        self.maps = maps
        self.actions = actions

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        map_tensor = torch.FloatTensor(self.maps[idx])
        action_tensor = torch.LongTensor([self.actions[idx]])
        return map_tensor, action_tensor.squeeze()

class PacmanNet(nn.Module):
    def __init__(self, input_shape, hidden_size_fc, output_size):
        """
        input_shape: una tupla (height, width) del mapa.
        hidden_size_fc: número de neuronas en la capa oculta densa.
        output_size: número de acciones posibles.
        """
        super(PacmanNet, self).__init__()
        height, width = input_shape

        # Bloque Convolucional 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # Batch Normalization
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Salida: H/2, W/2

        # Bloque Convolucional 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # Batch Normalization
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Salida: H/4, W/4

        # Calcular el tamaño de las características aplanadas después de las capas convolucionales
        # Esta es una forma de hacerlo dinámicamente
        h_after_convs = height // 4
        w_after_convs = width // 4

        # Asegurarse de que las dimensiones no sean cero si el mapa es muy pequeño
        if h_after_convs == 0:
            print(f"Advertencia: La altura del mapa ({height}) es demasiado pequeña para 2 capas de pooling. "
                  f"Altura resultante tras convoluciones es 0. Ajustando a 1.")
            h_after_convs = 1
        if w_after_convs == 0:
            print(f"Advertencia: La anchura del mapa ({width}) es demasiado pequeña para 2 capas de pooling. "
                  f"Anchura resultante tras convoluciones es 0. Ajustando a 1.")
            w_after_convs = 1
            
        self.conv_output_features = 32 * h_after_convs * w_after_convs

        # Capas Completamente Conectadas (Feedforward)
        self.fc1 = nn.Linear(self.conv_output_features, hidden_size_fc)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Aumentamos el dropout para una red más compleja

        self.fc2 = nn.Linear(hidden_size_fc, output_size)

    def forward(self, x):
        # Forma de entrada x: (batch_size, height, width)
        # Añadir una dimensión de canal: (batch_size, 1, height, width)
        x = x.unsqueeze(1)

        # Pasar por los bloques convolucionales
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))

        # Aplanar la salida para las capas completamente conectadas
        # Forma después de convoluciones: (batch_size, 32, h_after_convs, w_after_convs)
        x = x.view(x.size(0), -1) # Forma: (batch_size, conv_output_features)

        # Pasar por las capas completamente conectadas
        x = self.dropout(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)

        return x

def load_and_merge_data(data_dir="pacman_data"):
    """Carga todos los archivos CSV de partidas y los combina en un único DataFrame"""
    all_maps = []
    all_actions = []

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    # print(f"Archivos CSV encontrados: {csv_files}") # Descomentar para depuración

    if not csv_files:
        raise ValueError(f"No se encontraron archivos CSV en {data_dir}")

    print(f"Cargando {len(csv_files)} archivos de partidas...")

    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Solo usar movimientos de Pacman (agente 0)
                if int(row.get('agent_index', 0)) == 0:
                    action = row.get('action')
                    map_matrix_str = row.get('map_matrix', '[]')
                    try:
                        map_matrix = json.loads(map_matrix_str)
                    except json.JSONDecodeError:
                        print(f"Error decodificando JSON en {csv_file} para la fila: {row}")
                        continue # Saltar esta fila si hay un error

                    # Verificar que los datos sean válidos
                    if action in ACTION_TO_IDX and map_matrix and isinstance(map_matrix, list) and \
                       isinstance(map_matrix[0], list) and len(map_matrix[0]) > 0: # Chequeo básico de estructura
                        all_maps.append(map_matrix)
                        all_actions.append(ACTION_TO_IDX[action])
                    # else: # Descomentar para depurar datos inválidos
                    #     print(f"Datos inválidos o acción desconocida en {csv_file}: action='{action}', map_matrix='{map_matrix_str[:50]}...'")


    print(f"Datos cargados: {len(all_maps)} ejemplos")
    if not all_maps:
        raise ValueError("No se cargaron datos válidos. Verifica los archivos CSV y su contenido.")
    return all_maps, all_actions

def preprocess_maps(maps):
    """Preprocesa las matrices del juego para preparar los datos de entrada para la red"""
    if not maps:
        raise ValueError("La lista de mapas está vacía en preprocess_maps.")
        
    # Determinar las dimensiones del mapa desde el primer mapa válido
    height = len(maps[0])
    width = len(maps[0][0])

    processed_maps = np.array(maps).astype(np.float32)

    # Normalizar los valores: dividir por el valor máximo esperado en los mapas (ej. 6.0 si incluye fantasmas asustados)
    # Ajusta este valor si el máximo en tus datos es diferente.
    # El NeuralAgent en multiAgents.py normaliza por 6.0. Para consistencia:
    max_val_in_map = np.max(processed_maps) if processed_maps.size > 0 else 6.0
    if max_val_in_map == 0: max_val_in_map = 6.0 # Evitar división por cero si el mapa está vacío o solo ceros
    print(f"Valor máximo detectado en los mapas para normalización: {max_val_in_map}")
    
    # Si el script de entrenamiento usa datos con un máximo diferente (ej. 5.0),
    # es crucial que la normalización sea consistente con los datos de entrenamiento.
    # Aquí usaremos 6.0 asumiendo que es el máximo posible como en NeuralAgent.
    normalization_factor = 6.0
    processed_maps = processed_maps / normalization_factor
    
    print(f"Forma de los datos de entrada: {processed_maps.shape}")
    print(f"Tamaño del mapa detectado: {height}x{width}")
    
    return processed_maps, (height, width)


def train_model(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    """Entrena el modelo con el dataset proporcionado"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_accuracy = 0.0
    best_model_state = None
    
    print(f"Comenzando entrenamiento por {num_epochs} épocas en dispositivo {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (maps_batch, actions_batch) in enumerate(train_loader):
            maps_batch, actions_batch = maps_batch.to(device), actions_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(maps_batch)
            loss = criterion(outputs, actions_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += actions_batch.size(0)
            train_correct += predicted.eq(actions_batch).sum().item()
            
            if (batch_idx + 1) % 20 == 0: # Imprimir con menos frecuencia
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {train_loss/(batch_idx+1):.4f}, Acc: {100.*train_correct/train_total:.2f}%')
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for maps_batch, actions_batch in test_loader:
                maps_batch, actions_batch = maps_batch.to(device), actions_batch.to(device)
                outputs = model(maps_batch)
                loss = criterion(outputs, actions_batch)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += actions_batch.size(0)
                test_correct += predicted.eq(actions_batch).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        current_test_accuracy = 100. * test_correct / test_total
        
        print(f'Epoch: {epoch+1}/{num_epochs} -> '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | '
              f'Test Loss: {avg_test_loss:.4f}, Test Acc: {current_test_accuracy:.2f}%')
        
        if current_test_accuracy > best_accuracy:
            best_accuracy = current_test_accuracy
            best_model_state = model.state_dict().copy() # Guardar una copia
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        print("No se guardó ningún mejor estado del modelo, usando el estado final.")
        
    return model

def save_model(model, input_shape_tuple, model_path="models/pacman_model_cnn.pth"): # Nuevo nombre de archivo
    """Guarda el modelo entrenado"""
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_size': input_shape_tuple, # Guardar la tupla (height, width)
    }
    torch.save(model_info, model_path)

def main():
    import time
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    maps_data, actions_data = load_and_merge_data(data_dir="pacman_data") # Asegúrate que esta carpeta exista y tenga datos
    
    # INPUT_SIZE_SHAPE se determinará aquí
    global INPUT_SIZE_SHAPE 
    processed_maps_data, INPUT_SIZE_SHAPE = preprocess_maps(maps_data)
    
    # Corregir el desbalance de clases si existe (opcional pero recomendado)
    # print(f"Distribución de acciones antes de dividir: {Counter(actions_data)}")

    X_train, X_test, y_train, y_test = train_test_split(
        processed_maps_data, actions_data, test_size=0.2, random_state=22, stratify=actions_data
    )
    
    # print(f"Distribución de acciones en entrenamiento: {Counter(y_train)}")
    # print(f"Distribución de acciones en test: {Counter(y_test)}")

    train_dataset = PacmanDataset(X_train, y_train)
    test_dataset = PacmanDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    model = PacmanNet(INPUT_SIZE_SHAPE, HIDDEN_SIZE_FC, NUM_ACTIONS).to(device)
    
    # Calcular número de parámetros (opcional)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trained_model = train_model(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS)
    
    save_model(trained_model, INPUT_SIZE_SHAPE, model_path=os.path.join(MODELS_DIR, "pacman_model_cnn_final.pth"))

if __name__ == "__main__":
    main()