import os
import glob
import json
import csv
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Fijamos todas las semillas para reproducibilidad
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Constantes para la arquitectura convolucional
NUM_ACTIONS = 4  # North, South, East, West (sin Stop según especificaciones)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
MODELS_DIR = "models"

# Mapeo de acciones a índices (sin Stop)
ACTION_TO_IDX = {
    'North': 0,
    'South': 1, 
    'East': 2,
    'West': 3
}

# Mapeo de índices a acciones
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Implementación sencilla de una red neuronal feed-forward
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """Red neuronal artificial básica (perceptrón multicapa)."""

    def __init__(self, layers_sizes):
        """Inicializa la red con una lista de tamaños de capas."""
        self.num_layers = len(layers_sizes)
        self.layers_sizes = layers_sizes

        # Pesos y biases para cada capa
        self.weight_matrices = []
        self.bias_vectors = []

        random.seed(0)  # reproducibilidad
        for i in range(self.num_layers - 1):
            input_size = layers_sizes[i]
            output_size = layers_sizes[i + 1]
            weights = [[random.uniform(-0.5, 0.5) for _ in range(output_size)]
                       for _ in range(input_size)]
            biases = [random.uniform(-0.5, 0.5) for _ in range(output_size)]
            self.weight_matrices.append(weights)
            self.bias_vectors.append(biases)

    # Funciones de activación ------------------------------------------------
    def _sigmoid(self, x):
        if x < -100:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _sigmoid_deriv(self, sigmoid_output):
        return sigmoid_output * (1.0 - sigmoid_output)

    # Propagación hacia adelante --------------------------------------------
    def forward(self, inputs):
        if len(inputs) != self.layers_sizes[0]:
            raise ValueError(
                f"Dimensión de entrada {len(inputs)} no coincide con esperada ({self.layers_sizes[0]})")

        activation = inputs
        for i in range(self.num_layers - 1):
            next_activation = []
            weights = self.weight_matrices[i]
            biases = self.bias_vectors[i]
            for j in range(self.layers_sizes[i + 1]):
                weighted_sum = 0.0
                for k in range(self.layers_sizes[i]):
                    weighted_sum += activation[k] * weights[k][j]
                weighted_sum += biases[j]
                next_activation.append(self._sigmoid(weighted_sum))
            activation = next_activation
        return activation

    # Entrenamiento por backpropagation -------------------------------------
    def train(self, X, Y, learning_rate=0.5, epochs=1000):
        if len(X) != len(Y):
            raise ValueError("El número de ejemplos de X y Y no coincide.")
        n_samples = len(X)

        for _ in range(epochs):
            for idx in range(n_samples):
                inputs = X[idx]
                target = Y[idx]

                activations = [inputs]
                net_sums = []
                for i in range(self.num_layers - 1):
                    layer_input = activations[i]
                    weights = self.weight_matrices[i]
                    biases = self.bias_vectors[i]
                    z_values = []
                    next_activation = []
                    for j in range(self.layers_sizes[i + 1]):
                        z = 0.0
                        for k in range(self.layers_sizes[i]):
                            z += layer_input[k] * weights[k][j]
                        z += biases[j]
                        z_values.append(z)
                        next_activation.append(self._sigmoid(z))
                    net_sums.append(z_values)
                    activations.append(next_activation)

                # Cálculo del error en la salida
                output_errors = [0.0] * self.layers_sizes[-1]
                for j in range(self.layers_sizes[-1]):
                    output_errors[j] = target[j] - activations[-1][j]

                # Retropropagación
                deltas = [None] * (self.num_layers - 1)
                last_idx = self.num_layers - 2
                deltas[last_idx] = [0.0] * self.layers_sizes[-1]
                for j in range(self.layers_sizes[-1]):
                    delta = output_errors[j] * self._sigmoid_deriv(activations[-1][j])
                    deltas[last_idx][j] = delta

                for i in range(self.num_layers - 3, -1, -1):
                    deltas[i] = [0.0] * self.layers_sizes[i + 1]
                    for j in range(self.layers_sizes[i + 1]):
                        err = 0.0
                        for k in range(self.layers_sizes[i + 2]):
                            err += self.weight_matrices[i + 1][j][k] * deltas[i + 1][k]
                        activation_val = self._sigmoid(net_sums[i][j])
                        deltas[i][j] = err * self._sigmoid_deriv(activation_val)

                # Actualizar pesos y biases
                for i in range(self.num_layers - 1):
                    layer_activation = activations[i]
                    delta_layer = deltas[i]
                    for k in range(self.layers_sizes[i]):
                        for j in range(self.layers_sizes[i + 1]):
                            self.weight_matrices[i][k][j] += learning_rate * layer_activation[k] * delta_layer[j]
                    for j in range(self.layers_sizes[i + 1]):
                        self.bias_vectors[i][j] += learning_rate * delta_layer[j]


# Esto es obligatorio para poder usar dataloaders en pytorch
class PacmanDataset(Dataset):
    def __init__(self, maps, actions):
        """
        Dataset para mapas de Pacman en formato multicanal.
        maps: numpy array de forma (N, 5, H, W)
        actions: lista de índices de acciones
        """
        self.maps = maps
        self.actions = actions
    
    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        # Los mapas ya están en formato (5, H, W)
        map_tensor = torch.FloatTensor(self.maps[idx])
        action_tensor = torch.LongTensor([self.actions[idx]])
        return map_tensor, action_tensor.squeeze()


class PacmanNet(nn.Module):
    """
    Red neuronal convolucional para Pacman según especificaciones:
    - Entrada: (C,H,W) maps [walls, food, capsules, ghosts, pacman] (C=5)
    - Conv1: 64 filtros, 8×8, stride 4, ReLU
    - Conv2: 64 filtros, 4×4, stride 2, ReLU
    - BatchNorm(64)
    - Conv3: 64 filtros, 3×3, stride 2, ReLU
    - Flatten: →64
    - FC1: 64 → 32, ReLU
    - Salida: 32 → 4 logits (acciones: North, South, East, West)
    """
    def __init__(self, map_height, map_width):
        super(PacmanNet, self).__init__()
        
        # Activaciones (definir primero)
        self.relu = nn.ReLU()
        
        # Capas convolucionales adaptadas para mapas pequeños (20x11)
        # Conv1: 64 filtros, reducir kernel y stride para mapas pequeños
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1)
        # Conv2: 64 filtros, usar pooling para reducir dimensiones
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(64)
        # Conv3: 64 filtros, kernel más pequeño
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        # Calcular el tamaño de la salida de las convoluciones con un tensor dummy
        with torch.no_grad():
            dummy_input = torch.zeros(1, 5, map_height, map_width)
            dummy_output = self._forward_conv(dummy_input)
            self.conv_output_size = dummy_output.numel()
        
        # Capas fully connected
        self.fc1 = nn.Linear(self.conv_output_size, 32)
        self.output = nn.Linear(32, NUM_ACTIONS)
        
    def _forward_conv(self, x):
        """Procesa solo las capas convolucionales"""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.batch_norm(x)
        x = self.relu(self.conv3(x))
        return x
    
    def forward(self, x):
        """
        Forward pass de la red.
        Entrada: (batch_size, 5, height, width) - 5 canales separados
        Salida: (batch_size, 4) - logits para 4 acciones
        """
        # Procesar con capas convolucionales
        x = self._forward_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Capas fully connected
        x = self.relu(self.fc1(x))
        x = self.output(x)  # Sin softmax, se aplica en CrossEntropyLoss
        
        return x


def load_and_merge_data(data_dir="pacman_data"):
    """Carga todos los archivos CSV de partidas y los combina, filtrando acciones Stop"""
    all_maps = []
    all_actions = []
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Archivos CSV encontrados: {csv_files}")
    
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
                    map_matrix = json.loads(row.get('map_matrix', '[]'))
                    
                    # Filtrar acciones Stop y verificar que los datos sean válidos
                    if action in ACTION_TO_IDX and map_matrix and action != 'Stop':
                        all_maps.append(map_matrix)
                        all_actions.append(ACTION_TO_IDX[action])
    
    print(f"Datos cargados: {len(all_maps)} ejemplos (sin acción Stop)")
    return all_maps, all_actions


def preprocess_maps_to_channels(maps):
    """
    Convierte las matrices del juego a formato de 5 canales para la CNN.
    Entrada: Lista de matrices con valores 0-5
    Salida: Array numpy (N, 5, H, W) donde cada canal representa un elemento del juego
    
    Codificación original:
    0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
    
    Canales de salida:
    Canal 0: walls (paredes)
    Canal 1: food (comida)
    Canal 2: capsules (cápsulas)
    Canal 3: ghosts (fantasmas)
    Canal 4: pacman
    """
    if not maps:
        raise ValueError("Lista de mapas vacía")
    
    # Detectar dimensiones más comunes para filtrar mapas inconsistentes
    dimension_counts = {}
    for map_matrix in maps:
        if isinstance(map_matrix, list) and len(map_matrix) > 0:
            height = len(map_matrix)
            width = len(map_matrix[0]) if isinstance(map_matrix[0], list) else 0
            dim = (height, width)
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
    
    # Usar las dimensiones más frecuentes
    most_common_dim = max(dimension_counts.keys(), key=dimension_counts.get)
    height, width = most_common_dim
    
    print(f"📏 Dimensiones más comunes: {height}x{width}")
    print(f"📊 Distribución de dimensiones: {dimension_counts}")
    
    # Filtrar mapas que coincidan con las dimensiones más comunes
    valid_maps = []
    valid_indices = []
    for i, map_matrix in enumerate(maps):
        try:
            map_array = np.array(map_matrix)
            if map_array.shape == most_common_dim:
                valid_maps.append(map_matrix)
                valid_indices.append(i)
            else:
                print(f"⚠️ Descartando mapa {i} con dimensiones {map_array.shape}")
        except Exception as e:
            print(f"❌ Error procesando mapa {i}: {e}")
    
    if not valid_maps:
        raise ValueError("No hay mapas con dimensiones consistentes")
    
    print(f"✅ Mapas válidos: {len(valid_maps)}/{len(maps)}")
    
    num_maps = len(valid_maps)
    
    # Crear array de 5 canales
    processed_maps = np.zeros((num_maps, 5, height, width), dtype=np.float32)
    
    for i, map_matrix in enumerate(valid_maps):
        map_array = np.array(map_matrix, dtype=np.float32)
        
        # Canal 0: Paredes (valor 0 en original)
        processed_maps[i, 0] = (map_array == 0).astype(np.float32)
        
        # Canal 1: Comida (valor 2 en original)
        processed_maps[i, 1] = (map_array == 2).astype(np.float32)
        
        # Canal 2: Cápsulas (valor 3 en original)
        processed_maps[i, 2] = (map_array == 3).astype(np.float32)
        
        # Canal 3: Fantasmas (valor 4 en original)
        processed_maps[i, 3] = (map_array == 4).astype(np.float32)
        
        # Canal 4: Pacman (valor 5 en original)
        processed_maps[i, 4] = (map_array == 5).astype(np.float32)
    
    print(f"Forma de los datos de entrada: {processed_maps.shape}")
    print(f"Dimensiones del mapa: {height}x{width}")
    print(f"Canales: 5 [walls, food, capsules, ghosts, pacman]")
    
    return processed_maps, (height, width), valid_indices


def train_model(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    """Entrena el modelo con el dataset proporcionado usando la configuración especificada"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_accuracy = 0.0
    best_model_state = None
    
    print(f"Comenzando entrenamiento por {num_epochs} épocas...")
    print(f"Learning rate: {LEARNING_RATE}, Weight decay: {WEIGHT_DECAY}")
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (maps, actions) in enumerate(train_loader):
            maps, actions = maps.to(device), actions.to(device)
            
            # Forward pass
            outputs = model(maps)
            loss = criterion(outputs, actions)
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += actions.size(0)
            train_correct += predicted.eq(actions).sum().item()
        
        # Evaluación
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for maps, actions in test_loader:
                maps, actions = maps.to(device), actions.to(device)
                outputs = model(maps)
                loss = criterion(outputs, actions)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += actions.size(0)
                test_correct += predicted.eq(actions).sum().item()
        
        train_accuracy = 100. * train_correct / train_total
        test_accuracy = 100. * test_correct / test_total
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # Early stopping: guardar el mejor modelo
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()
            print(f'¡Nuevo mejor modelo con {best_accuracy:.2f}% de precisión en validación!')
    
    # Cargar el mejor modelo
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f'Modelo final cargado: precisión en validación {best_accuracy:.2f}%')
    
    return model


def save_model(model, input_size, model_path="models/pacman_model.pth"):
    """Guarda el modelo entrenado"""
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    
    # Guardar el modelo junto con información sobre el tamaño de entrada
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
    }
    torch.save(model_info, model_path)
    print(f'Modelo guardado en {model_path}')


def main():
    import time
    start_time = time.time()
    
    # Verificar disponibilidad de GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar datos
    maps, actions = load_and_merge_data()
    
    if len(maps) == 0:
        print("ERROR: No se encontraron datos válidos para entrenar")
        return
    
    # Preprocesar mapas a formato de 5 canales
    maps, (map_height, map_width), valid_indices = preprocess_maps_to_channels(maps)
    
    # Filtrar las acciones usando los índices válidos
    actions = [actions[i] for i in valid_indices]
    
    # Verificar distribución de acciones
    action_counts = Counter(actions)
    print(f"Distribución de acciones: {action_counts}")
    
    # Dividir en conjunto de entrenamiento y validación (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        maps, actions, test_size=0.2, random_state=42, stratify=actions
    )
    
    print(f"Datos de entrenamiento: {len(X_train)}")
    print(f"Datos de validación: {len(X_test)}")
    
    # Crear datasets
    train_dataset = PacmanDataset(X_train, y_train)
    test_dataset = PacmanDataset(X_test, y_test)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Crear modelo con la arquitectura convolucional
    model = PacmanNet(map_height, map_width).to(device)
    print(f"Modelo creado:")
    print(f"  - Entrada: (5, {map_height}, {map_width}) canales")
    print(f"  - Salida: {NUM_ACTIONS} acciones")
    print(f"  - Parámetros totales: {sum(p.numel() for p in model.parameters())}")
    
    # Entrenar modelo
    trained_model = train_model(model, train_loader, test_loader, device)
    
    # Guardar modelo con el nombre correcto para NeuralAgent
    save_model(trained_model, (map_height, map_width))
    print(f"Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")
    
    print("\n" + "="*50)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*50)
    print("Para probar el modelo entrenado, ejecuta:")
    print("python pacman.py -p NeuralAgent -n 10 -q")
    print("="*50)


if __name__ == "__main__":
    main()
