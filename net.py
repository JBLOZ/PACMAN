import os
import glob
import json
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Fijamos todas las semillas para reproducibilidad
torch.manual_seed(42)
random.seed(42)
# Constantes
INPUT_SIZE = None  # Se determinará en tiempo de ejecución basado en el tamaño del mapa
HIDDEN_SIZE = 128
NUM_ACTIONS = 4  # North, South, East, West (sin Stop)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
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
# Esto es obligatorio para poder usar dataloaders en pytorch
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
    def __init__(self, input_size, hidden_size, output_size):
        super(PacmanNet, self).__init__()
        # input_size: (H, W)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        # compute conv output size
        h, w = input_size
        def conv_dim(size, kernel, stride, pad): return (size + 2*pad - (kernel - 1) - 1) // stride + 1
        h1, w1 = conv_dim(h,4,2,1), conv_dim(w,4,2,1)
        h2, w2 = conv_dim(h1,4,2,1), conv_dim(w1,4,2,1)
        h3, w3 = conv_dim(h2,3,1,1), conv_dim(w2,3,1,1)
        conv_out = 32 * h3 * w3
        # fully connected head
        self.fc1 = nn.Linear(conv_out, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x: (batch, H, W)
        x = x.unsqueeze(1)  # add channel dim
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def load_and_merge_data(data_dir="pacman_data"):
    """Carga todos los archivos CSV de partidas y los combina en un único DataFrame"""
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
                    
                    # Verificar que los datos sean válidos
                    if action in ACTION_TO_IDX and map_matrix:
                        all_maps.append(map_matrix)
                        all_actions.append(ACTION_TO_IDX[action])
    
    print(f"Datos cargados: {len(all_maps)} ejemplos")
    return all_maps, all_actions

def preprocess_maps(maps):
    """Preprocesa las matrices del juego para preparar los datos de entrada para la red"""
    # Determinar las dimensiones del mapa
    height = len(maps[0])
    width = len(maps[0][0])
    
    # Convertir a numpy array
    processed_maps = np.array(maps).astype(np.float32)
    
    # Normalizar los valores: dividir por 5 (el valor máximo) para obtener valores entre 0 y 1
    processed_maps = processed_maps / 5.0
    
    print(f"Forma de los datos de entrada: {processed_maps.shape}")
    print(f"Tamaño del mapa: {height}x{width}")
    
    return processed_maps, (height, width)


def train_model(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    """Entrena el modelo con el dataset proporcionado"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_accuracy = 0.0
    best_model_state = None
    
    print(f"Comenzando entrenamiento por {num_epochs} épocas...")
    
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
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {train_loss/(batch_idx+1):.4f}, Acc: {100.*train_correct/train_total:.2f}%')
        
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
        
        test_accuracy = 100. * test_correct / test_total
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # Guardar el mejor modelo
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()
            print(f'¡Nuevo mejor modelo con {best_accuracy:.2f}% de precisión!')
    
    # Cargar el mejor modelo
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f'Modelo final: precisión en test {best_accuracy:.2f}%')
    
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
    
    # Preprocesar mapas
    maps, input_size = preprocess_maps(maps)
    
    
    # Dividir en conjunto de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        maps, actions, test_size=0.2, random_state=42, stratify=actions
    )
    
    # Crear datasets
    train_dataset = PacmanDataset(X_train, y_train)
    test_dataset = PacmanDataset(X_test, y_test)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Crear modelo
    model = PacmanNet(input_size, HIDDEN_SIZE, NUM_ACTIONS).to(device)
    print(f"Modelo creado: {model}")
    
    # Entrenar modelo
    trained_model = train_model(model, train_loader, test_loader, device)
    
    # Guardar modelo
    save_model(trained_model, input_size)
    print(f"Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")
# Agregar manejo de argumentos y punto de entrada
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar PacmanNet')
    parser.add_argument('--data-dir', type=str, default='pacman_data', help='Directorio de CSV de partidas')
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR, help='Directorio para guardar modelos')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Tamaño de batch')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate para el optimizador')
    args = parser.parse_args()

    # Cargar y procesar datos
    maps, actions = load_and_merge_data(data_dir=args.data_dir)
    processed_maps, input_size = preprocess_maps(maps)
    X_train, X_test, y_train, y_test = train_test_split(
        processed_maps, actions, test_size=0.2, random_state=42
    )

    # Crear datasets y loaders
    train_dataset = PacmanDataset(X_train, y_train)
    test_dataset = PacmanDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Preparar dispositivo y modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PacmanNet(input_size, HIDDEN_SIZE, NUM_ACTIONS).to(device)
    # Ajustar hiperparámetros dinámicamente
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    # Ejecutar entrenamiento
    trained_model = train_model(model, train_loader, test_loader, device, num_epochs=args.epochs)

    # Guardar modelo
    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, 'pacman_model.pth')
    # Guardar checkpoint con tamaño de entrada y pesos
    checkpoint = {
        'input_size': input_size,
        'model_state_dict': trained_model.state_dict()
    }
    torch.save(checkpoint, model_path)
    print(f'Modelo guardado en: {model_path} (checkpoint con input_size y model_state_dict)')