# pacman

## Play Pacman
### Install

```bash
git clone https://github.com/AhmedBegggaUA/pacman.git
cd pacman
pip install numpy
pip install matplotlib
pip install pandas
pip install torch torchvision torchaudio
```
### Run

```bash
python pacman.py # to play
python pacman.py -p RandomAgent # to play with random agent
python pacman.py -p NeuralAgent # to play with neural agent
python net.py # to train the neural agent
```


# Pacman with AI

## Practical Session 3: Pacman with AI

### Introduction

In this practical session, we will implement a simple net that plays the game Pacman. The goal is to understand all the repository and how to use it for further development.

## Repository Overview

This repository contains an implementation of the classic Pacman game with an AI component that allows for different intelligent agents to play the game. The project is based on UC Berkeley's Pacman AI projects, modified to incorporate neural networks for agent decision-making.

### Key Components

1. **Game Engine** (`game.py`): Contains the core game mechanics, including:
   - Agent classes
   - Directional movement logic
   - Game state management
   - Configuration handling for game entities

2. **Pacman Engine** (`pacman.py`): The main game runner that:
   - Implements Pacman-specific rules
   - Handles user input
   - Manages game flow
   - Provides game visualization
   - Includes command-line interfaces for running games

3. **Neural Agent** (`multiAgents.py`): Contains various agent implementations including:
   - Reflex agents
   - Minimax agents
   - Neural network-based agent (`NeuralAgent`)
   - Functions for evaluating game states

4. **Neural Network** (`net.py`): Implements the neural network architecture:
   - `PacmanNet` class using PyTorch
   - Training functionality
   - Data loading and preprocessing
   - Model saving/loading capabilities

5. **Ghost Agents** (`ghostAgents.py`): Different ghost behaviors:
   - Random ghost movement
   - Directional ghosts that chase Pacman

6. **Data Collection** (`gamedata.py`): Records game play for training:
   - Captures state, actions, and results
   - Saves game data to CSV files
   - Formats game maps as numeric matrices

7. **Replay Functionality** (`playback.py`): Allows replaying recorded games from CSV data

8. **Utilities** (`util.py`): Helper functions and data structures:
   - Queue, Stack, PriorityQueue implementations
   - Random number generation
   - Distance calculations
   - Various utility functions for game logic

## How the Neural Agent Works

The `NeuralAgent` class in `multiAgents.py` provides an AI-powered Pacman that:

1. Loads a pre-trained neural network model from a file
2. Converts game states into numeric matrices for neural network input
3. Makes decisions by combining neural network outputs with heuristic game knowledge
4. Balances exploration and exploitation with a decaying exploration rate
5. Evaluates potential moves by considering:
   - Network confidence scores
   - Proximity to food
   - Distance from ghosts (avoiding or chasing based on ghost state)
   - Game score

## Data Collection and Training

The system includes a complete pipeline for collecting game data and training models:

1. The `GameDataCollector` in `gamedata.py` captures game states during play
2. Game states are converted to numeric matrices (representing walls, food, capsules, ghosts, and Pacman)
3. The training pipeline in `net.py` loads recorded games, processes them, and trains a neural network
4. The model is saved for later use by the `NeuralAgent`

## Getting Started

To run the game with a keyboard-controlled Pacman:
```
python pacman.py
```

To run the game with the neural agent:
```
python pacman.py -p NeuralAgent
```

Other useful options:
- `-l [layout_name]`: Choose a game layout
- `-g [ghost_type]`: Select ghost agent type
- `-k [num_ghosts]`: Set number of ghosts
- `-n [num_games]`: Run multiple games
- `--csv [file_path]`: Replay a recorded game


# Source Web and GitHub
- [GitHub Repository](
    https://github.com/AhmedBegggaUA/pacman)
- [Web Page](https://inst.eecs.berkeley.edu/~cs188/fa24/projects/proj2/)