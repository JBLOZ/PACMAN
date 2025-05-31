
## Introduccion

Dividiremos la memoria en 4 partes que corresponden a las 4 implementaciones que hemos realizado en esta practica

- Mejora de la red
- Integracion AlphaBeta con la red
- Heuristica aplicada a la red (Evaluacion de posibles estados)
- Entrenamientos mejorados


### Mejora de la red

La implementación anterior de la red...

### Integración AlphaBeta con la red

Nuestro sistema de inteligencia artificial para Pacman implementa una **arquitectura híbrida de dos niveles** que combina de manera sinérgica el algoritmo AlphaBeta con redes neuronales convolucionales. Esta integración opera en dos fases complementarias: **ordenación inteligente de movimientos** y **evaluación híbrida de estados**, creando un sistema que aprovecha tanto la garantía de optimalidad de los algoritmos de búsqueda clásicos como la capacidad de reconocimiento de patrones del aprendizaje profundo.

#### Fundamentos: AlphaBeta y la Importancia del Ordenamiento

El algoritmo **AlphaBeta** optimiza Minimax mediante poda que elimina ramas del árbol de búsqueda que no pueden mejorar la solución. Mantiene dos valores críticos: **Alpha (α)** para el mejor valor del maximizador (Pacman) y **Beta (β)** para el minimizador (fantasmas). La **poda** ocurre cuando α ≥ β, eliminando ramas enteras sin explorar.

La eficiencia de AlphaBeta depende crucialmente del **orden de exploración**: examinar primero los mejores movimientos genera más podas y reduce drásticamente el tiempo de búsqueda. Aquí es donde nuestra red neuronal proporciona su primera contribución fundamental.

#### Primera Fase: Ordenación Neural de Movimientos

La red neuronal convolucional actúa como un **sistema de priorización inteligente** que ordena los movimientos antes de que AlphaBeta comience su exploración. Este proceso se ejecuta de la siguiente manera:

**Representación del Estado**: El tablero de juego se convierte en una matriz numérica donde cada elemento se codifica específicamente: paredes (5.0), comida (3.0), fantasmas normales (2.0), fantasmas asustados (6.0), Pacman (1.0) y espacios vacíos (0.0), todo normalizado por un factor de 6.0 para estabilidad numérica.

**Predicción Neural**: La red procesa esta representación matricial utilizando capas convolucionales con batch normalization y pooling, extrayendo patrones espaciales tanto locales como globales. El resultado son probabilidades para cada una de las cinco acciones posibles, basadas en la experiencia de miles de partidas entrenadas.

**Ordenación Inteligente**: Las acciones legales se ordenan según estas probabilidades, asegurando que AlphaBeta explore primero los movimientos más prometedores según el conocimiento acumulado de la red.

Esta ordenación neural se aplica estratégicamente en el **nivel raíz** (decisión inicial de Pacman) y en todos los **nodos maximizadores** durante la búsqueda recursiva, optimizando la eficiencia de las podas en cada nivel del árbol.

#### Segunda Fase: Evaluación Híbrida de Estados

Mientras que la ordenación neural optimiza la **eficiencia de búsqueda**, la evaluación híbrida perfecciona la **calidad de la evaluación** en los nodos terminales y de corte. Esta segunda fase combina dos enfoques complementarios que operan sobre la misma representación matricial del estado.

### Heurística aplicada a la red (Evaluación de posibles estados)

La evaluación híbrida constituye el núcleo intelectual de nuestro sistema, fusionando **heurísticas tradicionales probadas** con **predicciones neuronales contextuales** para crear una función de evaluación que supera las limitaciones inherentes de cada enfoque individual.

#### Arquitectura de la Evaluación Dual

**Componente Heurístico Tradicional**: Proporciona una base sólida y interpretable mediante suma ponderada de factores críticos. Utiliza la puntuación oficial del juego como fundamento, implementa métricas de distancia Manhattan para proximidad a comida (reflejando el movimiento real en rejilla), y aplica gestión sofisticada de fantasmas que distingue entre amenazas (peso -30.0) y oportunidades de puntuación (peso 150.0 para fantasmas asustados). Incluye protección absoluta mediante penalizaciones infinitas para movimientos que resulten en muerte inmediata.

**Componente Neural Contextual**: Utiliza la misma red convolucional de la fase de ordenación, pero ahora sus predicciones se interpretan como **refinamiento contextual** de la evaluación heurística. La red aporta adaptabilidad situacional, identificando patrones complejos que las heurísticas fijas no pueden capturar y proporcionando ajustes contextuales basados en la experiencia de partidas exitosas.

#### Proceso de Integración Sinérgica

El sistema ejecuta **evaluación secuencial** donde primero se calcula la puntuación heurística tradicional para establecer una base sólida e interpretable. Simultáneamente, la red neuronal procesa el estado completo generando predicciones contextuales. Estas puntuaciones se combinan mediante **suma ponderada calibrada**, donde el componente heurístico garantiza comportamiento mínimo aceptable mientras el neural aporta refinamiento adaptativo.

La calibración de pesos resulta crítica: los pesos heurísticos (comida: 12.0, fantasmas normales: -30.0, fantasmas asustados: 150.0) se ajustaron experimentalmente para equilibrar comportamiento agresivo y defensivo, mientras que la contribución neural se calibró para influir significativamente sin dominar las garantías heurísticas tradicionales.

#### Ventajas del Sistema Integrado

Esta arquitectura dual ofrece **robustez mejorada** mediante redundancia que previene fallos catastróficos, **adaptabilidad contextual** que permite refinamiento basado en patrones aprendidos, **interpretabilidad mantenida** al preservar la base heurística tradicional, y **eficiencia computacional** ya que ambos componentes operan en paralelo sobre la misma representación de estado.

La **consistencia de decisiones** mejora al combinar la estabilidad heurística con la adaptabilidad neural, resultando en mejor balance entre objetivos inmediatos (supervivencia) y a largo plazo (maximización de puntuación). El sistema demuestra **capacidad de generalización** mejorada, manejando situaciones no vistas durante entrenamiento mediante la sinergia entre conocimiento explícito y patrones aprendidos.

#### Funcionamiento del Pipeline Completo

El sistema integrado opera como un **pipeline coherente**: el estado del juego se convierte una sola vez en representación matricial que alimenta tanto la ordenación neural (optimizando AlphaBeta) como la evaluación híbrida (refinando calidad de decisiones). Esta arquitectura unificada maximiza eficiencia computacional mientras proporciona dos niveles complementarios de inteligencia neural.

La integración representa un equilibrio óptimo entre **garantía algorítmica** (AlphaBeta asegura exploración sistemática) y **adaptabilidad aprendida** (redes neuronales aportan experiencia contextual), resultando en un sistema más robusto, eficiente y efectivo que supera significativamente las aproximaciones tradicionales en el dominio complejo de Pacman.



tras la ejecuccion de de 10 seeds distintas hemos obtenido los siguientes resultados:


| Métrica      | Seed 0 | Seed 10 | Seed 15 | Seed 100 | Seed 130 | Seed 150 | Seed 195 | Seed 200 | Seed 205 | Seed 360 | **TOTAL** |
|--------------|--------|---------|---------|----------|----------|----------|----------|----------|----------|----------|-----------|
| **Average Score** | 1388.4 | 1320.2  | 1390.9  | **1493.3** | **1448.3** | 1328.8   | 1391.8   | 1328.8   | 1346.9   | **1520.1** | **1405.7** |
| **Win Rate**      | 60%    | 60%     | 50%     | **70%**    | **70%**    | 50%      | 50%      | 50%      | 50%      | **70%**    | **58%** |


Podemos observar que lejos de ser un agente perfecto, hemos conseguido mejorar las capacidades del modelo 