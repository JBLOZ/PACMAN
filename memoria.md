## Introduccion

Dividiremos la memoria en 4 partes que corresponden a las 4 implementaciones que hemos realizado en esta practica

- Mejora de la red
- Integracion AlphaBeta con la red
- Heuristica aplicada a la red (Evaluacion de posibles estados)
- Entrenamientos mejorados

### Mejora de la red

El proyecto se empezó con una red neuronal multicapa perceptrón, lo cual investigando un poco en diversos foros nos dimos cuenta que no era una red eficiente en comparacion a una red neuronal convolucional en cuanto a datos espaciales hablamos, por lo tanto, decidimos implementarla. El problema principal con la red MLP original era que, al aplanar la matriz del mapa de juego para su entrada, se perdía la información crucial sobre la proximidad y las relaciones espaciales entre elementos clave como Pacman, los fantasmas y la comida. Una pared adyacente a Pacman, por ejemplo, no se diferenciaba estructuralmente de una al otro lado del mapa, dificultando la comprensión contextual del entorno. Esta nueva red convolucional trabaja con matrices 2d lo cual nos es muy util para que aprenda estrategias avanzadas en el espacio del mapa de pacman, tambien nos dimos cuenta de que mejoraba el rendimiento a la hora de entrenar el modelo con más de 50 partidas, investigando descubrimos que se debe a que este tipo de redes utilizan menos parametros y reducen la dimensionalidad a medida que la información viaja por las capas convolucionales.

En conclusión, hemos aprendido que usar redes convolucionales para este tipo de juegos, son mas eficientes que una multicapa perceptron, aunque son mas complejas, conseguimos una eficiencia y una mejora significativa respecto a la red original, lo cual nos ha servido para mejorar mucho los resultados finales.

### Integración AlphaBeta con la red

Nuestro sistema de inteligencia artificial para Pacman se distingue por implementar una  arquitectura híbrida de dos niveles , combinando de manera sinérgica el algoritmo de búsqueda AlphaBeta con el poder de las  redes neuronales convolucionales (CNN) . El objetivo es aprovechar tanto la garantía de optimalidad de los algoritmos de búsqueda clásicos como la capacidad de reconocimiento de patrones del aprendizaje profundo, resultando en un agente de Pacman más eficiente y efectivo.

#### Fundamentos: AlphaBeta y la importancia del ordenamiento

En el núcleo de nuestro agente se encuentra el algoritmo  **AlphaBeta** , una optimización del tradicional Minimax. AlphaBeta poda eficientemente ramas del árbol de búsqueda que no pueden mejorar la solución actual, empleando dos valores cruciales: **Alpha (**α**)** para el mejor valor encontrado hasta el momento por el maximizador (Pacman) y **Beta (**β**)** para el mejor valor del minimizador (los fantasmas). Una **poda** ocurre cuando **α**≥**β**, lo que permite descartar secciones enteras del árbol sin explorarlas, reduciendo drásticamente el tiempo de computación.

La eficacia de AlphaBeta depende en gran medida del  **orden en que se exploran los movimientos** . Si los movimientos más prometedores se evalúan primero, las podas ocurren de manera más temprana y frecuente, mejorando la eficiencia general de la búsqueda. Aquí es precisamente donde nuestra red neuronal aporta su primera y más significativa contribución.

#### Primera fase: Ordenación neural de movimientos

La red neuronal convolucional se integra como un **sistema de priorización inteligente** que organiza las acciones antes de que AlphaBeta inicie su exploración. Este proceso se descompone en los siguientes pasos:

1. **Representación del estado:** El tablero de juego actual de Pacman se transforma en una  **matriz numérica** . Cada elemento del entorno se codifica con un valor específico: paredes con 5.0, comida con 3.0, fantasmas normales con 2.0, fantasmas asustados con 6.0, Pacman con 1.0 y espacios vacíos con 0.0. Esta matriz se **normaliza por un factor de 6.0** para asegurar la estabilidad numérica y consistencia con los datos de entrenamiento de la red.
2. **Predicción neural:** La matriz de estado preprocesada es alimentada a la red neuronal convolucional. Gracias a sus capas convolucionales, el **Batch Normalization** y las capas de pooling, la red es capaz de extraer y comprender patrones espaciales tanto a nivel local como global del mapa. La salida de la red son las **probabilidades** para cada una de las cinco acciones posibles de Pacman ('Stop', 'North', 'South', 'East', 'West'), reflejando el conocimiento adquirido a través del entrenamiento con miles de partidas.
3. **Ordenación inteligente:** Utilizando estas probabilidades predichas, las acciones legales disponibles para Pacman se  **ordenan de mayor a menor probabilidad** . Esto asegura que el algoritmo AlphaBeta explore primero aquellos movimientos que, según la inteligencia de la red neuronal, tienen más probabilidades de conducir a un buen resultado.

Esta estrategia de ordenación neural se aplica de manera crítica en el **nivel raíz de la búsqueda** (la primera decisión de Pacman) y, de forma recursiva, en **todos los nodos maximizadores** dentro del árbol de búsqueda. Esta aplicación sistemática optimiza la ocurrencia de podas, acelerando considerablemente el proceso de decisión en cada turno.

#### Segunda fase: Evaluación híbrida de estados

Mientras que la ordenación neural se enfoca en la **eficiencia de la búsqueda** del algoritmo AlphaBeta, la segunda fase se centra en perfeccionar la  **calidad de la evaluación de los estados del juego** . Esta fase es crucial porque la precisión de las evaluaciones de los nodos hoja (o de los nodos de corte profundo) determina la calidad de la decisión final del agente. Una evaluación robusta permite a AlphaBeta elegir el mejor camino incluso con profundidades de búsqueda limitadas. Nuestra aproximación combina inteligentemente dos componentes sobre la misma representación matricial del estado:

1. **Evaluación basada en heurísticas (evaluación tradicional):**

   Este componente se encarga de calcular una puntuación de utilidad para un estado dado basándose en un conjunto de  **características heurísticas predefinidas y expertas** . Estas características cuantifican aspectos críticos del estado del juego desde la perspectiva de Pacman. Las métricas incluyen:

   * **Puntuación actual del juego:** El valor numérico de la puntuación acumulada de Pacman.
   * **Distancia a la comida más cercana:** Una heurística vital que guía a Pacman hacia su objetivo principal, incentivando la recolección de puntos y la progresión. Se utiliza la distancia de Manhattan o una búsqueda de caminos para encontrar la comida más próxima.
   * **Número de cápsulas de poder restantes:** Las cápsulas son un recurso estratégico. Reducir su número (indicando que Pacman las consume) generalmente se considera beneficioso, ya que permite a Pacman cazar fantasmas.
   * **Distancia a los fantasmas activos/asustados:** Esta es una característica crítica para la supervivencia y la agresión. Pacman debe maximizar la distancia a los fantasmas activos (peligrosos) y minimizar la distancia a los fantasmas asustados (cazables).
   * **Consideraciones por quedarse quieto:** Penalizar el movimiento "Stop" si existen otras opciones que mejoren la situación de Pacman, para evitar estancamientos innecesarios.

   Las heurísticas proporcionan una **base de conocimiento sólida y comprensible** sobre lo que constituye un buen o mal estado en Pacman. Son fundamentales porque codifican la lógica del juego que no siempre es trivial de aprender solo a través de datos. Permiten que el agente tome decisiones racionales incluso en escenarios poco comunes o cuando la red neuronal no ha visto suficientes ejemplos similares. Su diseño se basa en la experiencia humana y reglas explícitas, ofreciendo una capa de interpretabilidad y fiabilidad.
3. **Ajuste neural de la evaluación (predicción de utilidad adicional):**
   Este es el componente innovador que integra el aprendizaje profundo en la evaluación del estado. Además de las características heurísticas, se utiliza la misma **red neuronal convolucional** que en la fase de ordenación de movimientos para obtener una **predicción de utilidad cruda (logits)** del estado actual.

   **Funciona de la siguiente manera:**

   * La matriz del mapa del juego, ya preprocesada y normalizada, se pasa a la CNN.
   * En lugar de usar las salidas de la red directamente como acciones, o incluso como probabilidades de acción finales, se interpreta la **activación de la capa de salida (logits)** de una manera diferente. Estos logits representan una "puntuación" o "preferencia" que la red ha aprendido para cada acción posible en ese estado.
   * Aunque la red fue entrenada para predecir la  *acción óptima* , la magnitud de los logits para esa acción (o la distribución general de los logits) puede interpretarse como una **señal heurística adicional** de la "bondad" general del estado. Una alta confianza de la red en una acción deseable puede implicar que el estado actual es inherentemente favorable.

   Este componente añade una capa de **inteligencia aprendida** a la función de evaluación. Mientras que las heurísticas son estáticas y definidas por el programador, la predicción neural es  **dinámica y adaptable** . Permite a la función de evaluación capturar patrones sutiles y no lineales en el estado del juego que serían extremadamente difíciles o imposibles de codificar manualmente mediante heurísticas. Por ejemplo, la red podría haber aprendido que ciertas configuraciones de fantasmas y comida, aunque no inmediatamente obvias para una heurística simple, son muy prometedoras. Este enfoque **híbrido** capitaliza la robustez de las heurísticas tradicionales y la flexibilidad y el poder de descubrimiento de patrones de las redes neuronales, resultando en una evaluación más completa y matizada. La red "suaviza" y refina las evaluaciones heurísticas con una comprensión más global y basada en la experiencia.

Al combinar de esta forma la lógica estructurada de AlphaBeta con la capacidad de reconocimiento de patrones de las redes neuronales convolucionales, nuestro agente Pacman no solo mejora drásticamente la **eficiencia de su búsqueda** mediante la ordenación inteligente de movimientos, sino que también eleva la **calidad de sus decisiones** a través de una función de evaluación de estados más sofisticada. Esta **arquitectura híbrida** le permite a Pacman responder de manera más inteligente y estratégica a las dinámicas del juego, adaptándose a situaciones complejas y superando las limitaciones que tendrían los enfoques puramente heurísticos o puramente basados en el aprendizaje automático.

### Heurística aplicada a la red (evaluación de posibles estados)

La **evaluación híbrida** constituye el núcleo intelectual de nuestro sistema de IA para Pacman, fusionando de manera innovadora **heurísticas tradicionales probadas** con  **predicciones neuronales contextuales** . Este enfoque sinérgico ha permitido crear una función de evaluación que supera las limitaciones inherentes de cada método individual, proporcionando una comprensión más profunda y adaptativa del estado del juego.

#### Arquitectura de la evaluación dual

1. Componente heurístico tradicional

Este componente establece una **base sólida, transparente e interpretable** para la evaluación de cualquier estado del juego. Se basa en una **suma ponderada de factores críticos** que reflejan el conocimiento experto sobre los objetivos y peligros en Pacman.

* **Puntuación oficial del juego como fundamento:** La evaluación inicial se ancla en la **puntuación actual oficial** de `currentGameState.getScore()`, multiplicada por 5.

  * Al ponderar la puntuación base, se asegura que los objetivos a largo plazo del juego (ganar puntos, limpiar el mapa) tengan una **influencia significativa y persistente** en la evaluación, evitando que Pacman se centre únicamente en objetivos inmediatos y subóptimos. Esto proporciona una dirección global consistente.
* **Métricas de distancia manhattan para cercanía a la comida:** Se calcula la **distancia de Manhattan** desde la posición actual de Pacman hasta todas las piezas de comida restantes, y se selecciona la más cercana. La contribución a la evaluación es inversamente proporcional a esta distancia mínima, ponderada por `WEIGHT_FOOD` (12.0).

  * La distancia de Manhattan es la métrica adecuada para entornos de cuadrados como Pacman, ya que refleja el número real de movimientos necesarios para alcanzar un objetivo. Al priorizar la comida más cercana, se incentiva un  **comportamiento eficiente de recolección** , guiando a Pacman a través del laberinto para maximizar su puntuación. El peso `12.0` ha sido ajustado experimentalmente para que la atracción por la comida sea **suficientemente fuerte** como para ser un motor principal de movimiento, pero no tan dominante que ignore otros factores críticos.
* **Gestión sofisticada de fantasmas; Amenazas y oportunidades:** Este es el aspecto más complejo de la heurística, ya que los fantasmas pueden representar tanto un peligro mortal como una fuente de puntos. La función distingue inteligentemente entre estos dos estados:**Fantasmas Normales (Amenazas):** Se aplica un **peso negativo considerable** (`WEIGHT_GHOST = -30.0`), inversamente proporcional a la distancia del fantasma. Si se detecta una colisión inminente (distancia cero), se retorna una penalización de `-float('inf')`.

  * El peso negativo extremadamente alto para fantasmas normales (y la penalización infinita por colisión) garantiza la  **supervivencia como prioridad máxima** . El valor `-30.0` está calibrado para que la aversión a los fantasmas sea  **más fuerte que la atracción por la comida** , promoviendo un comportamiento defensivo y de evasión inteligente. Es crucial que Pacman evite la muerte a toda costa, y esta heurística lo codifica explícitamente.

  **Fantasmas asustados:** Se asigna un **peso positivo muy alto** (`WEIGHT_SCARED_GHOST = 150.0`), también inversamente proporcional a la distancia.**Justificación:** Los fantasmas asustados son una fuente temporal de  **puntos muy elevados** . El peso `150.0` es **significativamente mayor** que cualquier otro peso positivo para reflejar la prioridad de aprovechar estas ventanas de oportunidad. Esto incentiva un **comportamiento agresivo y de caza** cuando Pacman está potenciado, capitalizando al máximo las ventajas estratégicas del juego.

  2. Componente neural contextual

Este componente integra el **aprendizaje profundo** en la evaluación del estado, utilizando la misma **red convolucional (CNN)** previamente entrenada en la fase de ordenación de movimientos. Sus predicciones, que antes servían para ordenar acciones, ahora se reinterpretan como un **refinamiento contextual** de la evaluación heurística.

* **Interpretación como refinamiento contextual:** La CNN procesa la misma representación matricial del tablero. Sus salidas representan las "preferencias" o "puntuaciones" aprendidas para cada acción posible, se utilizan para ajustar la evaluación heurística.
  * Este enfoque permite a la red aportar una **adaptabilidad situacional** que las heurísticas fijas no pueden. A través de su entrenamiento en miles de partidas, la CNN ha aprendido a identificar **patrones espaciales y dinámicos complejos** que son difíciles o imposibles de codificar explícitamente en reglas heurísticas. Por ejemplo, la red puede reconocer una configuración de fantasmas y cápsulas que, aunque no esté cubierta por una regla simple de distancia, representa una oportunidad o peligro particular. Esto proporciona un **ajuste fino** basado en la experiencia, añadiendo una capa de "intuición" al agente.

#### Proceso de integración sinérgica

El sistema de evaluación funciona como una **pipeline coherente y eficiente** , donde ambos componentes operan en paralelo y sus resultados se combinan de forma inteligente.

1. **Evaluación secuencial y paralela:** Primero, se calcula la **puntuación heurística tradicional** para establecer una base sólida e interpretable. Simultáneamente, la red neuronal procesa la misma representación matricial del estado para generar sus **predicciones contextuales** .
   * Esta ejecución conjunta asegura que la evaluación siempre tenga una base lógica y robusta (heurísticas) complementada por la perspicacia de la IA.
2. **Combinación por suma ponderada calibrada:** Las puntuaciones de ambos componentes se combinan a través de una  **suma ponderada calibrada** .
   * Los **pesos heurísticos** (comida: 12.0, fantasmas normales: -30.0, fantasmas asustados: 150.0) fueron ajustados experimentalmente no solo para definir el comportamiento individual, sino también para que el **componente heurístico garantice un comportamiento mínimo aceptable** y la adhesión a las reglas fundamentales del juego. La **contribución neural** se calibró para **influir significativamente** en la decisión final y aportar ese refinamiento adaptativo, pero sin dominar por completo las garantías y la estabilidad proporcionadas por las heurísticas tradicionales. Esto evita comportamientos erráticos que podrían surgir de una dependencia exclusiva de un modelo de aprendizaje profundo, especialmente en situaciones no vistas durante el entrenamiento.

#### Ventajas del sistema integrado

Esta arquitectura dual ofrece un conjunto de ventajas significativas que elevan el rendimiento del agente:

* **Robustez mejorada:** La combinación de dos fuentes de inteligencia (heurísticas y red neuronal) proporciona una  **redundancia inherente** . Si uno de los componentes falla o encuentra una situación ambigua, el otro puede compensar, previniendo fallos catastróficos y asegurando un comportamiento estable.
* **Adaptabilidad contextual:** El componente neural permite un **refinamiento basado en patrones aprendidos** que las heurísticas fijas no pueden lograr. Esto habilita al agente a adaptarse a situaciones complejas y matizadas, donde la "mejor" acción podría depender de interacciones sutiles entre múltiples elementos del tablero.
* **Interpretabilidad mantenida:** Al conservar una fuerte base heurística tradicional, se mantiene un grado de **interpretabilidad** en el comportamiento del agente. Podemos entender por qué Pacman se acerca a la comida o huye de los fantasmas, incluso cuando la red neuronal aporta un ajuste fino.
* **Eficiencia computacional:** La optimización del pipeline permite que ambos componentes operen en paralelo o de manera optimizada sobre la  **misma representación de estado matricial** , maximizando la eficiencia computacional y minimizando el tiempo de procesamiento por turno.
* **Consistencia de decisiones:** La sinergia entre la estabilidad heurística y la adaptabilidad neural resulta en una  **consistencia mejorada en las decisiones** . Esto permite un balance más efectivo entre los objetivos inmediatos (como la supervivencia y la recolección de comida cercana) y los objetivos a largo plazo (como la maximización de la puntuación total y la limpieza del laberinto).
* **Capacidad de generalización:** El sistema demuestra una  **capacidad de generalización mejorada** . Al combinar el conocimiento explícito de las heurísticas con los patrones aprendidos por la red, el agente puede manejar de manera más efectiva situaciones no vistas durante el entrenamiento de la red, así como laberintos y configuraciones de juego diversas.

#### Funcionamiento del pipeline completa

El sistema integrado opera como una **pipeline coherente**: el estado del juego se convierte una sola vez en representación matricial que alimenta tanto la ordenación neural (optimizando AlphaBeta) como la evaluación híbrida (refinando calidad de decisiones). Esta arquitectura unificada maximiza eficiencia computacional mientras proporciona dos niveles complementarios de inteligencia neural.

La integración representa un equilibrio óptimo entre **garantía algorítmica** (AlphaBeta asegura exploración sistemática) y **adaptabilidad aprendida** (redes neuronales aportan experiencia contextual), resultando en un sistema más robusto, eficiente y efectivo que supera significativamente las aproximaciones tradicionales en el dominio complejo de Pacman.

tras la ejecuccion de de 10 seeds distintas hemos obtenido los siguientes resultados:

| Métrica                | Seed 0 | Seed 10 | Seed 15 | Seed 100         | Seed 130         | Seed 150 | Seed 195 | Seed 200 | Seed 205 | Seed 360         | **TOTAL**  |
| ----------------------- | ------ | ------- | ------- | ---------------- | ---------------- | -------- | -------- | -------- | -------- | ---------------- | ---------------- |
| **Average Score** | 1388.4 | 1320.2  | 1390.9  | **1493.3** | **1448.3** | 1328.8   | 1391.8   | 1328.8   | 1346.9   | **1520.1** | **1405.7** |
| **Win Rate**      | 60%    | 60%     | 50%     | **70%**    | **70%**    | 50%      | 50%      | 50%      | 50%      | **70%**    | **58%**    |

Como podemos observar en la tabla de resultados, nuestro agente no se presenta como un "agente perfecto" con una tasa de victorias del 100%, los datos demuestran una mejora significativa en las capacidades del modelo en comparación con un agente AlphaBeta sin las optimizaciones neuronales. Un average score de 1405.7 puntos y un win rate del 58% promedio en un entorno de juego tan dinámico y competitivo como Pacman, donde los fantasmas actúan como agentes adversarios, son indicadores de una estrategia de toma de decisiones robusta y efectiva.
