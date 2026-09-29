# Sequential Recommendation with Learning Paths and Pedagogical Ontology Graphs (seq_emb)

Este proyecto implementa y compara modelos de recomendación secuencial de recursos pedagógicos para trayectorias de aprendizaje de estudiantes.

## Modelos Soportados

Cada arquitectura cuenta con dos modalidades de entrada:
1. **Con POG (*Pedagogical Ontology Graph*)**: Inicializa las representaciones de los ítems usando embeddings continuos pre-entrenados con **RotatE** sobre el grafo curricular de prerrequisitos y remediales.
2. **Sin POG (*noembed*)**: Aprende las representaciones de los ítems de forma integral (*end-to-end*) mediante una capa `nn.Embedding`.

Arquitecturas secuenciales:
- **CASER**: Modelo convolucional sobre secuencias (filtros horizontales y verticales).
- **GRU4Rec**: Red recurrente con unidades GRU y proyecciones lineales.
- **SASRec**: Transformer autorregresivo con Self-Attention multi-cabezal y positional embeddings.

## Estructura del Proyecto

```text
seq_emb/
├── configs/
│   └── default.yaml             # Configuración declarativa de hiperparámetros
├── data/
│   ├── raw/                     # JSONs de rutas, calificaciones, prerrequisitos, remediales
│   └── states/                  # know.pth (RotatE pre-entrenado)
├── src/seq_emb/
│   ├── core/                    # Semillas reproducibles y gestión de dispositivos
│   ├── data/                    # Loader tipado y generador de tensores
│   ├── models/                  # BaseSequentialModel, CaserModel, GRURecModel, SASRecModel
│   ├── recommender/             # Latent k-NN cosine recommendation
│   ├── evaluation/              # Ranking metrics (MAP, NDCG, HR, MRR) y Path metrics (DAG)
│   └── experiments/             # KFoldRunner estratificado y determinista
├── scripts/
│   ├── run_model.py             # CLI para entrenar y evaluar cualquier modelo individual
│   └── run_benchmark.py         # Ejecuta la comparativa completa (6 variantes)
└── pyproject.toml               # Configuración del paquete y dependencias
```

## Uso Rápido

### 1. Ejecutar un Modelo Individual
```bash
# CASER con POG (5 folds)
python scripts/run_model.py --model caser --pog

# CASER sin POG
python scripts/run_model.py --model caser --no-pog

# SASRec con POG
python scripts/run_model.py --model sasrec --pog

# GRU4Rec con POG
python scripts/run_model.py --model grurec --pog
```

### 2. Ejecutar el Benchmark Completo (6 Variantes)
```bash
python scripts/run_benchmark.py --epochs 500 --seed 42
```
Los resultados se imprimirán en una tabla estructurada y se almacenarán en `results/folds_full_path.json`.

### 3. Modo Ventana Local (Intervención Pedagógica)
Para evaluar las métricas de grafo únicamente sobre los pasos de la recomendación ($N=3$) en lugar de toda la trayectoria:
```bash
python scripts/run_benchmark.py --local-window
```

## Hyperparameter Tuning con Taguchi (L16 y L32b)

El módulo [`tuning`](src/seq_emb/tuning) permite explorar sistemáticamente el espacio de hiperparámetros mediante matrices ortogonales balanceadas de Taguchi, evaluando cada combinación mediante 5-Fold Cross Validation determinista.

### 1. Ejecutar el Tuning (Individual o Todos a la Vez)
```bash
# Lanzar el tuning de TODOS los modelos (CASER + GRU4Rec + SASRec)
python scripts/run_taguchi.py --model all --pog

# O para un modelo específico:
python scripts/run_taguchi.py --model caser --pog
python scripts/run_taguchi.py --model grurec --pog
python scripts/run_taguchi.py --model sasrec --pog

# Si deseas evaluar TODAS las variantes (con POG y sin POG):
python scripts/run_taguchi.py --model all --all-variants

# Ejecutar las corridas de Taguchi en paralelo (ej. 4 workers):
python scripts/run_taguchi.py --model caser --pog --jobs 4
```

### 2. Análisis Estadístico Taguchi y Frente de Pareto
Puedes ejecutar directamente el análisis estadístico sobre resultados ya existentes (individual o de todos):
```bash
# Analizar todos los modelos a la vez desde un directorio con CSVs:
python scripts/run_taguchi.py --model all --analyze-only results/tuning

# O analizar un modelo específico:
python scripts/run_taguchi.py --model caser --analyze-only results/tuning/caser_pog_fine_results.csv
```
El análisis calcula:
- Normalización Z-score multiobjetivo (invirtiendo métricas donde menor es mejor: `backward_ratio` y `repeat_ratio`).
- Descomposición de varianza ANOVA por factor (% de varianza explicada).
- Identificación de los niveles óptimos por factor.
- Detección estricta de las soluciones no dominadas en el **Frente de Pareto**.
- Exportación automática a:
  - `results/tuning/{model}_analysis_results_with_zscores.csv`
  - `results/tuning/{model}_taguchi_best_levels.csv`
  - `results/tuning/{model}_pareto_front_runs.csv`

