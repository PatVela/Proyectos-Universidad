# Experimento: ResNet-34 vs CNN convencional

Dataset de evaluación: `C:\Users\juerg\Investigacion\TIF-Biomedica\examples\cinc17\dev.json`

| Modelo | Arquitectura | Parámetros | Tiempo entrenamiento (min) | Accuracy | Macro-F1 | Weighted-F1 | Challenge-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-34 | ResNet-34 | 10,462,276 | 63.42 | 0.8689 | 0.7653 | 0.8651 | 0.8498 |
| CNN convencional | CNN convencional | 10,462,276 | 15.95 | 0.5937 | 0.1882 | 0.4453 | 0.2509 |

## F1 por clase

| Modelo | Clase | Precisión | Recall | F1 | n real | n pred |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-34 | A | 0.8592 | 0.8026 | 0.8299 | 76 | 71 |
| ResNet-34 | N | 0.9021 | 0.9429 | 0.9220 | 508 | 531 |
| ResNet-34 | O | 0.8059 | 0.7893 | 0.7975 | 242 | 237 |
| ResNet-34 | ~ | 0.7333 | 0.3929 | 0.5116 | 28 | 15 |
| CNN convencional | A | 0.0000 | 0.0000 | 0.0000 | 76 | 0 |
| CNN convencional | N | 0.5946 | 0.9961 | 0.7447 | 508 | 851 |
| CNN convencional | O | 0.3333 | 0.0041 | 0.0082 | 242 | 3 |
| CNN convencional | ~ | 0.0000 | 0.0000 | 0.0000 | 28 | 0 |

> Challenge-F1 corresponde al promedio oficial de F1 en N/A/O; la clase `~` se excluye del promedio oficial.
