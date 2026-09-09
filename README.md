# ecg-pytorch · Clasificación de arritmias en ECG con PyTorch

Reimplementación en **PyTorch** de `awni/ecg`, el código abierto asociado al artículo:

> **Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory Electrocardiograms Using a Deep Neural Network** — Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C., Turakhia, M. P., & Ng, A. Y. — *Nature Medicine*, 2019.

El proyecto reproduce el experimento público sobre **PhysioNet/CinC 2017** y añade una aplicación web Flask para inferencia, visualización, comparación con etiquetas reales y reporte técnico.

---

## Qué incluye este repositorio

- **Réplica ResNet-34 del paper**: red convolucional profunda de 34 capas, 16 bloques residuales, pre-activación BatchNorm+ReLU, dropout y salida por intervalos.
- **CNN convencional equivalente**: misma profundidad, filtros, submuestreo, dropout y crecimiento de canales, pero **sin conexiones residuales**, para estudiar el aporte real de los shortcuts.
- **Pipeline CinC2017**: una derivación, una etiqueta por registro, normalización global, truncado a múltiplos de 256 muestras y predicción por intervalo.
- **Entrenamiento reproducible**: `--seed`, Adam, `clipnorm`, reducción de learning rate en plateau, early stopping y checkpoints por época.
- **Métricas formales**: Accuracy, Macro-F1, Weighted-F1, F1 por clase y **Challenge-F1 oficial CinC2017** sobre N/A/O.
- **Registro experimental**: cada entrenamiento guarda `history.csv` y `training_summary.json` con tiempo, parámetros y curvas por época.
- **Experimento ResNet vs CNN**: script automático de comparación y generación de tablas para el informe.
- **Experimento de robustez**: evaluación ante ruido, deriva de línea base, escalamiento de amplitud y recortes de duración.
- **App Flask interactiva**: carga de ECG, gráfico Plotly, predicción por tramos, PDF, métricas reales, pestaña de experimentos y comparación automática con `REFERENCE-v3.csv`.

---

## Decisión sobre el conjunto de datos

El artículo de Hannun et al. tiene dos partes:

1. Un resultado principal sobre un conjunto privado de iRhythm con 91,232 registros y 12 clases.
2. Un experimento de generalización sobre el dataset público **PhysioNet/CinC 2017**.

Este repositorio se centra en **CinC 2017**, porque permite mantener el mismo tipo de problema del pipeline original:

| Aspecto | Paper / iRhythm | Este proyecto / CinC2017 | CinC2020 |
|---|---:|---:|---:|
| Derivaciones | 1 | 1 | 12 |
| Frecuencia | 200 Hz | 300 Hz | 500 Hz |
| Duración | 30 s | 30–60 s | 10 s |
| Etiquetado | una clase por registro | una clase por registro | multi-etiqueta |
| Salida | softmax categórica | softmax categórica | multi-label / BCE |

CinC2020 no se usa en esta fase porque obligaría a cambiar el problema a 12 derivaciones y multi-label, alejándose de la réplica.

---

## Arquitecturas

### Modelo A · ResNet-34 de la réplica

```text
Entrada (B, 1, T)
  ├─ Conv1d(k=16, stride=1, SAME) → BN → ReLU                 32 canales
  ├─ 16 bloques residuales, submuestreo [1,2,1,2,...,1,2]
  │    ├─ rama residual: 2 convoluciones por bloque
  │    └─ shortcut: MaxPool + zero-padding cuando duplican canales
  ├─ BN → ReLU
  └─ Linear(num_clases) → softmax por intervalo
```

### Modelo B · CNN convencional equivalente

Activada con:

```json
"is_regular_conv": true
```

en `examples/cinc17/config_regular_cnn.json`.

Mantiene la misma profundidad y configuración de la ResNet, pero elimina los shortcuts residuales. Esto permite responder una pregunta experimental concreta:

> ¿Las conexiones residuales mejoran realmente la clasificación de ECG en CinC2017?

Ambos modelos tienen, con la configuración actual, aproximadamente:

```text
10,462,276 parámetros entrenables
```

---

## Instalación

Requisitos recomendados:

- Python 3.13.5.
- PyTorch con CUDA si se desea entrenar en GPU.
- `numpy`, `scipy`, `tqdm`, `flask`, `matplotlib`, `reportlab`.

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux/macOS
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verificación de PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Datos CinC2017

El dataset requiere cuenta de PhysioNet y aceptar el acuerdo de uso:

<https://physionet.org/content/challenge-2017/1.0.0/>

Descarga y coloca, por ejemplo, en `dataset2017/`:

```text
dataset2017/
├── REFERENCE-v3.csv
└── training2017/
    ├── A00001.mat
    ├── A00002.mat
    └── ...
```

Genera `train.json` y `dev.json`:

```bash
python examples/cinc17/build_datasets.py \
  --data_dir dataset2017/training2017 \
  --label_file dataset2017/REFERENCE-v3.csv \
  --out_dir examples/cinc17 \
  --relative \
  --stratify
```

Salidas:

```text
examples/cinc17/train.json
examples/cinc17/dev.json
```

> Nota: `REFERENCE-v3.csv` y los archivos `.mat/.dat` del dataset no deberían subirse al repositorio si el acuerdo de uso no lo permite.

---

## Entrenamiento

### Experimento 1 · Réplica ResNet-34

```bash
python -m ecg.train examples/cinc17/config.json \
  -e cinc17_resnet \
  --seed 2018
```

### Experimento 2 · CNN convencional equivalente

```bash
python -m ecg.train examples/cinc17/config_regular_cnn.json \
  -e cinc17_cnn \
  --seed 2018
```

Cada ejecución guarda checkpoints en:

```text
saved/<experimento>/<timestamp>/<val_loss>-<val_acc>-<epoch>-<loss>-<acc>.pt
```

y además:

```text
history.csv
training_summary.json
preproc.bin
```

El mejor checkpoint se selecciona por el primer número del nombre del archivo: **menor `val_loss`**.

---

## Evaluación formal

Evaluar un checkpoint específico:

```bash
python examples/cinc17/evaluate.py \
  --data_json examples/cinc17/dev.json \
  --model_path saved/cinc17_resnet/<timestamp>/<checkpoint>.pt
```

O elegir automáticamente el mejor checkpoint dentro de una carpeta:

```bash
python examples/cinc17/evaluate.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet
```

Exportar métricas e imágenes para la app Flask:

```bash
python examples/cinc17/evaluate.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet \
  --save_metrics_dir webapp/static/metrics
```

La app muestra estos resultados en **Detalle Técnico** cuando existe:

```text
webapp/static/metrics/metrics.json
```

---

## Comparación ResNet-34 vs CNN convencional

Después de entrenar ambos modelos:

```bash
python examples/cinc17/compare_models.py \
  --data_json examples/cinc17/dev.json \
  --resnet_saved saved/cinc17_resnet \
  --cnn_saved saved/cinc17_cnn \
  --out_dir results/cinc17/resnet_vs_cnn
```

Salidas:

```text
results/cinc17/resnet_vs_cnn/summary.csv
results/cinc17/resnet_vs_cnn/per_class_metrics.csv
results/cinc17/resnet_vs_cnn/comparison_metrics.json
results/cinc17/resnet_vs_cnn/comparison_report.md
```

La app Flask muestra automáticamente esta comparación en la pestaña **Experimentos** cuando existe:

```text
results/cinc17/resnet_vs_cnn/comparison_metrics.json
```

### Última corrida local de referencia

En la ejecución local más reciente del proyecto:

| Modelo | Parámetros | Tiempo entrenamiento | Accuracy | Macro-F1 | Challenge-F1 |
|---|---:|---:|---:|---:|---:|
| ResNet-34 | 10,462,276 | 63.42 min | 0.8689 | 0.7653 | 0.8498 |
| CNN convencional | 10,462,276 | 15.95 min | 0.5937 | 0.1882 | 0.2509 |

Interpretación:

> Al quitar las conexiones residuales, una red igual de profunda pierde capacidad de entrenamiento y rendimiento. En esta corrida, la ResNet supera ampliamente a la CNN convencional.

---

## Robustez frente a perturbaciones ECG

Evaluar robustez de la ResNet:

```bash
python examples/cinc17/robustness.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet \
  --out_dir results/cinc17/robustness_resnet \
  --seed 1234
```

Opcionalmente, robustez de la CNN:

```bash
python examples/cinc17/robustness.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_cnn \
  --out_dir results/cinc17/robustness_cnn \
  --seed 1234
```

Salidas:

```text
robustness_summary.csv
robustness_per_class.csv
robustness_metrics.json
robustness_report.md
robustness_curves.png
```

La app Flask las muestra automáticamente en **Experimentos** cuando existen:

```text
results/cinc17/robustness_resnet/robustness_metrics.json
results/cinc17/robustness_cnn/robustness_metrics.json
```

### Última corrida local de robustez ResNet

| Condición | Accuracy | Macro-F1 | Challenge-F1 |
|---|---:|---:|---:|
| Original | 0.8689 | 0.7653 | 0.8498 |
| Ruido SNR 20 dB | 0.8689 | 0.7767 | 0.8451 |
| Ruido SNR 10 dB | 0.8173 | 0.6983 | 0.7815 |
| Ruido SNR 5 dB | 0.6768 | 0.4935 | 0.5761 |
| Baseline wander | 0.8724 | 0.7762 | 0.8488 |
| Amplitud ×0.5 | 0.8700 | 0.7554 | 0.8405 |
| Amplitud ×1.5 | 0.8618 | 0.7486 | 0.8432 |
| Recorte 10 s | 0.7916 | 0.6615 | 0.7320 |
| Recorte 20 s | 0.8466 | 0.7272 | 0.8233 |

Interpretación general:

> El modelo es relativamente estable ante ruido leve, baseline wander y cambios moderados de amplitud, pero se degrada claramente con ruido fuerte y recortes agresivos de duración.

---

## Aplicación web Flask

La app permite:

- subir ECG en CSV, `.mat`, `.dat` o `.npy`;
- seleccionar derivación si el archivo tiene múltiples canales;
- re-muestrear automáticamente a 300 Hz;
- ver el ECG con bandas de predicción por intervalo;
- descargar informe PDF;
- ver métricas reales del modelo;
- ver comparación ResNet vs CNN;
- ver resultados de robustez;
- comparar automáticamente la predicción contra la etiqueta real de `REFERENCE-v3.csv`.

### Ejecutar con ResNet

```bash
python webapp/app.py \
  --saved saved/cinc17_resnet \
  --reference dataset2017/REFERENCE-v3.csv \
  --host 127.0.0.1 \
  --port 5000
```

Abre:

```text
http://127.0.0.1:5000/
```

### Modelo usado en la pestaña Resultado

La pestaña **Resultado** usa únicamente el checkpoint cargado al iniciar Flask.

- Con `--saved saved/cinc17_resnet`, usa ResNet-34.
- Con `--saved saved/cinc17_cnn`, usa la CNN convencional.
- Con `--model <checkpoint.pt>`, usa exactamente ese checkpoint.
- Con `--saved saved`, elige automáticamente el checkpoint con menor `val_loss` en toda la carpeta, por lo que puede cargar cualquiera de los modelos disponibles.

Para demostraciones de la réplica principal, se recomienda usar:

```bash
python webapp/app.py --saved saved/cinc17_resnet --reference dataset2017/REFERENCE-v3.csv
```

### Comparación con etiquetas reales del CSV

Si se pasa `--reference dataset2017/REFERENCE-v3.csv`, la app compara automáticamente cuando el archivo subido conserva el ID oficial:

```text
A00001.mat  -> busca A00001 en REFERENCE-v3.csv
A00001.dat  -> busca A00001 en REFERENCE-v3.csv
A00001.csv  -> busca A00001 en REFERENCE-v3.csv
```

En **Resultado** se muestra:

```text
Etiqueta real: Normal (N)
Predicción: Normal (N)
✔ Predicción correcta
```

Si el archivo no conserva el ID oficial, puedes escribir manualmente la etiqueta en el campo **Diagnóstico conocido**.

---

## Endpoints principales de la app

| Ruta | Método | Uso |
|---|---:|---|
| `/` | GET | Interfaz principal. |
| `/predict` | POST | Clasifica un ECG subido. |
| `/example` | POST | Genera y clasifica una señal sintética. |
| `/report.pdf` | POST | Genera el informe PDF. |
| `/models` | GET | Lista checkpoints disponibles. |
| `/use_model` | POST | Cambia checkpoint activo sin reiniciar. |
| `/metrics` | GET | Devuelve métricas exportadas por `evaluate.py`. |
| `/experiments` | GET | Devuelve resultados de comparación y robustez. |
| `/reference` | GET | Estado de `REFERENCE-v3.csv` y búsqueda de etiquetas. |

Ejemplo:

```text
http://127.0.0.1:5000/reference?record=A00004
```

---

## Prueba rápida sin dataset real

Para verificar que el pipeline funciona sin descargar CinC2017:

```bash
python examples/cinc17/make_syntethic.py --train 80 --dev 40
python -m ecg.train examples/cinc17/config_syntethic.json -e synth --epochs 2
python examples/cinc17/evaluate.py \
  --data_json examples/cinc17/synthetic/dev.json \
  --saved saved/synth
```

Estas señales son falsas y solo sirven como prueba técnica del código.

---

## Estructura del repositorio

```text
.
├── ecg/
│   ├── __init__.py
│   ├── load.py                  # carga, padding, normalización y lectores ECG
│   ├── network.py               # ResNet-34 y CNN convencional equivalente
│   ├── predict.py               # inferencia desde checkpoints
│   ├── train.py                 # entrenamiento + logs reproducibles
│   └── util.py                  # utilidades de checkpoints/parámetros
├── examples/
│   └── cinc17/
│       ├── build_datasets.py
│       ├── compare_models.py
│       ├── config.json
│       ├── config_regular_cnn.json
│       ├── config_syntethic.json
│       ├── evaluate.py
│       ├── make_syntethic.py
│       └── robustness.py
├── webapp/
│   ├── app.py                   # Flask + rutas principales
│   ├── prediction.py            # servicio de inferencia y gráficos
│   ├── report_pdf.py            # generación del PDF
│   ├── wsgi.py                  # entrada WSGI; lee ECG_SAVED/ECG_MODEL/ECG_REFERENCE
│   ├── templates/index.html
│   └── static/
│       ├── app.js
│       ├── style.css
│       └── plotly.min.js
├── docs/
│   └── experimentos_cinc2017.md
├── requirements.txt
└── README.md
```

---

## Consideraciones metodológicas

- `dev.json` puede haber participado en la selección del checkpoint por `val_loss`; para una evaluación final imparcial usa un `test.json` separado si lo tienes.
- El `Challenge-F1` oficial de CinC2017 promedia F1 en **N/A/O** y excluye la clase `~`.
- Las métricas impresas durante entrenamiento no son idénticas a las métricas formales de `evaluate.py` o `compare_models.py`, porque estas últimas trabajan a nivel de registro mediante voto mayoritario.
- La comparación automática con `REFERENCE-v3.csv` en la app depende del nombre del archivo subido; si el ID oficial no está en el nombre, usa el campo manual de diagnóstico conocido.

---

## Trabajo futuro

- Evaluar sobre un conjunto test independiente para evitar leakage de selección de checkpoint.
- Probar regularización o estrategias de optimización adicionales para la CNN convencional.
- Ajustar ponderación de clases o sampling para mejorar la clase `~`.
- Exportar el modelo a ONNX/TorchScript.
- Extender a datasets multi-derivación y multi-etiqueta en una línea de trabajo separada.

---

## Licencia

Distribuido bajo licencia **GPL-3.0**, igual que el repositorio original `awni/ecg`. Consulta `LICENSE`.

---

## Cómo citar

Si utilizas este proyecto en investigación, cita el artículo original:

```bibtex
@article{hannun2019cardiologist,
  title={Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory
         Electrocardiograms Using a Deep Neural Network},
  author={Hannun, Awni Y and Rajpurkar, Pranav and Haghpanahi, Masoumeh and
          Tison, Geoffrey H and Bourn, Codie and Turakhia, Mintu P and Ng, Andrew Y},
  journal={Nature Medicine},
  volume={25},
  number={1},
  pages={65},
  year={2019},
  publisher={Nature Publishing Group}
}
```

Para el dataset CinC2017:

```bibtex
@article{clifford2017af,
  title={AF Classification from a short single lead ECG recording: the
         PhysioNet/Computing in Cardiology Challenge 2017},
  author={Clifford, Gari D and Liu, Chengyu and Moody, Benjamin and others},
  journal={Computing in Cardiology},
  year={2017}
}
```
