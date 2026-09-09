# ecg-pytorch · Detección de arritmias mediante redes neuronales profundas

Reimplementación en **PyTorch** de `awni/ecg`, el código fuente abierto que acompaña al artículo:

> **Cardiologist-Level Arrhythmia Detection and Classification in Ambulatory Electrocardiograms Using a Deep Neural Network** — Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C., Turakhia, M. P., & Ng, A. Y. — *Nature Medicine*, 2019.

El repositorio original está escrito en **Keras/TensorFlow (Python 2)**. Este proyecto traslada la **misma arquitectura y el mismo pipeline de datos** a PyTorch moderno, y añade una **aplicación web interactiva** (Flask + Plotly) para clasificar señales de ECG de una sola derivación.

## Características

- **Réplica 1:1 de la red** del artículo (34 capas, 16 bloques residuales, pre-activación BatchNorm+ReLU, Dropout 0.2) portada a `torch.nn`.
- **Pipeline de datos idéntico**: truncado a múltiplos de 256 muestras, normalización global, una predicción por cada 256 muestras.
- **Entrenamiento fiel al original**: Adam con `clipnorm`, `ReduceLROnPlateau`, `EarlyStopping` y checkpoint por época.
- **Evaluación** a nivel de registro e intervalo, con la **métrica oficial del challenge** (`Challenge-F1` sobre N/A/O) además de macro-F1. Exporta un `metrics.json` legible para la app.
- **Aplicación web** (Flask + Plotly) con gráfico interactivo, selector de checkpoint, drag-and-drop y exportación de resultados.
- **Detalle técnico** en la web: **tabla de métricas reales** (acc, macro-F1, weighted-F1, Challenge-F1 + precisión/recall/F1 por clase), **tabla de hiperparámetros vs. el artículo** (reproducibilidad) y clasificación por tramos **agrupada** en tabla.
- **Re-muestreo automático**: cualquier señal de entrada se relocaliza a 300 Hz antes de inferir.
- **PyTorch con CUDA** (soporta GPU) y detección automática de dispositivo.
- Herramientas de datos sintéticos para pruebas rápidas sin dataset.

## Decisión sobre el conjunto de datos

El artículo tiene **dos experimentos**:

1. **Resultado principal** (AUC ≈ 0.97): un clasificador de 12 clases entrenado sobre **91,232 registros privados de iRhythm** (una derivación, 200 Hz, 30 s). Estos datos **no son públicos**.
2. **Experimento de generalización**: el artículo también entrena la misma red sobre el **dataset público de PhysioNet CinC 2017** (n = 8,528) y evalúa sobre el test oculto (n = 3,658), reportando **F1 promedio de clases = 0.83**.

Este proyecto reproduce el **experimento de generalización a CinC 2017**, no el resultado principal de 12 clases, porque:

| Aspecto | Artículo (iRhythm) | CinC 2017 (este proyecto) | CinC 2020 (descartado) |
|---|---|---|---|
| Derivaciones | 1 | 1 | 12 |
| Frecuencia | 200 Hz | 300 Hz | 500 Hz |
| Duración | 30 s | 30–60 s | 10 s |
| Tarea | Ritmo (12 clases) | Ritmo (4–5 clases) | Diagnóstico multi-etiqueta |
| Métrica | AUC, F1 por clase | F1 por clase (N/A/O) | *challenge metric* |

Elegir CinC 2017 permite mantener **una sola derivación, una etiqueta por registro, salida por intervalo de 256 muestras y softmax categórica**, que es exactamente el diseño del artículo. CinC 2020, por el contrario, exigiría 12 canales de entrada y pérdida multi-etiqueta (BCE), alejándose de la réplica.

## Arquitectura

```
Entrada (B, 1, T)                     # B registros, 1 derivación, T muestras
   │
   ├─ Conv1d(k=16, stride=1, SAME) → BatchNorm → ReLU         32 canales
   ├─ 16 × bloques residuales, submuestreo [1,2,1,2,…,1,2]
   │     └─ shortcut: MaxPool1d(subsample) + zero-pad de canales
   │         cuando el nº de canales se duplica (cada 4 bloques)
   │     └─ rama residual: 2 × [BatchNorm → ReLU → Conv1d(k=16)]
   │     └─ canales: 32 → 64 → 128 → 256  (×2 cada 4 bloques)
   ├─ BatchNorm → ReLU
   └─ Linear(num_clases) → softmax     ⇒  Salida (B, T/256, num_clases)
```

- **Entrada**: señal cruda de ECG (una derivación), sin características manuales ni metadatos de paciente.
- **Bloques principales**: 16 bloques residuales de 2 capas convolucionales, con shortcut y padding `SAME`. La primera y la última capa se tratan de forma especial por la estructura de pre-activación.
- **Salida**: una distribución softmax por cada **256 muestras** (≈0.85 s a 300 Hz); el factor de submuestreo temporal es 2⁸ = 256.
- **Hiperparámetros** (`examples/cinc17/config.json`): `conv_filter_length=16`, `conv_num_filters_start=32`, `conv_dropout=0.2`, `conv_num_skip=2`, `conv_increase_channels_at=4`, `learning_rate=0.001`, `batch_size=32` (ver [Fidelidad](#fidelidad-respecto-al-artículo)).

> Parámetros del modelo: ~10.5 M.

## Correspondencia con la implementación original

| Keras / TF (`awni/ecg`) | PyTorch (este proyecto) |
|---|---|
| `Conv1D(padding='same')` | `nn.Conv1d` + padding `SAME` asimétrico |
| `MaxPooling1D(pool_size=s)` | `nn.MaxPool1d(s)` + padding `SAME` |
| `BatchNormalization` + `ReLU` | `nn.BatchNorm1d(eps=1e-3, momentum=0.99)` + `nn.ReLU` |
| `Dropout` | `nn.Dropout` |
| `TimeDistributed(Dense)` + softmax | `nn.Linear` sobre `(B,T,C)` + `nn.Softmax` |
| Categorical crossentropy | `nn.CrossEntropyLoss` sobre logits aplanados |
| `Adam(clipnorm=1)` | `torch.optim.Adam(eps=1e-7)` + `clip_grad_norm_` |
| `ReduceLROnPlateau` / `EarlyStopping` | LR en plateau + early stopping (custom) |
| `ModelCheckpoint .hdf5` | `torch.save` `.pt` en cada época |

**Convenciones de forma**: Keras usa `(batch, tiempo, canales)`; PyTorch, `(batch, canales, tiempo)`. La red lo gestiona internamente, por lo que las interfaces siempre reciben y devuelven `(B, 1, T)`.

## Instalación

Requisitos: Python **3.13.5**, PyTorch con CUDA 13.0 (RTX 3050 Laptop, Ampere / sm_86).

```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verifica que CUDA es visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
# esperado → 2.14.0+cu130 True 13.0
```

> El `requirements.txt` fija `torch==2.14.0+cu130` mediante el tag de versión local, forzando el build CUDA en lugar del build CPU de PyPI.

## Obtención de datos

El dataset **PhysioNet CinC 2017** está restringido (requiere cuenta y aceptar el acuerdo de uso). Descarga desde [physionet.org/content/challenge-2017/1.0.0/](https://physionet.org/content/challenge-2017/1.0.0/):

- `training2017.zip` — registros de entrenamiento.
- `REFERENCE-v3.csv` — etiquetas por registro.

Colócalos en `dataset2017/` (o `training2017/` + `REFERENCE-v3.csv`) y genera los conjuntos:

```bash
python examples/cinc17/build_datasets.py \
  --data_dir   dataset2017/training2017 \
  --label_file dataset2017/REFERENCE-v3.csv \
  --out_dir    examples/cinc17 \
  --relative            # rutas relativas (portable entre máquinas)
  --stratify            # opcional: split proporcional a la clase
```

Esto produce `examples/cinc17/train.json` y `dev.json` (formato JSONL, un objeto por línea con `ecg` y `labels`). Por defecto el split es aleatorio 90/10; usa `--stratify` para que las clases minoritarias (`A`, `~`) queden representadas en `dev` a la misma tasa que en `train`.

## Entrenamiento

```bash
# desde la raíz del proyecto
python -m ecg.train examples/cinc17/config.json -e cinc17
```

Los checkpoints se guardan **en cada época** en `saved/<experimento>/<timestamp>/<val_loss>-<val_acc>-<epoch>-<loss>-<acc>.pt`, junto con el preprocesador (`preproc.bin`). El mejor modelo es el de **menor `val_loss`** (primer número del nombre de archivo).

## Inferencia o uso

Predicción por registro (voto mayoritario sobre los intervalos):

```bash
python -m ecg.predict examples/cinc17/dev.json "saved/cinc17/<timestamp>/0.408-0.862-011-0.274-0.904.pt"
```

Evaluación formal con métricas (impresión en consola + matriz de confusión):

```bash
python examples/cinc17/evaluate.py --data_json examples/cinc17/dev.json --saved saved
```

Para exportar también las **figuras (matriz de confusión, F1 por clase) y el `metrics.json`** que la app web lee en la pestaña *Detalle Técnico*:

```bash
python examples/cinc17/evaluate.py --data_json examples/cinc17/dev.json --saved saved \
  --save_metrics_dir webapp/static/metrics
```

> La pestaña *Detalle Técnico* de la app muestra automáticamente la tabla de métricas cuando existe `webapp/static/metrics/metrics.json` (endpoint de diagnóstico en `GET /metrics`).

## Aplicación web

Interfaz interactiva para clasificar señales de una sola derivación, con:

- **Gráfico Plotly interactivo** (zoom/pan, tooltip, fondo tipo "papel milimétrico" con cuadrícula y escala adaptativa). El JS de Plotly está **embebido localmente**, sin CDN.
- **Detalle Técnico**: tabla de **métricas de evaluación reales** (por registro y por clase), **hiperparámetros vs. el artículo**, y **clasificación por tramos agrupada** (filas consecutivas del mismo ritmo se consolidan).
- **Selector de checkpoint** en la interfaz: cambia de modelo **sin reiniciar el servidor**.
- **Botón "Probar con ejemplo"** (genera y clasifica una señal sin subir nada).
- **Drag-and-drop** del archivo y campos de etiqueta real (comparación ✔/✘).
- **Informe PDF real** que se descarga automáticamente (ReportLab), no un diálogo de impresión del navegador.
- **Re-muestreo automático** a 300 Hz y aviso en pantalla si se sube a otra frecuencia.

Acepta **CSV** (una derivación), **`.mat`**, **`.dat`** y **`.npy`**:

```bash
# desarrollo
python webapp/app.py --saved saved
# con comparación automática contra etiquetas reales de CinC2017:
python webapp/app.py --saved saved --reference dataset2017/REFERENCE-v3.csv
# abre http://127.0.0.1:5000/
# para ver desde otro dispositivo / WSL2:
python webapp/app.py --saved saved --host 0.0.0.0 --port 5000
```

Si se carga `REFERENCE-v3.csv`, la app compara automáticamente la predicción con la etiqueta real cuando el archivo subido conserva el ID oficial del registro, por ejemplo `A00001.mat`, `A00001.dat` o `A00001.csv`. También se puede consultar el estado desde `GET /reference`.

### Despliegue en producción

El servidor de desarrollo de Flask (`app.run`) **no** está pensado para producción. Usa un servidor WSGI:

```bash
# gunicorn (Linux/macOS/servidor)
pip install gunicorn
ECG_SAVED="saved" gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 "webapp.wsgi:app"

# waitress (Windows nativo)
pip install waitress
waitress-serve --listen=*:5000 "webapp.wsgi:app"
```

`webapp/wsgi.py` lee las variables de entorno `ECG_SAVED` (carpeta de checkpoints) o `ECG_MODEL` (ruta exacta) y **carga el modelo una sola vez al arrancar** (compartido entre request). Se incluye un `Procfile` para despliegue en plataformas PaaS (Heroku/Render).

## Prueba rápida

Sin descargar el dataset, verifica el pipeline completo con señales sintéticas (no aprende nada real, solo comprueba que el código funciona):

```bash
python examples/cinc17/make_synthetic.py --train 80 --dev 40
python -m ecg.train examples/cinc17/config_synthetic.json -e synth --epochs 2
python examples/cinc17/evaluate.py --data_json examples/cinc17/synthetic/dev.json --saved saved_synthetic
```

## Réplica paso a paso (de cero a los resultados)

Secuencia completa de comandos para reproducir el experimento de CinC 2017, en orden, desde la raíz del proyecto (con el venv activado y CUDA visible).

**1 · Instalación**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source venv/bin/activate           # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# esperado → 2.14.0+cu130 True
```

**2 · Datos (CinC 2017, con acuerdo de uso)**

Descarga `training2017.zip` y `REFERENCE-v3.csv` desde [physionet.org/content/challenge-2017/1.0.0/](https://physionet.org/content/challenge-2017/1.0.0/) y colócalos en `dataset2017/`. Luego:

```bash
python examples/cinc17/build_datasets.py \
  --data_dir   dataset2017/training2017 \
  --label_file dataset2017/REFERENCE-v3.csv \
  --out_dir    examples/cinc17 \
  --relative --stratify
```

Produce `examples/cinc17/train.json` y `dev.json`.

**3 · Entrenamiento**

```bash
python -m ecg.train examples/cinc17/config.json -e cinc17
```

Los checkpoints se guardan en `saved/cinc17/<timestamp>/<val_loss>-<val_acc>-<epoch>-<loss>-<acc>.pt` (el mejor es el de menor `val_loss`).

**4 · Inferencia (una-predicción-por-registro por voto mayoritario)**

```bash
python -m ecg.predict examples/cinc17/dev.json \
  "saved/cinc17/<timestamp>/0.371-0.870-013-0.247-0.916.pt"
```

**5 · Evaluación formal + métricas reales para la web**

```bash
python examples/cinc17/evaluate.py --data_json examples/cinc17/dev.json --saved saved
# exportar figuras + metrics.json (para la pestaña Detalle Técnico):
python examples/cinc17/evaluate.py --data_json examples/cinc17/dev.json --saved saved \
  --save_metrics_dir webapp/static/metrics
```

**6 · Aplicación web**

```bash
# desarrollo
python webapp/app.py --saved saved --host 127.0.0.1 --port 5000
# abre http://127.0.0.1:5000/
# desde otro dispositivo / WSL2:
python webapp/app.py --saved saved --host 0.0.0.0 --port 5000
```

Con el checkpoint real cargado, la pestaña **Detalle Técnico** muestra la tabla de métricas (Challenge-F1 ≈ **0.860**) y la tabla de hiperparámetros vs. el artículo.

## Fidelidad respecto al artículo

### Qué se reproduce exactamente

- La **arquitectura** del artículo (34 capas, 16 bloques residuales, filtro 16, canales 32·2^k, submuestreo alterno, pre-activación, Dropout 0.2).
- El **pipeline** (una derivación, una etiqueta por registro, salida cada 256 muestras, softmax categórica).
- Los **hiperparámetros de optimización**: Adam `lr=1e-3`, `ReduceLROnPlateau(factor=0.1, patience=2, min_lr=lr*0.001)`, `EarlyStopping(patience=8)`, checkpoint en cada época.
- La **paridad numérica** Keras↔PyTorch: `BatchNorm(eps=1e-3, momentum=0.99)` y `Adam(eps=1e-7)`.
- El **experimento de generalización** sobre CinC 2017, tal como lo describe el propio artículo.

### Qué no puede reproducirse

- El **resultado principal** (AUC ≈ 0.97 sobre 12 clases) porque usa el dataset privado de iRhythm, que no es público.
- El **número 0.83** reportado por el artículo, porque se mide sobre el **test set oculto de 3,658 registros** evaluado por el sistema oficial de PhysioNet.

### Diferencias inevitables

- El dataset CinC 2017 usa **300 Hz** (no 200 Hz), por lo que un intervalo de 256 muestras representa ≈0.85 s (no ≈1.28 s).
- El número de clases se deriva del vocabulario de los datos (`A, N, O, ~`, a veces `|`), en lugar de las 12 clases fijas del artículo.
- `batch_size` es **32**, igual que `awni/ecg` (el artículo usa 128 para su build con datos privados). El resto de hiperparámetros coincide.
- **`dev.json` se usa para la selección de checkpoint (early stopping)**, por lo que evaluar sobre él introduce *leakage* de selección de modelo. Para una evaluación imparcial, descarga el **test set oficial** de PhysioNet y evalúa sobre `test.json`.

### Métricas comparables con el artículo

Métricas obtenidas sobre el `dev.json` local (sujetas a la limitación de la sección anterior). **No son comparables directamente con el 0.83** (que es sobre el test oculto):

| Métrica | Nivel registro | Nivel intervalo |
|---|---|---|
| Exactitud | 0.875 | 0.871 |
| Macro-F1 (todas las clases) | 0.764 | 0.760 |
| Weighted-F1 | 0.871 | 0.869 |
| **Challenge-F1 (solo N/A/O, oficial)** | **0.860** | 0.856 |

Reporte por clase (nivel registro, `n = 854`):

| Clase | Precisión | Recall | F1 | n_real | n_pred |
|---|---|---|---|---|---|
| A (fibrilación auricular) | 0.800 | 0.895 | 0.845 | 76 | 85 |
| N (ritmo normal) | 0.922 | 0.931 | 0.927 | 508 | 513 |
| O (otro ritmo) | 0.810 | 0.810 | 0.810 | 242 | 242 |
| ~ (ruido) | 0.714 | 0.357 | 0.476 | 28 | 14 |

> El nivel intervalo es **solo un diagnóstico interno**: CinC 2017 carece de etiquetas reales por intervalo, por lo que el artículo reporta únicamente métrica a nivel de registro.

## Experimentos adicionales propuestos

Además de la réplica base, el repositorio deja preparados dos experimentos para fortalecer el trabajo sin cambiar el alcance de CinC2017. La metodología completa está en `docs/experimentos_cinc2017.md`.

### Experimento 2 · ResNet-34 vs CNN convencional

`examples/cinc17/config.json` entrena la ResNet-34 de la réplica. `examples/cinc17/config_regular_cnn.json` activa `is_regular_conv=true` y construye una CNN convencional emparejada: misma profundidad, filtros, submuestreo, dropout y crecimiento de canales, pero **sin conexiones residuales**.

```bash
python -m ecg.train examples/cinc17/config.json -e cinc17_resnet --seed 2018
python -m ecg.train examples/cinc17/config_regular_cnn.json -e cinc17_cnn --seed 2018

python examples/cinc17/compare_models.py \
  --data_json examples/cinc17/dev.json \
  --resnet_saved saved/cinc17_resnet \
  --cnn_saved saved/cinc17_cnn \
  --out_dir results/cinc17/resnet_vs_cnn
```

El script genera `summary.csv`, `per_class_metrics.csv`, `comparison_metrics.json` y `comparison_report.md` con Accuracy, Macro-F1, Weighted-F1, F1 por clase, Challenge-F1, parámetros y tiempo de entrenamiento. La app Flask los muestra automáticamente en la pestaña **Experimentos** cuando existe `results/cinc17/resnet_vs_cnn/comparison_metrics.json`.

### Experimento 3 · Robustez frente a perturbaciones ECG

`examples/cinc17/robustness.py` toma el conjunto de evaluación y aplica perturbaciones reproducibles: ruido con SNR controlado, deriva de línea base, escalamiento de amplitud y recortes de duración.

```bash
python examples/cinc17/robustness.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet \
  --out_dir results/cinc17/robustness_resnet \
  --seed 1234
```

Las salidas (`robustness_summary.csv`, `robustness_per_class.csv`, `robustness_metrics.json`, `robustness_report.md` y opcionalmente `robustness_curves.png`) permiten construir una tabla del tipo: condición vs Accuracy, Macro-F1 y Challenge-F1. La app Flask los muestra automáticamente en **Experimentos** cuando existen `results/cinc17/robustness_resnet/robustness_metrics.json` o `results/cinc17/robustness_cnn/robustness_metrics.json`.

## Trabajo futuro

- Evaluación sobre el **test set oficial** de PhysioNet (una vez liberado y descargado) para obtener una métrica comparable al 0.83 sin *leakage*.
- Soporte para **CinC 2020** (12 derivaciones + multi-etiqueta) reutilizando el `_Conv1d` con canales de entrada y pérdida BCE.
- **Ponderación de clases** o ajuste del umbral para mejorar el recall de la clase `~` (ruido), actualmente la más débil.
- Exportar a **ONNX / TorchScript** para desplegar la inferencia fuera de Python.

## Estructura del repositorio

```text
.
├── ecg/
│   ├── __init__.py
│   ├── network.py       # CNN (ECGNetwork) portada 1:1
│   ├── load.py          # Preproc, generador y lectores de WFDB/.mat/.dat
│   ├── train.py         # bucle de entrenamiento (Adam, LR plateau, early stop)
│   ├── predict.py       # inferencia + clase mayoritaria por registro
│   └── util.py          # guardar/cargar preprocesador + helpers de modelo
├── examples/
│   └── cinc17/
│       ├── config.json            # hiperparámetros del artículo
│       ├── config_synthetic.json  # config para datos sintéticos
│       ├── build_datasets.py      # genera train.json / dev.json
│       ├── evaluate.py            # evaluación formal + export metrics.json y figuras
│       ├── make_synthetic.py      # señales sintéticas de prueba
│       └── setup.sh
├── webapp/
│   ├── app.py             # rutas Flask (/, /predict, /example, /models, /use_model, /metrics, /report.pdf)
│   ├── prediction.py      # PredictionService + resampling + parser + PNG/Plotly
│   ├── report_pdf.py      # informe PDF (ReportLab) que se descarga automáticamente
│   ├── make_sample_csv.py # CSV de ejemplo
│   ├── wsgi.py            # entrada WSGI para gunicorn/waitress (ECG_SAVED/ECG_MODEL)
│   ├── templates/index.html
│   ├── static/metrics/    # metrics.json + figuras exportadas por evaluate.py
│   └── static/            # style.css, app.js, plotly.min.js (embebido)
├── Procfile                       # despliegue PaaS (gunicorn)
├── requirements.txt
└── README.md
```

## Licencia

Distribuido bajo la licencia **GPL-3.0** (igual que el repositorio original `awni/ecg`). Consulta el archivo `LICENSE`.

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

Y, si cita el dataset, el artículo de CinC 2017 de PhysioNet:

```bibtex
@article{clifford2017af,
  title={AF Classification from a short single lead ECG recording: the
         PhysioNet/Computing in Cardiology Challenge 2017},
  author={Clifford, Gari D and Liu, Chengyu and Moody, Benjamin and others},
  journal={Computing in Cardiology},
  year={2017}
}
```
