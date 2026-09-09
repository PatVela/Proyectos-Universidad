# Experimentos CinC2017

Esta carpeta contiene los scripts y configuraciones para reproducir el trabajo experimental sobre **PhysioNet/CinC 2017** usando el paquete `ecg/` del repositorio.

El objetivo es mantener el mismo problema del paper de Hannun et al. adaptado al dataset público:

- ECG de **una sola derivación**.
- Una etiqueta por registro.
- Clases principales: `A`, `N`, `O`, `~`.
- Salida del modelo cada **256 muestras**.
- Evaluación a nivel de registro mediante voto mayoritario sobre los intervalos.

---

## Archivos principales

| Archivo | Descripción |
|---|---|
| `build_datasets.py` | Genera `train.json` y `dev.json` desde los `.mat/.dat` y `REFERENCE-v3.csv`. |
| `config.json` | Configuración de la **ResNet-34** de la réplica. |
| `config_regular_cnn.json` | Configuración de la **CNN convencional equivalente**, sin conexiones residuales. |
| `evaluate.py` | Evaluación formal: Accuracy, Macro-F1, Weighted-F1, F1 por clase y Challenge-F1. |
| `compare_models.py` | Compara ResNet-34 vs CNN convencional con el mismo split y métricas. |
| `robustness.py` | Evalúa robustez ante ruido, baseline wander, cambios de amplitud y recortes. |
| `make_syntethic.py` | Genera señales sintéticas para prueba rápida del pipeline. |
| `config_syntethic.json` | Configuración para entrenamiento con datos sintéticos. |

---

## 1. Preparar el dataset

Descarga desde PhysioNet:

<https://physionet.org/content/challenge-2017/1.0.0/>

Estructura recomendada:

```text
dataset2017/
├── REFERENCE-v3.csv
└── training2017/
    ├── A00001.mat
    ├── A00002.mat
    └── ...
```

Construye los JSONL usados por el entrenamiento:

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

Cada línea de estos archivos contiene:

```json
{"ecg": "ruta/al/registro.mat", "labels": ["N", "N", "N", "..."]}
```

La etiqueta del registro se repite por cada intervalo de 256 muestras, siguiendo el pipeline original.

---

## 2. Entrenar la réplica ResNet-34

Desde la raíz del repositorio:

```bash
python -m ecg.train examples/cinc17/config.json \
  -e cinc17_resnet \
  --seed 2018
```

Se guardan checkpoints en:

```text
saved/cinc17_resnet/<timestamp>/<val_loss>-<val_acc>-<epoch>-<loss>-<acc>.pt
```

Además se guardan:

```text
history.csv
training_summary.json
preproc.bin
```

`training_summary.json` incluye:

- arquitectura,
- número de parámetros,
- época con mejor `val_loss`,
- tiempo total,
- historial de pérdida y accuracy.

---

## 3. Entrenar la CNN convencional equivalente

```bash
python -m ecg.train examples/cinc17/config_regular_cnn.json \
  -e cinc17_cnn \
  --seed 2018
```

Esta CNN mantiene una configuración comparable con la ResNet:

- misma cantidad de capas convolucionales,
- mismo patrón de submuestreo,
- mismo tamaño de filtro,
- mismo dropout,
- mismo crecimiento de canales,
- pero **sin conexiones residuales**.

Esto permite aislar el efecto de los shortcuts residuales.

---

## 4. Evaluar un modelo

### Usar checkpoint exacto

```bash
python examples/cinc17/evaluate.py \
  --data_json examples/cinc17/dev.json \
  --model_path saved/cinc17_resnet/<timestamp>/<checkpoint>.pt
```

### Elegir automáticamente el mejor checkpoint de una carpeta

```bash
python examples/cinc17/evaluate.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet
```

### Exportar métricas para la app Flask

```bash
python examples/cinc17/evaluate.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet \
  --save_metrics_dir webapp/static/metrics
```

Esto genera:

```text
webapp/static/metrics/metrics.json
webapp/static/metrics/confusion_matrix.png
webapp/static/metrics/f1_per_class.png
```

La app Flask muestra esos archivos en la pestaña **Detalle Técnico**.

---

## 5. Comparar ResNet-34 vs CNN convencional

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

La app Flask lee automáticamente:

```text
results/cinc17/resnet_vs_cnn/comparison_metrics.json
```

y lo muestra en la pestaña **Experimentos**.

### Métricas de la última corrida local

| Modelo | Parámetros | Tiempo | Accuracy | Macro-F1 | Challenge-F1 |
|---|---:|---:|---:|---:|---:|
| ResNet-34 | 10,462,276 | 63.42 min | 0.8689 | 0.7653 | 0.8498 |
| CNN convencional | 10,462,276 | 15.95 min | 0.5937 | 0.1882 | 0.2509 |

---

## 6. Evaluar robustez

```bash
python examples/cinc17/robustness.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet \
  --out_dir results/cinc17/robustness_resnet \
  --seed 1234
```

Para la CNN convencional:

```bash
python examples/cinc17/robustness.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_cnn \
  --out_dir results/cinc17/robustness_cnn \
  --seed 1234
```

Condiciones por defecto:

| Condición | Descripción |
|---|---|
| `original` | ECG sin modificación. |
| `noise_snr_20p0db` | Ruido gaussiano leve. |
| `noise_snr_10p0db` | Ruido gaussiano moderado. |
| `noise_snr_5p0db` | Ruido gaussiano fuerte. |
| `baseline_wander` | Deriva senoidal de línea base. |
| `amplitude_x0p5` | Escalamiento de amplitud a 50%. |
| `amplitude_x1p5` | Escalamiento de amplitud a 150%. |
| `crop_10p0s` | Recorte central a 10 segundos. |
| `crop_20p0s` | Recorte central a 20 segundos. |

Salidas:

```text
results/cinc17/robustness_resnet/robustness_summary.csv
results/cinc17/robustness_resnet/robustness_per_class.csv
results/cinc17/robustness_resnet/robustness_metrics.json
results/cinc17/robustness_resnet/robustness_report.md
results/cinc17/robustness_resnet/robustness_curves.png
```

La app Flask muestra automáticamente los JSON de robustez en la pestaña **Experimentos**.

### Métricas de robustez ResNet de la última corrida local

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

---

## 7. Usar la app Flask con estos resultados

Para mostrar la réplica principal y comparar cada ECG con su etiqueta real:

```bash
python webapp/app.py \
  --saved saved/cinc17_resnet \
  --reference dataset2017/REFERENCE-v3.csv \
  --host 127.0.0.1 \
  --port 5000
```

La pestaña **Resultado** usa el checkpoint cargado por Flask. Si quieres evitar ambigüedad, usa `--saved saved/cinc17_resnet` o un `--model` exacto.

La app compara automáticamente con `REFERENCE-v3.csv` si el archivo subido conserva el ID oficial:

```text
A00001.mat
A00001.dat
A00001.csv
A00001_filtrado.csv
```

Si el nombre no contiene el ID, usa el campo manual **Diagnóstico conocido**.

---

## 8. Prueba rápida sin dataset real

```bash
python examples/cinc17/make_syntethic.py --train 80 --dev 40
python -m ecg.train examples/cinc17/config_syntethic.json -e synth --epochs 2
python examples/cinc17/evaluate.py \
  --data_json examples/cinc17/synthetic/dev.json \
  --saved saved/synth
```

Estas señales son sintéticas y no sirven para resultados científicos; solo verifican que el pipeline funcione.

---

## Notas metodológicas

- La métrica comparable con CinC2017 es **Challenge-F1**, promedio de F1 en `N`, `A` y `O`; la clase `~` se excluye del promedio oficial.
- Las métricas de entrenamiento (`val_acc`) no siempre coinciden con las métricas formales a nivel de registro.
- Si `dev.json` se usa para early stopping, reporta esa limitación en el informe. Para evaluación final usa un `test.json` independiente.
- La comparación ResNet vs CNN debe usar mismo dataset, split, preprocesamiento, semilla, hiperparámetros y métricas.
