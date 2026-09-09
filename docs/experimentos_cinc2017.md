# Diseño experimental sobre CinC2017

Este documento deja preparado el trabajo experimental adicional sin cambiar el problema original de CinC2017: una derivación, una etiqueta por registro y clasificación categórica con softmax.

## Objetivo principal

Replicar el modelo del paper de Hannun et al. sobre PhysioNet/CinC2017 y analizar si sus decisiones arquitectónicas y su comportamiento ante degradaciones de señal son consistentes en este repositorio.

## Uso del modelo dentro de la app Flask

La pestaña **Resultado** no mezcla modelos: usa únicamente el checkpoint que Flask tiene cargado en ese momento (`PredictionService`). Por tanto:

- si arrancas la app con `--saved saved/cinc17_resnet` o `--model <checkpoint_resnet.pt>`, las predicciones del ECG se hacen con **ResNet-34**;
- si arrancas la app con `--saved saved/cinc17_cnn` o `--model <checkpoint_cnn.pt>`, las predicciones se hacen con la **CNN convencional**;
- si arrancas con `--saved saved`, la app elige automáticamente el checkpoint con menor `val_loss` dentro de toda esa carpeta, por lo que podría cargar ResNet o CNN según cuál tenga menor pérdida.

La pestaña **Experimentos** solo muestra resultados ya calculados por `compare_models.py` y `robustness.py`; no cambia el modelo usado para clasificar el ECG cargado por el usuario.

### Comparación automática con etiqueta real del CSV

También puedes iniciar la app con el archivo oficial de etiquetas:

```bash
python webapp/app.py --saved saved/cinc17_resnet --reference dataset2017/REFERENCE-v3.csv
```

Cuando subas un ECG cuyo nombre conserve el ID oficial (`A00001.mat`, `A00001.dat`, `A00001.csv`, etc.), la app buscará esa fila en `REFERENCE-v3.csv` y mostrará en **Resultado**:

- etiqueta real del dataset;
- predicción del modelo;
- si coincide o no coincide;
- fuente de la etiqueta (`REFERENCE-v3.csv`).

Si el nombre del archivo no permite identificar el registro, todavía puedes escribir manualmente el diagnóstico conocido en el campo **Diagnóstico conocido (opcional)**.

## Experimento 1 · Réplica del modelo original

Entrena la arquitectura residual del paper con `examples/cinc17/config.json`.

```bash
python -m ecg.train examples/cinc17/config.json -e cinc17_resnet --seed 2018
python examples/cinc17/evaluate.py --data_json examples/cinc17/dev.json --saved saved/cinc17_resnet
```

Métricas recomendadas:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- F1 por clase.
- Challenge-F1 oficial de CinC2017: promedio de F1 en N/A/O.

## Experimento 2 · ResNet-34 vs CNN convencional

El repositorio ya tenía el parámetro `is_regular_conv`. Ahora se usa para construir una CNN convencional **emparejada** con la ResNet: mantiene profundidad, tamaños de filtro, patrón de submuestreo, dropout y crecimiento de canales, pero elimina las conexiones residuales.

### Entrenamiento

```bash
# Modelo A: ResNet-34 de la réplica
python -m ecg.train examples/cinc17/config.json \
  -e cinc17_resnet --seed 2018

# Modelo B: CNN convencional equivalente, sin shortcuts residuales
python -m ecg.train examples/cinc17/config_regular_cnn.json \
  -e cinc17_cnn --seed 2018
```

Ambos comandos usan:

- Mismo `train.json` y `dev.json`.
- Mismo split.
- Mismo preprocesamiento.
- Mismos hiperparámetros de entrenamiento.
- Misma semilla.
- Mismas métricas.

El entrenamiento guarda automáticamente `history.csv` y `training_summary.json` dentro de cada carpeta de checkpoint. Ahí quedan registrados el número de parámetros y el tiempo de entrenamiento.

### Comparación automática

```bash
python examples/cinc17/compare_models.py \
  --data_json examples/cinc17/dev.json \
  --resnet_saved saved/cinc17_resnet \
  --cnn_saved saved/cinc17_cnn \
  --out_dir results/cinc17/resnet_vs_cnn
```

Salidas:

- `results/cinc17/resnet_vs_cnn/summary.csv`
- `results/cinc17/resnet_vs_cnn/per_class_metrics.csv`
- `results/cinc17/resnet_vs_cnn/comparison_metrics.json`
- `results/cinc17/resnet_vs_cnn/comparison_report.md`

La app Flask muestra esta comparación automáticamente en la pestaña **Experimentos** cuando existe `results/cinc17/resnet_vs_cnn/comparison_metrics.json`.

Tabla esperada para el informe:

| Modelo | Parámetros | Tiempo entrenamiento | Accuracy | Macro-F1 | F1 por clase | Challenge-F1 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-34 | completar | completar | completar | completar | completar | completar |
| CNN convencional | completar | completar | completar | completar | completar | completar |

Pregunta que responde: **¿las conexiones residuales aportan una mejora real en la clasificación de ECG de CinC2017 bajo condiciones controladas?**

## Experimento 3 · Robustez frente a perturbaciones de ECG

La robustez se evalúa aplicando perturbaciones controladas al conjunto de evaluación, sin reentrenar el modelo.

Condiciones por defecto:

| Condición | Modificación |
|---|---|
| Original | ECG sin modificación |
| Ruido leve | Ruido gaussiano con SNR = 20 dB |
| Ruido moderado | Ruido gaussiano con SNR = 10 dB |
| Ruido fuerte | Ruido gaussiano con SNR = 5 dB |
| Baseline wander | Deriva senoidal de baja frecuencia |
| Amplitud baja/alta | Escalamiento ×0.5 y ×1.5 |
| Recorte | Recorte central a 10 s y 20 s |

### Evaluación de robustez

```bash
python examples/cinc17/robustness.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet \
  --out_dir results/cinc17/robustness_resnet \
  --seed 1234
```

Si también deseas comparar la robustez de la CNN convencional:

```bash
python examples/cinc17/robustness.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_cnn \
  --out_dir results/cinc17/robustness_cnn \
  --seed 1234
```

Salidas:

- `robustness_summary.csv`
- `robustness_per_class.csv`
- `robustness_metrics.json`
- `robustness_report.md`
- `robustness_curves.png` si `matplotlib` está instalado.

La app Flask muestra esta tabla automáticamente en **Experimentos** cuando existen `results/cinc17/robustness_resnet/robustness_metrics.json` o `results/cinc17/robustness_cnn/robustness_metrics.json`.

Tabla esperada para el informe:

| Condición | Accuracy | Macro-F1 | Challenge-F1 |
|---|---:|---:|---:|
| Original | completar | completar | completar |
| Ruido SNR 20 dB | completar | completar | completar |
| Ruido SNR 10 dB | completar | completar | completar |
| Ruido SNR 5 dB | completar | completar | completar |
| Baseline wander | completar | completar | completar |
| Amplitud ×0.5 | completar | completar | completar |
| Amplitud ×1.5 | completar | completar | completar |
| Recorte 10 s | completar | completar | completar |
| Recorte 20 s | completar | completar | completar |

Pregunta que responde: **¿qué tan sensible es el modelo a ruido, deriva de línea base, variaciones de amplitud y cambios de duración en ECG reales?**

## Recomendación metodológica

Para evitar sesgos, usa un `test.json` separado si lo tienes. Si solo usas `dev.json`, aclara en el informe que ese conjunto también participa en la selección de checkpoint por `val_loss`, por lo que las métricas son válidas como análisis interno, pero no como evaluación final independiente.
