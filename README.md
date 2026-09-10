# ECG CINC2020 — ResNet tipo Hannun con 12 clases SNOMED

Adaptación académica del enfoque de Hannun et al. (*Nature Medicine*, 2019) al dataset público **PhysioNet/Computing in Cardiology Challenge 2020**.

El dataset original de Hannun et al. pertenece a Zio Patch/iRhythm y no es público. Por eso esta implementación usa CINC2020, con ECG de 12 derivaciones y etiquetas SNOMED-CT. El problema se formula como **clasificación multilabel real**, porque un ECG puede contener más de un diagnóstico simultáneo.

---

## Estructura del proyecto

La organización separa paquete, ejemplos, webapp y documentación técnica:

```text
.
├── ecg/
│   ├── __init__.py
│   ├── load.py                  # carga, padding, normalización, lectores ECG y mapeo SNOMED
│   ├── network.py               # ResNet-34 tipo Hannun y CNN convencional equivalente
│   ├── predict.py               # inferencia desde checkpoints
│   ├── train.py                 # entrenamiento + logs reproducibles
│   └── util.py                  # checkpoints, parámetros, seeds y utilidades
├── examples/
│   └── cinc2020/
│       ├── build_datasets.py
│       ├── compare_models.py
│       ├── debug_record_prediction.py
│       ├── config.json
│       ├── config_regular_cnn.json
│       ├── config_syntethic.json
│       ├── evaluate.py
│       ├── make_syntethic.py
│       └── robustness.py
├── webapp/
│   ├── app.py                   # Flask + rutas principales
│   ├── prediction.py            # servicio de inferencia, conversión y gráficos
│   ├── project_info.py          # datos institucionales mostrados en la app
│   ├── report_pdf.py            # generación del PDF
│   ├── wsgi.py                  # entrada WSGI; acepta CLI o variables opcionales
│   ├── README.md
│   ├── templates/index.html
│   └── static/
│       ├── app.js
│       ├── style.css
│       └── plotly.min.js
├── docs/
│   ├── experimentos_cinc2020.md
│   └── informe_cinc2020_12.md
├── requirements.txt
└── README.md
```

---

## Esquema de 12 clases

La fuente de verdad está en:

```text
ecg/load.py
```

| Índice | Clase | Códigos SNOMED-CT incluidos |
|---:|---|---|
| 0 | `NSR` | `426783006` |
| 1 | `LAD` | `39732003` |
| 2 | `MI` | `164865005` |
| 3 | `TAb` | `164934002` |
| 4 | `AF` | `164889003`, `164890007`, `195080001`, `282825002`, `426749004`, `314208002` |
| 5 | `LVH` | `164873001` |
| 6 | `VEctopy` | `427172004`, `17338001`, `164884008`, `11157007`, `251180001`, `251182009`, `75532003`, `81898007` |
| 7 | `AVBlock` | `270492004`, `195042002`, `233917008`, `27885002`, `54016002`, `204384007` |
| 8 | `STach` | `427084000` |
| 9 | `RBBB` | `59118001`, `713427006` |
| 10 | `SB` | `426177001` |
| 11 | `AEctopy_Junctional` | `284470004`, `63593006`, `713422000`, `426664006`, `29320008`, `426995002`, `251164006`, `426648003`, `195101003`, `251268003`, `251170000`, `251168009`, `251173003` |

Notas:

- `RBBB` y `CRBBB` se fusionan en una única salida `RBBB`.
- Las salidas son independientes: se usa sigmoid por clase, no softmax.
- `TSV`, `TV`, `WPW` y ruido quedan excluidos. TSV/TV/WPW no alcanzan el mínimo de 1000 ECGs en CINC2020 ni agrupando códigos clínicamente relacionados; ruido no tiene SNOMED diagnóstico equivalente en el Challenge.
- Los registros sin ninguna de las 12 etiquetas se conservan por defecto como vectores all-zero. Pueden excluirse con `--drop_no_selected_labels`.

---

## Instalación

```bash
pip install -r requirements.txt
```

Para CUDA, instale PyTorch según su entorno. Este repositorio incluye una referencia para CUDA 12.8:

```bash
pip install -r requirements-cuda.txt
```

---

## Descargar CINC2020

```bash
bash descargar_datos_2020.sh
```

Subconjuntos descargados desde `training/`:

- `cpsc_2018`;
- `cpsc_2018_extra`;
- `georgia`;
- `ptb`;
- `ptb-xl`;
- `st_petersburg_incart`.

---

## Construir datasets HDF5

```bash
python examples/cinc2020/build_datasets.py \
  --data_dir dataset2020 \
  --output_dir data/cinc2020_12 \
  --workers 6 \
  --chunksize 8 \
  --write_batch_size 32
```

Salida esperada:

```text
data/cinc2020_12/
├── train.h5
├── val.h5
├── test.h5
├── class_mapping_12.csv
├── label_distribution.csv
├── source_distribution.csv
├── split_summary.csv
├── split_assignments.csv
├── preprocessing_summary.json
└── signal_processing_summary.json
```

Cada HDF5 contiene:

```text
signals -> (N, 5000, 12)
labels  -> (N, 12)
```

Las señales se remuestrean a 500 Hz, se normalizan por derivación con z-score y se recortan/rellenan hasta 5000 muestras.

---

## Entrenar ResNet-34 tipo Hannun

```bash
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
```

Los artefactos quedan en:

```text
saved/cinc2020/cinc2020_resnet/<run>/
├── best.pt
├── latest.pt
├── config_used.json
├── history.csv
└── training_summary.json
```

---

## Entrenar CNN convencional equivalente

```bash
python -m ecg.train examples/cinc2020/config_regular_cnn.json -e cinc2020_cnn
```

Esta CNN mantiene el calendario de convoluciones, filtros y downsampling, pero elimina las conexiones residuales. Sirve para una comparación justa frente a ResNet.

---

## Evaluar

```bash
python examples/cinc2020/evaluate.py \
  examples/cinc2020/config.json \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --output-dir results/cinc2020_12_resnet
```

Métricas guardadas:

```text
results/cinc2020_12_resnet/
├── thresholds_validation.csv
├── metrics_per_class.csv
├── metrics_global.csv
├── predictions_validation.csv
├── predictions_test.csv
├── confusion_matrices_validation.csv
├── confusion_matrices_test.csv
├── confusion_matrices/
│   ├── validation/
│   └── test/
└── evaluation_summary.json
```

Métricas incluidas:

- AUROC por clase;
- AUPRC por clase;
- sensibilidad/recall;
- especificidad;
- precisión;
- F1 por clase;
- F1 macro/micro;
- matrices de confusión 2x2 por clase.

Los thresholds se optimizan solo en validation y luego se aplican al test.

Para cuantificar el postprocesamiento normal opcional en todo el conjunto, ejecútelo explícitamente en evaluación. Esta regla añade `NSR` solo cuando ninguna clase supera su umbral y `P(NSR)` alcanza el mínimo elegido; no modifica probabilidades ni pesos del modelo:

```bash
python examples/cinc2020/evaluate.py \
  examples/cinc2020/config.json \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --output-dir results/cinc2020_12_resnet_fallback \
  --normal-fallback-min-prob 0.40
```

---

## Depurar un registro individual

Para revisar casos como `E00001`, compare el preprocesamiento de la webapp contra el HDF5 usado en evaluación:

```bash
python examples/cinc2020/debug_record_prediction.py \
  --config examples/cinc2020/config.json \
  --checkpoint saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --record E00001 \
  --hea training/georgia/g1/E00001.hea \
  --mat training/georgia/g1/E00001.mat \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv
```

Si la señal webapp y la señal HDF5 coinciden, la mala predicción es del checkpoint/umbral/postprocesamiento; si difieren, hay que corregir lectura, normalización, remuestreo, padding/recorte u orden de derivaciones y regenerar HDF5.

---

## Comparar ResNet vs CNN convencional

```bash
python examples/cinc2020/compare_models.py \
  --resnet results/cinc2020_12_resnet \
  --cnn results/cinc2020_12_cnn \
  --output results/cinc2020_12/model_comparison.csv
```

---

## Robustez controlada

```bash
python examples/cinc2020/robustness.py \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  data/cinc2020_12/test.h5 \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv \
  --output results/cinc2020_12_resnet/robustness.csv
```

Perturbaciones:

- ruido gaussiano;
- baseline wander;
- escalado de amplitud;
- dropout de derivaciones.

---

## Webapp

La app puede ejecutarse sin configurar variables de entorno. Por defecto selecciona automáticamente el mejor checkpoint `best.pt` disponible dentro de `saved/` o de la carpeta indicada con `--saved`:

```bash
python webapp/app.py --saved saved
```

Si necesita fijar un checkpoint exacto para una demostración reproducible, puede hacerlo por CLI, pero la interfaz web no requiere ni muestra selector de modelo:

```bash
python webapp/app.py --model saved/cinc2020/cinc2020_resnet/<run>/best.pt
```

Si los umbrales quedaron en una ruta no estándar, puede pasarlos explícitamente:

```bash
python webapp/app.py --saved saved --thresholds results/cinc2020_12_resnet/thresholds_validation.csv
```

La app incluye un fallback normal opcional para demos: si ninguna clase supera su umbral y `P(NSR) >= 0.40`, añade `NSR` como decisión final postprocesada. Para desactivarlo:

```bash
python webapp/app.py --saved saved --normal-fallback-min-prob 0
```

Opcionalmente también puede usar la entrada WSGI:

```bash
python webapp/wsgi.py --saved saved
```

La interfaz acepta:

- CSV de 12 derivaciones;
- par WFDB `.hea + .mat` de un mismo registro.

Cuando se sube `.hea + .mat`, la app lee el header, preprocesa la señal a 500 Hz / 5000 muestras / 12 derivaciones, compara automáticamente contra las etiquetas reales del campo `Dx` y genera un CSV convertido descargable. Para CSV puede escribir clases o códigos SNOMED reales manualmente para activar la comparación. La pantalla usa un dashboard con secciones de carga, resultado, detalle técnico y experimentos; muestra resultados multilabel, probabilidades por clase, comparación real vs predicho, trazado ECG con papel milimetrado, PDF, métricas exportadas, cambio ES/EN y resultados de comparación/robustez cuando existen en `results/`. Si existe `thresholds_validation.csv` generado por evaluación, la app usa automáticamente umbrales por clase en lugar del respaldo global 0.5.

---

## Dataset sintético para smoke tests

Si desea verificar que la estructura ejecuta sin descargar CINC2020:

```bash
python examples/cinc2020/make_syntethic.py
python -m ecg.train examples/cinc2020/config_syntethic.json -e smoke_test --epochs 1 --device cpu --no-amp
```

No use el dataset sintético para reportar métricas científicas.

---

## Diferencias frente a Hannun et al.

1. Se reemplaza el dataset privado Zio Patch/iRhythm por CINC2020.
2. Se usan ECG de 12 derivaciones.
3. Las etiquetas se definen con SNOMED-CT agrupado en 12 clases.
4. El problema se conserva multilabel; no se fuerza una clase única.
5. La evaluación usa métricas por clase y matrices 2x2 independientes.

---

## Estado del código

La rama original `SNOMED_CINC` estaba orientada a 27 clases puntuadas del Challenge. Esta versión reorganiza el código en la estructura canónica `ecg/`, `examples/`, `webapp/` y adapta el pipeline a 12 clases agrupadas, manteniendo sigmoid + `BCEWithLogitsLoss`.
