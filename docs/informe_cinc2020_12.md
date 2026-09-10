# Informe técnico — CINC2020-12 tipo Hannun

## Resumen

Se adaptó el enfoque de CNN residual profunda de Hannun et al. a PhysioNet/CinC Challenge 2020. El dataset original de Zio Patch/iRhythm no es público, por lo que se usa CINC2020 con ECG de 12 derivaciones y diagnósticos SNOMED-CT.

La implementación sigue la estructura canónica solicitada:

```text
ecg/        paquete principal
examples/   scripts de experimento
webapp/     Flask + inferencia + reportes
docs/       documentación experimental
```

## Etiquetas

El esquema final tiene 12 clases multilabel y está definido en `ecg/load.py`.

1. `NSR`: 426783006.
2. `LAD`: 39732003.
3. `MI`: 164865005.
4. `TAb`: 164934002.
5. `AF`: 164889003, 164890007, 195080001, 282825002, 426749004, 314208002.
6. `LVH`: 164873001.
7. `VEctopy`: 427172004, 17338001, 164884008, 11157007, 251180001, 251182009, 75532003, 81898007.
8. `AVBlock`: 270492004, 195042002, 233917008, 27885002, 54016002, 204384007.
9. `STach`: 427084000.
10. `RBBB`: 59118001, 713427006.
11. `SB`: 426177001.
12. `AEctopy_Junctional`: 284470004, 63593006, 713422000, 426664006, 29320008, 426995002, 251164006, 426648003, 195101003, 251268003, 251170000, 251168009, 251173003.

`RBBB` y `CRBBB` quedan fusionados.

## Exclusiones

Se excluyen TSV, TV, WPW y ruido. TSV/TV/WPW no alcanzan el mínimo de 1000 ECGs en CINC2020 incluso agrupando códigos relacionados. Ruido no tiene código SNOMED-CT diagnóstico equivalente en el Challenge.

## Preprocesamiento

Implementación: `ecg/load.py` y wrapper `examples/cinc2020/build_datasets.py`.

Pasos:

1. leer `.hea`;
2. extraer `# Dx`;
3. leer `.mat`;
4. exigir 12 derivaciones;
5. reordenar derivaciones estándar;
6. remuestrear a 500 Hz;
7. z-score por derivación antes del padding;
8. recorte centrado o padding hasta 5000 muestras;
9. guardar HDF5 train/val/test.

El split usa `MultilabelStratifiedShuffleSplit`; no hay fallback por fuente.

## Modelo

Implementación: `ecg/network.py`.

- `ECGResNet34`: ResNet 1D tipo Hannun, 16 bloques residuales, salida de 12 logits.
- `ECGRegularCNN`: CNN convencional equivalente, mismo calendario de convoluciones y downsampling, sin shortcuts.

## Entrenamiento

Implementación: `ecg/train.py`.

- `BCEWithLogitsLoss`.
- `pos_weight` calculado solo desde train.
- Adam.
- ReduceLROnPlateau.
- Early stopping.
- AMP opcional.
- Logs reproducibles en `history.csv` y `training_summary.json`.

## Evaluación

Implementación: `examples/cinc2020/evaluate.py`.

- thresholds por clase optimizados en validation;
- AUROC por clase;
- AUPRC por clase;
- sensibilidad;
- especificidad;
- F1;
- matrices de confusión 2x2.

## Diferencias frente al paper original

1. Dataset público CINC2020 en lugar de dataset privado iRhythm.
2. ECG de 12 derivaciones.
3. Etiquetas SNOMED agrupadas en 12 clases.
4. Formulación multilabel real.
5. Métricas por clase en lugar de matriz multiclase única.
