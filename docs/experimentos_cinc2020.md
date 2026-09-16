# Experimentos CINC2020-12

## Propósito

Adaptar el enfoque Hannun et al. a un dataset público de ECG de 12 derivaciones: PhysioNet/CinC Challenge 2020. El problema se mantiene multilabel con 12 clases agrupadas por SNOMED-CT.

## Preparación

```bash
bash descargar_datos_2020.sh
python examples/cinc2020/build_datasets.py --data_dir dataset2020 --output_dir data/cinc2020_12 --workers 6
```

## Experimento A — ResNet-34 tipo Hannun

```bash
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
python examples/cinc2020/evaluate.py examples/cinc2020/config.json saved/cinc2020/cinc2020_resnet/<run>/best.pt --output-dir results/cinc2020_12_resnet
```

## Experimento B — ResNet-34 v2 (receta anti-sobreajuste)

```bash
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2
python examples/cinc2020/evaluate.py examples/cinc2020/config_resnet_v2.json saved/cinc2020/cinc2020_resnet_v2/<run>/best.pt --output-dir <dir-evaluacion-v2> --threshold-beta 0.5
```

## Comparación justa

Ambos modelos deben usar:

- mismo dataset HDF5;
- mismo split train/validation/test;
- mismo preprocesamiento;
- mismo número de clases;
- mismo batch size, optimizador y criterio de pérdida;
- mismas métricas de evaluación.

```bash
python examples/cinc2020/compare_models.py \
  --eval-a results/cinc2020_12_resnet \
  --eval-b <dir-evaluacion-v2> \
  --label-a "ResNet v1" --label-b "ResNet v2" \
  --output results/cinc2020_12/model_comparison.csv
```

## Experimento C — Ensemble v1+v2 (modelo final)

```bash
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json <mejor-v1.pt> <mejor-v2.pt> --output-dir <dir-ensemble> --threshold-beta 0.5 --temperatures-a <temperaturas-a>.csv --temperatures-b <temperaturas-b>.csv
python examples/cinc2020/compare_models.py --eval-a <dir-eval-v2> --eval-b <dir-ensemble> --label-a "ResNet v2" --label-b "Ensemble" --output <comparacion-ensemble.csv>
```

Promedio simple con calibración por modelo y umbrales F0.5 tuneados sobre el promedio.
Ver resultados en [resultados_finales.md](resultados_finales.md).

## Métricas a reportar

- AUROC por clase;
- AUPRC por clase;
- sensibilidad;
- especificidad;
- precisión;
- F1 por clase;
- F1 macro/micro;
- matrices de confusión 2x2 por clase;
- número de parámetros;
- tiempo total de entrenamiento.

## Robustez controlada

```bash
python examples/cinc2020/robustness.py \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  data/cinc2020_12/test.h5 \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv \
  --output results/cinc2020_12_resnet/robustness.csv
```

Perturbaciones incluidas: ruido gaussiano, baseline wander, escalado de amplitud y dropout de derivaciones.
