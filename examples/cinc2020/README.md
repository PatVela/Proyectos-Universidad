# Ejemplo CINC2020-12

Esta carpeta contiene los scripts de construcción de datos, entrenamiento, evaluación, comparación y robustez para PhysioNet/CinC Challenge 2020 con 12 clases SNOMED multilabel.

## 1. Construir HDF5

```bash
python examples/cinc2020/build_datasets.py \
  --data_dir dataset2020 \
  --output_dir data/cinc2020_12 \
  --workers 6
```

## 2. Entrenar ResNet-34 tipo Hannun

```bash
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
```

## 3. Entrenar CNN convencional equivalente

```bash
python -m ecg.train examples/cinc2020/config_regular_cnn.json -e cinc2020_cnn
```

## 4. Evaluar

```bash
python examples/cinc2020/evaluate.py \
  examples/cinc2020/config.json \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --output-dir results/cinc2020_12_resnet
```

## 5. Comparar modelos

```bash
python examples/cinc2020/compare_models.py \
  --resnet results/cinc2020_12_resnet \
  --cnn results/cinc2020_12_cnn \
  --output results/cinc2020_12/model_comparison.csv
```

## 6. Robustez controlada

```bash
python examples/cinc2020/robustness.py \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  data/cinc2020_12/test.h5 \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv
```
