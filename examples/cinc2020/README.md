<a id="readme-top"></a>
<!--
*** Plantilla base: https://github.com/othneildrew/Best-README-Template
*** Submódulo: pipeline de datos, entrenamiento y experimentos CINC2020-12.
-->

[![Python 3.10+][python-shield]][python-url]
[![PyTorch 2.8][pytorch-shield]][pytorch-url]

<br />
<div align="center">
  <a href="https://github.com/PatVela/Proyectos-Universidad">
    <img src="../../images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Pipeline CINC2020-12</h3>

  <p align="center">
    Scripts de construcción de datos, entrenamiento, evaluación, comparación y robustez
    <br />
    para PhysioNet/CinC Challenge 2020 con 12 clases SNOMED multilabel.
    <br />
    <a href="../../README.md"><strong>Volver al README principal »</strong></a>
    <br />
    <br />
    <a href="../../webapp/README.md">Ver demo web</a>
    &middot;
    <a href="https://github.com/PatVela/Proyectos-Universidad/issues/new?labels=bug">Reportar Bug</a>
    &middot;
    <a href="https://github.com/PatVela/Proyectos-Universidad/issues/new?labels=enhancement">Pedir Funcionalidad</a>
  </p>
</div>

<details>
  <summary>Tabla de contenidos</summary>
  <ol>
    <li><a href="#sobre-este-módulo">Sobre este módulo</a></li>
    <li>
      <a href="#primeros-pasos">Primeros pasos</a>
      <ul>
        <li><a href="#prerrequisitos">Prerrequisitos</a></li>
      </ul>
    </li>
    <li><a href="#uso">Uso</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#licencia">Licencia</a></li>
  </ol>
</details>

## Sobre este módulo

Esta carpeta contiene el pipeline experimental completo del proyecto: desde la descarga de CINC2020 hasta la métrica oficial del Challenge. Los comandos se ejecutan desde la **raíz del repositorio**.

| Script | Propósito |
|---|---|
| `build_datasets.py` | Construye `train/val/test.h5` + CSV de distribución y resúmenes |
| `evaluate.py` | Métricas multilabel + umbrales calibrados en validación |
| `compare_models.py` | Compara dos evaluaciones (p. ej. ResNet v1 vs v2) |
| `robustness.py` | Robustez ante perturbaciones controladas de la señal |
| `challenge_score.py` | Métrica oficial estilo Challenge 2020 (27 códigos) |
| `diagnose_prediction.py` | Diagnóstico detallado de un registro |
| `debug_record_prediction.py` | Compara señal webapp vs señal HDF5 por registro |
| `make_syntethic.py` | Dataset sintético para smoke tests |
| `config*.json` | Configs de ResNet, ResNet v2 y sintético |
| `official/` | Scripts y tablas oficiales del Challenge 2020 |

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Primeros pasos

### Prerrequisitos

* Instalación base del proyecto (ver [README principal](../../README.md#instalación)).
* Datos CINC2020 descargados:
  ```sh
  bash descargar_datos_2020.sh
  ```

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Uso

Comandos PowerShell desde la **raíz del repositorio**, con variables para no escribir rutas de checkpoints a mano. Defínalas una vez por terminal (nombres según el [README principal](../../README.md#uso)):

```powershell
$resnet = Get-ChildItem saved/cinc2020/cinc2020_resnet/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$resnet2 = Get-ChildItem saved/cinc2020/cinc2020_resnet_v2*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$mejor = $resnet2
$evalMejor = "eval-resnet-v2"
```

### 1. Construir HDF5

```powershell
python examples/cinc2020/build_datasets.py --data_dir dataset2020 --output_dir data/cinc2020_12 --workers 8 --norm_mode physical --train_windows_max 4
```

### 2. Entrenar ResNet-34 tipo Hannun

```powershell
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
```

### 3. Entrenar ResNet-34 v2 (receta anti-sobreajuste)

```powershell
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2
```

Receta con aumentación de señal, label smoothing, más épocas y selección por F1-macro (ver §12).

### 4. Evaluar

```powershell
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir eval-resnet --temperatures eval-resnet/temperatures_validation.csv --threshold-beta 0.5
```

Genera umbrales por clase, métricas por clase y globales, predicciones, matrices de confusión 2×2 y curvas ROC/PR por clase. Para v2 use su config, checkpoint y carpeta (`eval-resnet-v2`). Para cuantificar el fallback normal en todo el conjunto:

```powershell
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir eval-resnet-fallback --normal-fallback-min-prob 0.40
```

### 5. Comparar modelos

```powershell
python examples/cinc2020/compare_models.py --eval-a eval-resnet --eval-b eval-resnet-v2 --label-a "ResNet v1" --label-b "ResNet v2" --output exp-files/model_comparison.csv
```

Las etiquetas son opcionales (por defecto usa el nombre de cada directorio).

### 6. Robustez controlada

```powershell
python examples/cinc2020/robustness.py $mejor data/cinc2020_12/test.h5 --thresholds "$evalMejor/thresholds_validation.csv" --output exp-files/robustness.csv
```

Perturbaciones: ruido gaussiano, baseline wander, escalado de amplitud y apagado de derivaciones (incluye fila base `clean`).

### 7. Diagnóstico de predicción

```powershell
$hea1 = Get-ChildItem dataset2020 -Recurse -Filter E00001.hea | Select-Object -First 1 -ExpandProperty FullName
$mat1 = [System.IO.Path]::ChangeExtension($hea1, '.mat')
python examples/cinc2020/diagnose_prediction.py --checkpoint $mejor --config examples/cinc2020/config.json --thresholds "$evalMejor/thresholds_validation.csv" --record E00001 --hea $hea1 --mat $mat1
```

### 8. Métrica oficial estilo Challenge 2020 (27 códigos)

```powershell
python examples/cinc2020/challenge_score.py --checkpoint $mejor --test-h5 data/cinc2020_12/test.h5 --thresholds "$evalMejor/thresholds_validation.csv" --output-dir metrica-challenge
```

### 9. Calibrar probabilidades (temperature scaling)

```powershell
python examples/cinc2020/calibrate.py --checkpoint $resnet --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet
```

Ajusta una temperatura por clase sobre validación y guarda el CSV de temperaturas + reporte ECE/Brier. Aplíquelo en evaluación con `--temperatures eval-resnet/temperatures_validation.csv`; la webapp lo detecta automáticamente.

### 10. Exportar a ONNX

```powershell
python examples/cinc2020/export_onnx.py --checkpoint $mejor --output resnet12.onnx
```

Usa `onnx`/`onnxruntime` de `requirements.txt`. Verifica el grafo y compara numéricamente contra PyTorch.

### 11. Análisis de errores por origen

```powershell
python examples/cinc2020/error_analysis.py --predictions "$evalMejor/predictions_test.csv" --test-h5 data/cinc2020_12/test.h5 --output errores_origen.csv
```

Reporta exact-match y F1 macro/micro por cada subconjunto de origen (hospital/fuente).

El diseño experimental completo está en [docs/experimentos_cinc2020.md](../../docs/experimentos_cinc2020.md).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

### 12. Máxima accuracy: F-beta y ensemble

Umbrales orientados a precisión (menos falsos positivos) con F0.5:

```powershell
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir eval-resnet --threshold-beta 0.5
```

`--threshold-beta 1.0` = F1 (equilibrio, valor por defecto); `0.5` = menos FPs; `2.0` = menos FNs. La opción `--nsr-exclusive-min-prob` (predecir solo NSR si P(NSR) es muy alta) resultó perjudicial en ablación (−11 pts de F1-macro sin ganancia de precisión, porque NSR coexiste legítimamente con clases morfológicas); no se recomienda. La columna `validation_objective` de `thresholds_validation.csv` registra el objetivo optimizado.

Ensemble promedio de dos checkpoints (mismo layout de salida que `evaluate.py`):

```powershell
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json $resnet $resnet2 --output-dir ensemble --alpha 0.5 --threshold-beta 0.5 --temperatures-a eval-resnet/temperatures_validation.csv --temperatures-b eval-resnet-v2/temperatures_validation.csv
```

Para re-entrenar la ResNet con receta anti-sobreajuste (más épocas, más regularización), usar `examples/cinc2020/config_resnet_v2.json` con `ecg.train`. Ese config selecciona el mejor checkpoint por F1-macro en validación (`early_stopping_metric: val_f1_macro`) en vez de por `val_loss`, porque ambas métricas pueden discrepar; `history.csv` registra ambas curvas en todos los runs. Además activa aumentación de señal (`augment: true`: ruido, deriva basal, escala, desplazamiento temporal, dropout de derivación) y label smoothing (`label_smoothing: 0.05`) para frenar el sobreajuste.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Roadmap

- [x] Evaluación multilabel con umbrales calibrados
- [x] Comparación de evaluaciones + robustez controlada
- [x] Métrica oficial del Challenge sobre 27 códigos
- [x] Curvas ROC/PR por clase exportadas a CSV
- [x] Análisis de errores por subconjunto de origen
- [x] Calibración temperature scaling + reporte ECE/Brier
- [x] Exportación a ONNX verificada numéricamente
- [ ] Validación cruzada leave-one-source-out (generalización por hospital)
- [ ] Curvas de calibración (reliability diagrams) exportadas a CSV
- [x] Aumentación de señal durante el entrenamiento
- [x] Umbrales F-beta + selección de checkpoint por F1-macro
- [x] Ensemble promedio de checkpoints (evaluación e inferencia webapp)

Ver los [issues abiertos](https://github.com/PatVela/Proyectos-Universidad/issues) para más propuestas.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Licencia

Distribuido bajo licencia GPL-3.0. Ver `LICENSE` en la raíz para más información.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

[python-shield]: https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/pytorch-2.8-ee4c2c?style=for-the-badge&logo=pytorch
[pytorch-url]: https://pytorch.org/
