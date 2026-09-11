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
    <li><a href="#contacto">Contacto</a></li>
  </ol>
</details>

## Sobre este módulo

Esta carpeta contiene el pipeline experimental completo del proyecto: desde la descarga de CINC2020 hasta la métrica oficial del Challenge. Los comandos se ejecutan desde la **raíz del repositorio**.

| Script | Propósito |
|---|---|
| `build_datasets.py` | Construye `train/val/test.h5` + CSV de distribución y resúmenes |
| `evaluate.py` | Métricas multilabel + umbrales calibrados en validación |
| `compare_models.py` | Compara ResNet-34 vs CNN convencional |
| `robustness.py` | Robustez ante perturbaciones controladas de la señal |
| `challenge_score.py` | Métrica oficial estilo Challenge 2020 (27 códigos) |
| `diagnose_prediction.py` | Diagnóstico detallado de un registro |
| `debug_record_prediction.py` | Compara señal webapp vs señal HDF5 por registro |
| `make_syntethic.py` | Dataset sintético para smoke tests |
| `config*.json` | Configs de ResNet, CNN convencional y sintético |
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

> Los placeholders `<...>` representan rutas de su máquina. Los archivos generados por evaluación se detectan automáticamente en la webapp.

### 1. Construir HDF5

```sh
python examples/cinc2020/build_datasets.py \
  --data_dir dataset2020 \
  --output_dir data/cinc2020_12 \
  --workers 6
```

### 2. Entrenar ResNet-34 tipo Hannun

```sh
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
```

### 3. Entrenar CNN convencional equivalente

```sh
python -m ecg.train examples/cinc2020/config_regular_cnn.json -e cinc2020_cnn
```

### 4. Evaluar

```sh
python examples/cinc2020/evaluate.py \
  examples/cinc2020/config.json \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --output-dir <dir-evaluacion>
```

Genera umbrales por clase, métricas por clase y globales, predicciones y matrices de confusión 2×2. Para cuantificar la regla post hoc NSR en todo el conjunto:

```sh
python examples/cinc2020/evaluate.py \
  examples/cinc2020/config.json \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --output-dir <dir-evaluacion-fallback> \
  --normal-fallback-min-prob 0.40
```

### 5. Comparar modelos

```sh
python examples/cinc2020/compare_models.py \
  --resnet <dir-eval-resnet> \
  --cnn <dir-eval-cnn> \
  --output <comparacion.csv>
```

### 6. Robustez controlada

```sh
python examples/cinc2020/robustness.py \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  data/cinc2020_12/test.h5 \
  --thresholds <umbrales-por-clase.csv> \
  --output <robustez.csv>
```

Perturbaciones: ruido gaussiano, baseline wander, escalado de amplitud y apagado de derivaciones (incluye fila base `clean`).

### 7. Diagnóstico de predicción

```sh
python examples/cinc2020/diagnose_prediction.py \
  --checkpoint saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --config examples/cinc2020/config.json \
  --thresholds <umbrales-por-clase.csv> \
  --record E00001 \
  --hea dataset2020/training/georgia/g1/E00001.hea \
  --mat dataset2020/training/georgia/g1/E00001.mat
```

### 8. Métrica oficial estilo Challenge 2020 (27 códigos)

```sh
python examples/cinc2020/challenge_score.py \
  --checkpoint saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --test-h5 data/cinc2020_12/test.h5 \
  --thresholds <umbrales-por-clase.csv> \
  --output-dir <dir-metrica-challenge>
```

El diseño experimental completo está en [docs/experimentos_cinc2020.md](../../docs/experimentos_cinc2020.md).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Roadmap

- [x] Evaluación multilabel con umbrales calibrados
- [x] Comparación ResNet vs CNN + robustez controlada
- [x] Métrica oficial del Challenge sobre 27 códigos
- [ ] Curvas ROC/PR por clase exportadas a CSV
- [ ] Análisis de errores por subconjunto de origen

Ver los [issues abiertos](https://github.com/PatVela/Proyectos-Universidad/issues) para más propuestas.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Licencia

Distribuido bajo licencia GPL-3.0. Ver `LICENSE` en la raíz para más información.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

## Contacto

PatVela — Universidad Nacional de San Agustín de Arequipa.

Link del proyecto: [https://github.com/PatVela/Proyectos-Universidad](https://github.com/PatVela/Proyectos-Universidad)

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

[python-shield]: https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/pytorch-2.8-ee4c2c?style=for-the-badge&logo=pytorch
[pytorch-url]: https://pytorch.org/
