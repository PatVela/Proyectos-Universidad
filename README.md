<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Plantilla base: https://github.com/othneildrew/Best-README-Template
*** Proyecto: réplica académica de Hannun et al. (Nat Med 2019) sobre CINC2020.
-->

<!-- PROJECT SHIELDS -->
<!--
*** Se usan "reference style" links para legibilidad.
*** Ver la declaración de variables al final del documento.
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL-3.0 License][license-shield]][license-url]
[![Python 3.10+][python-shield]][python-url]
[![PyTorch 2.8][pytorch-shield]][pytorch-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/PatVela/Proyectos-Universidad">
    <img src="images/logo.png" alt="Logo" width="100" height="100">
  </a>

<h3 align="center">ECG CINC2020-12 · ResNet tipo Hannun</h3>

  <p align="center">
    Clasificador multilabel de ECG de 12 derivaciones con redes convolucionales profundas.
    <br />
    Réplica académica de Hannun et al. (<i>Nature Medicine</i>, 2019) sobre el dataset público PhysioNet/CinC Challenge 2020.
    <br />
    <a href="docs/informe_cinc2020_12.md"><strong>Explorar la documentación »</strong></a>
    <br />
    <br />
    <a href="webapp/README.md">Ver demo web</a>
    &middot;
    <a href="https://github.com/PatVela/Proyectos-Universidad/issues/new?labels=bug">Reportar Bug</a>
    &middot;
    <a href="https://github.com/PatVela/Proyectos-Universidad/issues/new?labels=enhancement">Pedir Funcionalidad</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Tabla de contenidos</summary>
  <ol>
    <li>
      <a href="#sobre-el-proyecto">Sobre el proyecto</a>
      <ul>
        <li><a href="#construido-con">Construido con</a></li>
      </ul>
    </li>
    <li>
      <a href="#primeros-pasos">Primeros pasos</a>
      <ul>
        <li><a href="#prerrequisitos">Prerrequisitos</a></li>
        <li><a href="#instalación">Instalación</a></li>
      </ul>
    </li>
    <li><a href="#uso">Uso</a></li>
    <li><a href="#esquema-de-12-clases">Esquema de 12 clases</a></li>
    <li><a href="#estructura-del-proyecto">Estructura del proyecto</a></li>
    <li><a href="#aplicación-web">Aplicación web</a></li>
    <li><a href="#experimentos">Experimentos</a></li>
    <li><a href="#diferencias-frente-a-hannun-et-al">Diferencias frente a Hannun et al.</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contribuir">Contribuir</a></li>
    <li><a href="#licencia">Licencia</a></li>
    <li><a href="#contacto">Contacto</a></li>
    <li><a href="#agradecimientos">Agradecimientos</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## Sobre el proyecto

[![Informe de ejemplo][product-screenshot]](docs/informe_cinc2020_12.md)

El dataset original de Hannun et al. (Zio Patch/iRhythm) no es público, así que esta implementación adapta su enfoque ResNet-1D al dataset público **PhysioNet/Computing in Cardiology Challenge 2020**: ECG de 12 derivaciones con etiquetas diagnósticas SNOMED-CT.

El problema se formula como **clasificación multilabel real** —un ECG puede tener varios diagnósticos a la vez— con 12 salidas sigmoid independientes, umbrales calibrados por clase y preprocesamiento en unidades físicas (milivoltios + filtro pasa-banda 0.5–50 Hz). El repositorio incluye el pipeline completo: construcción de datasets HDF5, entrenamiento, evaluación, experimentos de comparación/robustez y una aplicación web con informe PDF.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

### Construido con

* [![Python][Python-badge]][Python-url]
* [![PyTorch][PyTorch-badge]][PyTorch-url]
* [![Flask][Flask-badge]][Flask-url]
* [![NumPy][NumPy-badge]][NumPy-url]
* [![pandas][pandas-badge]][pandas-url]
* [![scikit-learn][sklearn-badge]][sklearn-url]
* [![Matplotlib][Matplotlib-badge]][Matplotlib-url]
* [![WFDB][WFDB-badge]][WFDB-url]
* [![h5py][h5py-badge]][h5py-url]
* [![Plotly][Plotly-badge]][Plotly-url]

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- GETTING STARTED -->
## Primeros pasos

Para tener una copia local funcionando, siga estos pasos.

### Prerrequisitos

* Python 3.10 o superior.
* ~15 GB de disco libre si va a descargar CINC2020 completo.
* 16 GB de RAM recomendados (el pipeline precarga los HDF5 a memoria).
* GPU NVIDIA opcional (el código corre en CPU; con CUDA instale PyTorch según `requirements-cuda.txt`).

### Instalación

1. Clone el repositorio (rama del proyecto):
   ```sh
   git clone -b Paper_Replica_V3 https://github.com/PatVela/Proyectos-Universidad.git
   cd Proyectos-Universidad
   ```
2. Instale dependencias:
   ```sh
   pip install -r requirements.txt
   ```
3. Para CUDA, instale PyTorch según su entorno (referencia CUDA 12.8 incluida):
   ```sh
   pip install -r requirements-cuda.txt
   ```
4. Solo para verificar la estructura sin descargar datos, use el dataset sintético:
   ```sh
   python examples/cinc2020/make_syntethic.py
   python -m ecg.train examples/cinc2020/config_syntethic.json -e smoke_test --epochs 1 --device cpu --no-amp
   ```
   No use el dataset sintético para reportar métricas científicas.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- USAGE EXAMPLES -->
## Uso

Pipeline completo, en orden:

1. **Descargar CINC2020** (subconjuntos `cpsc_2018`, `cpsc_2018_extra`, `georgia`, `ptb`, `ptb-xl`, `st_petersburg_incart`):
   ```sh
   bash descargar_datos_2020.sh
   ```
2. **Construir datasets HDF5** (12 derivaciones → 500 Hz → mV + pasa-banda 0.5–50 Hz → recorte ±5 mV → 5000 muestras):
   ```sh
   python examples/cinc2020/build_datasets.py \
     --data_dir dataset2020 \
     --output_dir data/cinc2020_12 \
     --workers 6 --chunksize 8 --write_batch_size 32
   ```
   Cada HDF5 contiene `signals → (N, 5000, 12)` y `labels → (N, 12)`. En `train`, los registros largos aportan hasta 4 ventanas de 10 s; validación y test usan una ventana centrada.
3. **Entrenar ResNet-34 tipo Hannun**:
   ```sh
   python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
   ```
4. **Entrenar CNN convencional equivalente** (mismo calendario de convoluciones, sin residuales):
   ```sh
   python -m ecg.train examples/cinc2020/config_regular_cnn.json -e cinc2020_cnn
   ```
5. **Evaluar** (umbrales optimizados en validación, aplicados a test):
   ```sh
   python examples/cinc2020/evaluate.py \
     examples/cinc2020/config.json \
     saved/cinc2020/cinc2020_resnet/<run>/best.pt \
     --output-dir <dir-evaluacion>
   ```
   El directorio de salida contendrá umbrales por clase, métricas por clase y globales, predicciones de validación/test, matrices de confusión 2×2 y un resumen JSON.
6. **Abrir la aplicación web**:
   ```sh
   python webapp/app.py --saved saved
   ```

_Más ejemplos y comandos avanzados en [examples/cinc2020/README.md](examples/cinc2020/README.md) y [webapp/README.md](webapp/README.md)._

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- CLASS SCHEMA -->
## Esquema de 12 clases

Fuente de verdad: `ecg/load.py` (esquema `cinc2020_12_grouped_snomed_v2`). Cubre las 27 clases puntuadas oficiales del Challenge 2020 (ver `examples/cinc2020/official/dx_mapping_scored.csv`); el esquema v1 solo cubría ~14 y dejaba ~21% de registros como all-zero.

| # | Clase | Códigos SNOMED-CT incluidos |
|--:|---|---|
| 0 | `NSR` | `426783006`, `427393009` |
| 1 | `AxisDev` | `39732003`, `445118002`, `47665007`, `445211001`, `251200008` |
| 2 | `MI` | `164865005`, `57054005`, `164867002`, `54329005`, `164861001`, `413444003`, `413844008`, `426434006`, `425419005`, `425623009`, `164917005` |
| 3 | `TAb` | `164934002`, `59931005`, `428750005`, `429622005`, `164931005`, `164930006`, `55930002`, `704997005`, `111975006`, `77867006`, `164937009`, `251259000`, `428417006` |
| 4 | `AF` | `164889003`, `164890007`, `195080001`, `282825002`, `426749004`, `314208002` |
| 5 | `LVH` | `164873001`, `370365005`, `446813000`, `67741000119109`, `253352002`, `195126007`, `89792004`, `266249003`, `253339007`, `446358003` |
| 6 | `VEctopy` | `427172004`, `17338001`, `164884008`, `11157007`, `251180001`, `251182009`, `75532003`, `81898007`, `164895002`, `425856008`, `164896001`, `111288001`, `49260003`, `13640000` |
| 7 | `AVBlock` | `270492004`, `195042002`, `233917008`, `27885002`, `54016002`, `204384007`, `164947007`, `65778007` |
| 8 | `STach` | `427084000` |
| 9 | `BBB` | `59118001`, `713427006`, `713426002`, `164909002`, `251120003`, `6374002`, `698252002`, `82226007`, `251146004`, `164951009` |
| 10 | `SB` | `426177001`, `426627000`, `60423000`, `74615001` |
| 11 | `AEctopy_Junctional` | `284470004`, `63593006`, `713422000`, `426664006`, `29320008`, `426995002`, `251164006`, `426648003`, `195101003`, `251268003`, `251170000`, `251168009`, `251173003`, `426761007`, `67198005`, `10370003`, `251266004` |

Notas:

* Las salidas son independientes: sigmoid por clase, no softmax.
* Quedan fuera: WPW/preexcitación (muy infrecuente y morfológicamente singular), diagnósticos clínicos no-ECG (HF/HVD/CHD/TIA) y ruido (sin SNOMED equivalente).
* Los registros sin ninguna de las 12 etiquetas (<1% en v2) se conservan como vectores all-zero; pueden excluirse con `--drop_no_selected_labels`.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- PROJECT STRUCTURE -->
## Estructura del proyecto

```text
.
├── ecg/                    # paquete: carga, red, entrenamiento e inferencia
│   ├── load.py             # lectores ECG, normalización y mapeo SNOMED
│   ├── network.py          # ResNet-34 tipo Hannun y CNN convencional
│   ├── predict.py          # inferencia desde checkpoints (CSV/HDF5)
│   ├── train.py            # entrenamiento + logs reproducibles
│   └── util.py             # checkpoints, parámetros y seeds
├── examples/cinc2020/      # pipeline: datos, evaluación y experimentos
│   ├── build_datasets.py / evaluate.py / compare_models.py
│   ├── robustness.py / challenge_score.py
│   ├── diagnose_prediction.py / debug_record_prediction.py
│   ├── config*.json        # configs ResNet, CNN y sintético
│   └── official/           # scripts y tablas oficiales del Challenge 2020
├── webapp/                 # dashboard Flask + informe PDF
│   ├── app.py / prediction.py / report_pdf.py / wsgi.py
│   ├── project_info.py     # datos institucionales
│   └── templates/ + static/
├── docs/                   # informe técnico y guías de experimentos
├── images/                 # logo y capturas para documentación
├── requirements.txt / requirements-cuda.txt
├── descargar_datos_2020.sh
└── README.md
```

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- WEBAPP -->
## Aplicación web

Dashboard Flask en español/inglés: carga CSV o par WFDB `.hea + .mat`, predicción multilabel con probabilidades por clase, trazado en papel milimetrado, comparación automática contra el campo `Dx`, informe PDF descargable y secciones de detalle técnico, métricas y experimentos.

```sh
python webapp/app.py --saved saved
```

Detalles, formatos y endpoints en [webapp/README.md](webapp/README.md).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- EXPERIMENTS -->
## Experimentos

* **Evaluación multilabel**: AUROC/AUPRC por clase, sensibilidad, especificidad, F1 macro/micro y matrices 2×2 (`evaluate.py`).
* **Comparación arquitectónica**: ResNet-34 vs CNN convencional con idéntico dataset, split y métricas (`compare_models.py`).
* **Robustez controlada**: ruido gaussiano, baseline wander, escalado de amplitud y apagado de derivaciones (`robustness.py`).
* **Métrica oficial estilo Challenge 2020** sobre los 27 códigos puntuados (`challenge_score.py`).
* **Diagnóstico por registro**: inspección de casos individuales (`diagnose_prediction.py`, `debug_record_prediction.py`).

Los resultados se visualizan automáticamente en la webapp. Comandos en [examples/cinc2020/README.md](examples/cinc2020/README.md) y diseño experimental en [docs/experimentos_cinc2020.md](docs/experimentos_cinc2020.md).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- DIFFERENCES -->
## Diferencias frente a Hannun et al.

1. Se reemplaza el dataset privado Zio Patch/iRhythm por CINC2020.
2. Se usan ECG de 12 derivaciones.
3. Las etiquetas se definen con SNOMED-CT agrupado en 12 clases.
4. El problema se conserva multilabel; no se fuerza una clase única.
5. La evaluación usa métricas por clase y matrices 2×2 independientes.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Esquema v2 con cobertura total de las 27 clases puntuadas
- [x] Preprocesamiento en unidades físicas (mV + pasa-banda)
- [x] Precarga de HDF5 a RAM para entrenamiento y evaluación
- [x] Webapp con informe PDF profesional y secciones ES/EN
- [x] Experimentos de comparación arquitectónica y robustez
- [ ] Calibración de probabilidades (temperature scaling)
- [ ] Exportación a ONNX para inferencia ligera
- [ ] Curvas ROC/PR interactivas en la webapp

Ver los [issues abiertos](https://github.com/PatVela/Proyectos-Universidad/issues) para la lista completa de propuestas y problemas conocidos.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- CONTRIBUTING -->
## Contribuir

Las contribuciones hacen de la comunidad open source un lugar increíble para aprender e inspirarse. Cualquier aporte será **muy apreciado**.

1. Haga Fork del proyecto
2. Cree su rama (`git checkout -b feature/Funcionalidad`)
3. Haga Commit (`git commit -m 'Agrega Funcionalidad'`)
4. Haga Push (`git push origin feature/Funcionalidad`)
5. Abra un Pull Request

### Top contributors:

<a href="https://github.com/PatVela/Proyectos-Universidad/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PatVela/Proyectos-Universidad" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- LICENSE -->
## Licencia

Distribuido bajo licencia GPL-3.0. Ver `LICENSE` para más información.

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- CONTACT -->
## Contacto

PatVela — Universidad Nacional de San Agustín de Arequipa, Escuela Profesional de Ingeniería Electrónica.

Link del proyecto: [https://github.com/PatVela/Proyectos-Universidad](https://github.com/PatVela/Proyectos-Universidad)

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Agradecimientos

* Hannun AY et al. Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. *Nature Medicine*. 2019;25(1):65-69. [DOI 10.1038/s41591-018-0268-3](https://doi.org/10.1038/s41591-018-0268-3)
* PhysioNet/Computing in Cardiology Challenge 2020: Classification of 12-lead ECGs.
* Universidad Nacional de San Agustín de Arequipa — Facultad de Ingeniería de Producción y Servicios.
* Plantilla [Best-README-Template](https://github.com/othneildrew/Best-README-Template) e insignias de [shields.io](https://shields.io).

<p align="right">(<a href="#readme-top">volver arriba</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[contributors-url]: https://github.com/PatVela/Proyectos-Universidad/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[forks-url]: https://github.com/PatVela/Proyectos-Universidad/network/members
[stars-shield]: https://img.shields.io/github/stars/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[stars-url]: https://github.com/PatVela/Proyectos-Universidad/stargazers
[issues-shield]: https://img.shields.io/github/issues/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[issues-url]: https://github.com/PatVela/Proyectos-Universidad/issues
[license-shield]: https://img.shields.io/github/license/PatVela/Proyectos-Universidad.svg?style=for-the-badge
[license-url]: https://github.com/PatVela/Proyectos-Universidad/blob/Paper_Replica_V3/LICENSE
[python-shield]: https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/pytorch-2.8-ee4c2c?style=for-the-badge&logo=pytorch
[pytorch-url]: https://pytorch.org/
[product-screenshot]: images/informe-ejemplo.png
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch-badge]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Flask-badge]: https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/
[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[pandas-badge]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[sklearn-badge]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
[WFDB-badge]: https://img.shields.io/badge/WFDB-4B8BBE?style=for-the-badge
[WFDB-url]: https://archive.physionet.org/physiobank/database/wfdbcal/
[h5py-badge]: https://img.shields.io/badge/h5py-00A6ED?style=for-the-badge
[h5py-url]: https://www.h5py.org/
[Plotly-badge]: https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white
[Plotly-url]: https://plotly.com/
