# ECG Web App Flask

Aplicación web interactiva para clasificar señales ECG de **una sola derivación** usando checkpoints PyTorch entrenados con el paquete `ecg/` del repositorio.

La app está pensada para acompañar el trabajo experimental sobre **PhysioNet/CinC2017**:

- muestra la predicción por intervalos de 256 muestras;
- muestra la distribución de clases del registro;
- permite descargar un informe PDF;
- muestra métricas reales del modelo;
- muestra la comparación **ResNet-34 vs CNN convencional**;
- muestra el análisis de **robustez**;
- compara automáticamente contra las etiquetas reales de `REFERENCE-v3.csv` cuando el archivo subido conserva el ID oficial del registro.

---

## Requisitos

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
```

Necesitas al menos un checkpoint `.pt`, por ejemplo entrenado con:

```bash
python -m ecg.train examples/cinc17/config.json \
  -e cinc17_resnet \
  --seed 2018
```

Opcionalmente, para comparación automática con etiqueta real, necesitas el archivo oficial:

```text
REFERENCE-v3.csv
```

---

## Ejecutar en desarrollo

### Usar la ResNet-34 principal

```bash
python webapp/app.py \
  --saved saved/cinc17_resnet \
  --reference dataset2017/REFERENCE-v3.csv \
  --host 127.0.0.1 \
  --port 5000
```

Abre:

```text
http://127.0.0.1:5000/
```

### Usar un checkpoint exacto

```bash
python webapp/app.py \
  --model saved/cinc17_resnet/<timestamp>/<checkpoint>.pt \
  --reference dataset2017/REFERENCE-v3.csv
```

### Usar la CNN convencional

```bash
python webapp/app.py \
  --saved saved/cinc17_cnn \
  --reference dataset2017/REFERENCE-v3.csv
```

### Elegir automáticamente el mejor checkpoint global

```bash
python webapp/app.py --saved saved
```

> Si `saved/` contiene tanto ResNet como CNN, la app elegirá el checkpoint con menor `val_loss` dentro de toda la carpeta. Para demostraciones científicas, se recomienda usar `--model` exacto o `--saved saved/cinc17_resnet`.

---

## Modelo usado en la pestaña Resultado

La pestaña **Resultado** usa únicamente el modelo cargado en `PredictionService` al arrancar Flask.

| Comando | Modelo usado en Resultado |
|---|---|
| `--saved saved/cinc17_resnet` | ResNet-34. |
| `--saved saved/cinc17_cnn` | CNN convencional. |
| `--model ruta/checkpoint.pt` | Exactamente ese checkpoint. |
| `--saved saved` | Mejor checkpoint global por menor `val_loss`. |

La interfaz muestra el modelo usado para cada predicción, por ejemplo:

```text
Modelo usado para este resultado: ResNet-34
```

---

## Comparación automática con etiqueta real

Si la app se inicia con:

```bash
--reference dataset2017/REFERENCE-v3.csv
```

entonces carga el CSV oficial de etiquetas reales:

```text
A00001,N
A00002,N
A00003,N
A00004,A
...
```

Cuando subes un archivo cuyo nombre contiene el ID oficial, la app lo compara automáticamente:

```text
A00004.mat  -> busca A00004 en REFERENCE-v3.csv
A00004.dat  -> busca A00004 en REFERENCE-v3.csv
A00004.csv  -> busca A00004 en REFERENCE-v3.csv
A00004_filtrado.npy -> busca A00004 en REFERENCE-v3.csv
```

En **Resultado** se muestra:

```text
Etiqueta real: Fibrilación auricular (A)
Predicción: Fibrilación auricular (A)
✔ Predicción correcta
```

o, si no coincide:

```text
Etiqueta real: Normal (N)
Predicción: Otro ritmo (O)
✘ Predicción distinta a la etiqueta real
```

Si el archivo no conserva el ID oficial, puedes usar el campo manual:

```text
Diagnóstico conocido (opcional)
```

Acepta valores como:

```text
N, A, O, ~
Normal, AF, Otro, Ruido
```

### Rutas donde se busca automáticamente `REFERENCE-v3.csv`

Si no pasas `--reference`, la app intenta encontrarlo en:

```text
dataset2017/REFERENCE-v3.csv
training2017/REFERENCE-v3.csv
data/REFERENCE-v3.csv
examples/cinc17/REFERENCE-v3.csv
REFERENCE-v3.csv
```

---

## Secciones de la interfaz

### 1. Cargar ECG

Permite:

- arrastrar o seleccionar archivo;
- escribir datos opcionales de paciente;
- escribir diagnóstico conocido manualmente;
- usar ejemplos sintéticos rápidos;
- seleccionar derivación si el archivo tiene varios canales.

### 2. Resultado

Muestra:

- ritmo predominante;
- confianza media del ritmo predominante;
- distribución de clases;
- modelo usado para la predicción;
- comparación con etiqueta real si está disponible;
- ECG interactivo con bandas de color por intervalo;
- tabla de clasificación por tramos;
- descarga de informe PDF.

### 3. Detalle Técnico

Muestra:

- arquitectura cargada: ResNet-34 o CNN convencional;
- clases reconocidas;
- número de parámetros;
- checkpoint e ID del modelo;
- métricas exportadas por `evaluate.py`;
- matriz de confusión y F1 por clase si existen las imágenes.

Para generar esas métricas:

```bash
python examples/cinc17/evaluate.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet \
  --save_metrics_dir webapp/static/metrics
```

### 4. Experimentos

Muestra automáticamente resultados ya calculados de:

1. **ResNet-34 vs CNN convencional**.
2. **Robustez frente a perturbaciones ECG**.

La app no entrena ni evalúa todo el dataset desde el navegador; solo lee archivos generados previamente.

---

## Generar resultados para la pestaña Experimentos

### Comparación ResNet vs CNN

```bash
python examples/cinc17/compare_models.py \
  --data_json examples/cinc17/dev.json \
  --resnet_saved saved/cinc17_resnet \
  --cnn_saved saved/cinc17_cnn \
  --out_dir results/cinc17/resnet_vs_cnn
```

La app lee:

```text
results/cinc17/resnet_vs_cnn/comparison_metrics.json
```

### Robustez ResNet

```bash
python examples/cinc17/robustness.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_resnet \
  --out_dir results/cinc17/robustness_resnet \
  --seed 1234
```

La app lee:

```text
results/cinc17/robustness_resnet/robustness_metrics.json
```

### Robustez CNN

```bash
python examples/cinc17/robustness.py \
  --data_json examples/cinc17/dev.json \
  --saved saved/cinc17_cnn \
  --out_dir results/cinc17/robustness_cnn \
  --seed 1234
```

La app lee:

```text
results/cinc17/robustness_cnn/robustness_metrics.json
```

---

## Formatos de entrada admitidos

| Formato | Detalle |
|---|---|
| CSV fila | `300,12.0,12.1,...`; primer valor = frecuencia de muestreo. |
| CSV columna | una muestra por fila; si no hay frecuencia, asume 300 Hz. |
| `.mat` | lee la variable `val` si existe; preserva múltiples derivaciones. |
| `.dat` | PhysioNet formato 212. |
| `.npy` | array 1-D o 2-D. |

Si el archivo tiene varias derivaciones, la app pide elegir una antes de inferir.

---

## Re-muestreo automático

El modelo se entrena con CinC2017 a **300 Hz**. Si subes una señal a otra frecuencia, la app la re-muestrea automáticamente a 300 Hz con `scipy.signal.resample` y lo indica en pantalla.

Esto evita que el modelo interprete los latidos con una escala temporal incorrecta.

---

## Informe PDF

El botón **Descargar informe (PDF)** genera un PDF real en servidor mediante ReportLab.

Incluye:

- paciente y edad, si se ingresan;
- fecha;
- ritmo predominante;
- confianza;
- etiqueta real y coincidencia, si está disponible;
- parámetros del registro;
- distribución de clases;
- imagen del ECG con predicción por tramos.

Los archivos subidos se guardan con nombre temporal y se eliminan al terminar la predicción.

---

## Ejecutar en producción

El servidor de desarrollo de Flask no debe usarse en producción. Usa WSGI.

### Linux/macOS/servidor con gunicorn

```bash
pip install gunicorn

export ECG_SAVED="saved/cinc17_resnet"
export ECG_REFERENCE="dataset2017/REFERENCE-v3.csv"
# o usa un checkpoint exacto:
# export ECG_MODEL="saved/cinc17_resnet/<timestamp>/<checkpoint>.pt"

gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 "webapp.wsgi:app"
```

### Windows con waitress

PowerShell:

```powershell
$env:ECG_SAVED = "saved/cinc17_resnet"
$env:ECG_REFERENCE = "dataset2017/REFERENCE-v3.csv"
waitress-serve --listen=*:5000 "webapp.wsgi:app"
```

`webapp/wsgi.py` lee:

| Variable | Descripción |
|---|---|
| `ECG_SAVED` | Carpeta de checkpoints. |
| `ECG_MODEL` | Checkpoint exacto; tiene prioridad sobre `ECG_SAVED`. |
| `ECG_REFERENCE` | Ruta opcional a `REFERENCE-v3.csv`. |

---

## Acceso desde otra máquina

Para acceso en red local:

```bash
python webapp/app.py \
  --saved saved/cinc17_resnet \
  --reference dataset2017/REFERENCE-v3.csv \
  --host 0.0.0.0 \
  --port 5000
```

Luego abre desde otro dispositivo:

```text
http://<IP-de-tu-PC>:5000/
```

Recuerda permitir el puerto en el firewall.

Para una demo por internet sin configurar servidor público, puedes usar túnel HTTPS:

```bash
./webapp/run_public.sh
```

---

## HTTPS y seguridad

La app aplica cabeceras de seguridad desde `app.py`:

- `Content-Security-Policy`
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy`
- `Permissions-Policy`
- `Cache-Control: no-store`
- `Strict-Transport-Security` solo cuando se sirve por HTTPS.

Opciones de HTTPS:

```bash
./webapp/run_https.sh
```

O usa nginx/caddy/Render/Heroku/Railway delante del WSGI.

> Privacidad: no subas ECG reales ni datos personales a una demo pública sin las autorizaciones correspondientes.

---

## Endpoints

| Ruta | Método | Descripción |
|---|---:|---|
| `/` | GET | Interfaz principal. |
| `/predict` | POST | Clasifica una señal subida por multipart, campo `file`; opcional `label` y `channel`. |
| `/example` | POST | Genera y clasifica señal sintética: `normal`, `af` o `noise`. |
| `/report.pdf` | POST | Genera PDF desde el análisis actual. |
| `/models` | GET | Lista checkpoints disponibles y checkpoint activo. |
| `/use_model` | POST | Cambia checkpoint activo sin reiniciar; JSON `{ "model": "ruta_relativa.pt" }`. |
| `/metrics` | GET | Devuelve `webapp/static/metrics/metrics.json` si existe. |
| `/experiments` | GET | Devuelve comparación ResNet/CNN y robustez si existen en `results/cinc17/`. |
| `/reference` | GET | Estado de `REFERENCE-v3.csv`; permite consultar `?record=A00004`. |

Ejemplos:

```text
http://127.0.0.1:5000/metrics
http://127.0.0.1:5000/experiments
http://127.0.0.1:5000/reference?record=A00004
```

---

## Estructura

```text
webapp/
├── app.py              # Flask, rutas, seguridad, carga de métricas/experimentos/reference
├── prediction.py       # PredictionService, resampling, lectura de señales y gráficos
├── report_pdf.py       # generación del informe PDF
├── make_sample_csv.py  # genera CSV sintético de ejemplo
├── wsgi.py             # entrada WSGI; lee ECG_SAVED, ECG_MODEL, ECG_REFERENCE
├── run_https.sh        # demo HTTPS con certificado autofirmado o propio
├── run_public.sh       # túnel HTTPS para demo pública
├── templates/
│   └── index.html
└── static/
    ├── app.js
    ├── style.css
    ├── plotly.min.js
    └── metrics/        # metrics.json + figuras de evaluate.py, si se generan
```

---

## Notas para GitHub

- No incluyas checkpoints grandes si exceden las políticas del repositorio; considera Git LFS o publicar un release.
- No incluyas el dataset completo de CinC2017 si la licencia/acuerdo de PhysioNet no lo permite.
- Para reproducibilidad, sube scripts, configuraciones y reportes CSV/JSON/Markdown generados en `results/`, si el tamaño es razonable.
