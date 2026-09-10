# Webapp ECG CINC2020-12

Aplicación Flask tipo dashboard para analizar ECG de 12 derivaciones con el modelo multilabel CINC2020-12.

## Funcionalidades

- Carga por arrastrar/seleccionar archivo.
- Soporte para CSV de 12 derivaciones.
- Soporte para par WFDB `.hea + .mat` del mismo registro.
- Conversión automática del par `.hea + .mat` a CSV preprocesado descargable.
- Selección automática del mejor checkpoint disponible en la carpeta `saved/` configurada.
- Predicción multilabel con probabilidades por clase.
- Uso automático de `thresholds_validation.csv` si existe; si no, usa threshold global de respaldo.
- Trazado ECG con cuadrícula tipo papel milimetrado.
- Reporte PDF generado en servidor.
- Cambio de idioma Español/Inglés desde la interfaz.
- Comparación con etiquetas reales:
  - automática desde el campo `Dx` cuando se sube `.hea + .mat`;
  - manual para CSV mediante clases o códigos SNOMED separados por coma.
- Detalle técnico del registro, checkpoint, threshold, preprocesamiento y comparación real vs predicho.
- Sección de métricas y experimentos si existen archivos en `results/`.

## Ejecutar sin variables de entorno

Desde la raíz del proyecto, la opción recomendada es dejar que la app elija automáticamente el mejor checkpoint dentro de `saved/`:

```bash
python webapp/app.py --saved saved
```

También puede usar una carpeta específica:

```bash
python webapp/app.py --saved saved/cinc2020
```

Para una demostración reproducible puede fijar un checkpoint exacto por CLI. La interfaz web no muestra selector de modelo:

```bash
python webapp/app.py --model saved/cinc2020/cinc2020_resnet/<run>/best.pt
```

Si el archivo de umbrales está en una ruta no estándar, páselo explícitamente:

```bash
python webapp/app.py \
  --saved saved \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv
```

Al iniciar, la consola imprime la ruta de thresholds detectada y los valores de `LVH` y `NSR`.

Para demos puede dejar activo el fallback normal: si ninguna clase supera su umbral calibrado y `P(NSR) >= 0.40`, la salida final añade `NSR` como postprocesamiento explícito. No cambia las probabilidades del modelo ni debe reportarse como mejora de entrenamiento. Para desactivarlo:

```bash
python webapp/app.py \
  --saved saved \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv \
  --normal-fallback-min-prob 0
```

Parámetros útiles:

```bash
python webapp/app.py \
  --saved saved \
  --uploads webapp/uploads \
  --results webapp/results \
  --host 0.0.0.0 \
  --port 5002
```

## Formato CSV

El CSV debe contener las 12 derivaciones estándar como columnas:

```text
I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6
```

Opcionalmente puede empezar con una línea de frecuencia:

```text
# Sampling Rate: 500 Hz
```

Si no se indica frecuencia, se asume 500 Hz.

Para comparar contra etiquetas reales en CSV, use el campo opcional de la interfaz con clases o SNOMED:

```text
NSR, AF
426783006,164889003
```

## Formato WFDB

Seleccione ambos archivos del mismo registro:

```text
A0001.hea
A0001.mat
```

La app valida que el stem coincida, lee el header, carga la señal, aplica el mismo preprocesamiento del pipeline, compara predicción vs `Dx` y genera un CSV convertido descargable.

## Resultados experimentales en la interfaz

La sección de métricas se completa al generar:

```bash
python examples/cinc2020/evaluate.py \
  examples/cinc2020/config.json \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --output-dir results/cinc2020_12_resnet
```

Ese comando también crea `thresholds_validation.csv`. Si el archivo existe, la webapp lo detecta automáticamente y usa umbrales por clase, que suelen ser más adecuados que 0.5 en problemas multilabel desbalanceados.

Para evaluar el efecto del fallback NSR en el conjunto completo, úselo explícitamente en evaluación:

```bash
python examples/cinc2020/evaluate.py \
  examples/cinc2020/config.json \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --output-dir results/cinc2020_12_resnet_fallback \
  --normal-fallback-min-prob 0.40
```

Para depurar un registro individual y comparar la señal de la webapp contra la señal guardada en HDF5:

```bash
python examples/cinc2020/debug_record_prediction.py \
  --config examples/cinc2020/config.json \
  --checkpoint saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  --record E00001 \
  --hea training/georgia/g1/E00001.hea \
  --mat training/georgia/g1/E00001.mat \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv
```

La sección de comparación se completa con:

```bash
python examples/cinc2020/compare_models.py \
  --resnet results/cinc2020_12_resnet \
  --cnn results/cinc2020_12_cnn \
  --output results/cinc2020_12/model_comparison.csv
```

La sección de robustez se completa con:

```bash
python examples/cinc2020/robustness.py \
  saved/cinc2020/cinc2020_resnet/<run>/best.pt \
  data/cinc2020_12/test.h5 \
  --thresholds results/cinc2020_12_resnet/thresholds_validation.csv \
  --output results/cinc2020_12/robustness.csv
```

## Mejoras implementadas

- Validación de combinación de archivos en cliente y servidor.
- Botón de limpieza para reiniciar la carga y los paneles.
- Guardado persistente de tema claro/oscuro e idioma en el navegador.
- Comparación multilabel por registro con TP, FP, FN, F1 y Jaccard.
- PDF con resumen técnico, probabilidades, comparación real vs predicho y trazado.

## Nota

La webapp es una herramienta académica. Las predicciones no constituyen diagnóstico médico.
