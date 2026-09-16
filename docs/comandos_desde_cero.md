# Guía de ejecución desde cero — ResNet-34 CINC2020-12

Tres variantes del mismo pipeline: **Linux (bash)**, **Windows (PowerShell)** y
**Windows + Visual Studio Code**. Sin entornos virtuales ni variables de entorno.

Metodología final: ResNet v1 + v2, temperature scaling, umbrales F0.5, ensemble v1+v2
como modelo final (F1-macro 0.710, precisión-macro 0.824, AUROC 0.951 en test).
Ver [resultados_finales.md](resultados_finales.md).

## Nombres usados

| Carpeta/archivo | Contenido |
|---|---|
| `eval-resnet` | Calibración + evaluación ResNet v1 |
| `eval-resnet-v2` | Calibración + evaluación ResNet v2 |
| `exp-files` | `model_comparison.csv` (v1 vs v2) + `robustness.csv` |
| `ensemble` | Ensemble ResNet v1 + v2 (modelo final) |
| `comparacion_ensemble.csv` | Ensemble vs mejor individual |
| `metrica-challenge` | Métrica oficial estilo Challenge |
| `errores_origen.csv` | Errores por hospital |
| `resnet12.onnx` | Modelo exportado |

## Notas de hardware (RTX 3050 4 GB / 24 GB RAM)

- `batch_size: 64` cabe en 4 GB con AMP. Si hay `CUDA out of memory`, bajar a 32 en el config.
- La precarga a RAM usa ~9 GB; no tocar `num_workers: 0` en Windows.
- Laptop: conectada a corriente y plan de máximo rendimiento.
- Tiempos aprox.: HDF5 20–40 min · v1 ~35 min · v2 ~2.3 h · calibración min · evaluación ~10 min · ensemble ~20 min.

---

## Variante A — Linux (bash)

```bash
git clone -b Paper_Replica_V3 https://github.com/PatVela/Proyectos-Universidad.git
cd Proyectos-Universidad
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())"
```

Debe decir `cuda: True` (si no, actualizar el driver NVIDIA o usar `cu128`).
Smoke test:

```bash
python examples/cinc2020/make_syntethic.py
python -m ecg.train examples/cinc2020/config_syntethic.json -e smoke_test --epochs 1 --device cpu --no-amp
pytest tests/ -q
```

Debe decir `35 passed`. Pipeline:

```bash
bash descargar_datos_2020.sh
python examples/cinc2020/build_datasets.py --data_dir dataset2020 --output_dir data/cinc2020_12 --workers 8 --norm_mode physical --train_windows_max 4
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2
```

Opcional, segunda semilla v2:

```bash
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2b --seed 44
```

Checkpoints (verificar las rutas impresas; borrar runs abortados a medias):

```bash
RESNET=$(ls -t saved/cinc2020/cinc2020_resnet/*/best.pt | head -1)
RESNET2=$(ls -t saved/cinc2020/cinc2020_resnet_v2*/best.pt | head -1)
echo "$RESNET"; echo "$RESNET2"
python examples/cinc2020/calibrate.py --checkpoint "$RESNET" --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet
python examples/cinc2020/calibrate.py --checkpoint "$RESNET2" --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet-v2
python examples/cinc2020/evaluate.py examples/cinc2020/config.json "$RESNET" --output-dir eval-resnet --temperatures eval-resnet/temperatures_validation.csv --threshold-beta 0.5
python examples/cinc2020/evaluate.py examples/cinc2020/config_resnet_v2.json "$RESNET2" --output-dir eval-resnet-v2 --temperatures eval-resnet-v2/temperatures_validation.csv --threshold-beta 0.5
python examples/cinc2020/compare_models.py --eval-a eval-resnet --eval-b eval-resnet-v2 --label-a "ResNet v1" --label-b "ResNet v2" --output exp-files/model_comparison.csv
```

Elegir el ganador (ajustar según la tabla; aquí se asume v2) y continuar:

```bash
MEJOR="$RESNET2"
EVALMEJOR="eval-resnet-v2"
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json "$RESNET" "$RESNET2" --output-dir ensemble --alpha 0.5 --threshold-beta 0.5 --temperatures-a eval-resnet/temperatures_validation.csv --temperatures-b eval-resnet-v2/temperatures_validation.csv
python examples/cinc2020/compare_models.py --eval-a "$EVALMEJOR" --eval-b ensemble --label-a "ResNet mejor" --label-b "Ensemble" --output comparacion_ensemble.csv
python examples/cinc2020/robustness.py "$MEJOR" data/cinc2020_12/test.h5 --thresholds "$EVALMEJOR/thresholds_validation.csv" --output exp-files/robustness.csv
python examples/cinc2020/challenge_score.py --checkpoint "$MEJOR" --test-h5 data/cinc2020_12/test.h5 --thresholds "$EVALMEJOR/thresholds_validation.csv" --output-dir metrica-challenge
python examples/cinc2020/error_analysis.py --predictions "$EVALMEJOR/predictions_test.csv" --test-h5 data/cinc2020_12/test.h5 --output errores_origen.csv
```

Diagnóstico de registros ejemplo:

```bash
HEA1=$(find dataset2020 -name E00001.hea | head -1); MAT1="${HEA1%.hea}.mat"
python examples/cinc2020/diagnose_prediction.py --checkpoint "$MEJOR" --config examples/cinc2020/config.json --thresholds "$EVALMEJOR/thresholds_validation.csv" --record E00001 --hea "$HEA1" --mat "$MAT1"
HEA14=$(find dataset2020 -name E00014.hea | head -1); MAT14="${HEA14%.hea}.mat"
python examples/cinc2020/diagnose_prediction.py --checkpoint "$MEJOR" --config examples/cinc2020/config.json --thresholds "$EVALMEJOR/thresholds_validation.csv" --record E00014 --hea "$HEA14" --mat "$MAT14"
```

Webapp con el ensemble final:

```bash
python webapp/app.py --saved saved --eval-dir ensemble --exp-dir exp-files --model "$RESNET" --model-b "$RESNET2" --alpha 0.5 --temperatures eval-resnet/temperatures_validation.csv --temperatures-b eval-resnet-v2/temperatures_validation.csv --port 5002
```

Abrir http://127.0.0.1:5002. Opcionales:

```bash
python examples/cinc2020/export_onnx.py --checkpoint "$MEJOR" --output resnet12.onnx
docker build -t ecg-cinc2020 .
docker run --rm -p 5002:5002 -v ./saved:/app/saved:ro ecg-cinc2020
```

---

## Variante B — Windows (PowerShell)

Mismos pasos; cambian la selección de archivos y variables. Desde `powershell.exe` en la
carpeta del proyecto (rama `Paper_Replica_V3`, código actualizado):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, '| cuda:', torch.cuda.is_available())"
python examples/cinc2020/make_syntethic.py
python -m ecg.train examples/cinc2020/config_syntethic.json -e smoke_test --epochs 1 --device cpu --no-amp
pytest tests/ -q
```

Datos (requiere Git Bash instalado; si `bash` no se reconoce, ejecutar esa línea en Git Bash):

```powershell
bash descargar_datos_2020.sh
python examples/cinc2020/build_datasets.py --data_dir dataset2020 --output_dir data/cinc2020_12 --workers 8 --norm_mode physical --train_windows_max 4
python -m ecg.train examples/cinc2020/config.json -e cinc2020_resnet
python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2
```

Opcional: `python -m ecg.train examples/cinc2020/config_resnet_v2.json -e cinc2020_resnet_v2b --seed 44`.
Checkpoints:

```powershell
$resnet = Get-ChildItem saved/cinc2020/cinc2020_resnet/*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$resnet2 = Get-ChildItem saved/cinc2020/cinc2020_resnet_v2*/best.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$resnet
$resnet2
python examples/cinc2020/calibrate.py --checkpoint $resnet --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet
python examples/cinc2020/calibrate.py --checkpoint $resnet2 --val-h5 data/cinc2020_12/val.h5 --output-dir eval-resnet-v2
python examples/cinc2020/evaluate.py examples/cinc2020/config.json $resnet --output-dir eval-resnet --temperatures eval-resnet/temperatures_validation.csv --threshold-beta 0.5
python examples/cinc2020/evaluate.py examples/cinc2020/config_resnet_v2.json $resnet2 --output-dir eval-resnet-v2 --temperatures eval-resnet-v2/temperatures_validation.csv --threshold-beta 0.5
python examples/cinc2020/compare_models.py --eval-a eval-resnet --eval-b eval-resnet-v2 --label-a "ResNet v1" --label-b "ResNet v2" --output exp-files/model_comparison.csv
$mejor = $resnet2
$evalMejor = "eval-resnet-v2"
python examples/cinc2020/ensemble_evaluate.py examples/cinc2020/config.json $resnet $resnet2 --output-dir ensemble --alpha 0.5 --threshold-beta 0.5 --temperatures-a eval-resnet/temperatures_validation.csv --temperatures-b eval-resnet-v2/temperatures_validation.csv
python examples/cinc2020/compare_models.py --eval-a "$evalMejor" --eval-b ensemble --label-a "ResNet mejor" --label-b "Ensemble" --output comparacion_ensemble.csv
python examples/cinc2020/robustness.py $mejor data/cinc2020_12/test.h5 --thresholds "$evalMejor/thresholds_validation.csv" --output exp-files/robustness.csv
python examples/cinc2020/challenge_score.py --checkpoint $mejor --test-h5 data/cinc2020_12/test.h5 --thresholds "$evalMejor/thresholds_validation.csv" --output-dir metrica-challenge
python examples/cinc2020/error_analysis.py --predictions "$evalMejor/predictions_test.csv" --test-h5 data/cinc2020_12/test.h5 --output errores_origen.csv
$hea1 = Get-ChildItem dataset2020 -Recurse -Filter E00001.hea | Select-Object -First 1 -ExpandProperty FullName
$mat1 = [System.IO.Path]::ChangeExtension($hea1, '.mat')
python examples/cinc2020/diagnose_prediction.py --checkpoint $mejor --config examples/cinc2020/config.json --thresholds "$evalMejor/thresholds_validation.csv" --record E00001 --hea $hea1 --mat $mat1
$hea14 = Get-ChildItem dataset2020 -Recurse -Filter E00014.hea | Select-Object -First 1 -ExpandProperty FullName
$mat14 = [System.IO.Path]::ChangeExtension($hea14, '.mat')
python examples/cinc2020/diagnose_prediction.py --checkpoint $mejor --config examples/cinc2020/config.json --thresholds "$evalMejor/thresholds_validation.csv" --record E00014 --hea $hea14 --mat $mat14
python webapp/app.py --saved saved --eval-dir ensemble --exp-dir exp-files --model $resnet --model-b $resnet2 --alpha 0.5 --temperatures eval-resnet/temperatures_validation.csv --temperatures-b eval-resnet-v2/temperatures_validation.csv --port 5002
python examples/cinc2020/export_onnx.py --checkpoint $mejor --output resnet12.onnx
docker build -t ecg-cinc2020 .
docker run --rm -p 5002:5002 -v ./saved:/app/saved:ro ecg-cinc2020
```

---

## Variante C — Windows + Visual Studio Code

Igual que la variante B, ejecutada en la terminal integrada (Ctrl+Ñ, perfil PowerShell):

1. Abrir la carpeta del proyecto en VS Code (`code .` o Archivo → Abrir carpeta).
2. Instalar la extensión **Python** de Microsoft; elegir intérprete con `Ctrl+Shift+P` →
   *Python: Select Interpreter* (el mismo donde se hace `pip install`).
3. Para `descargar_datos_2020.sh`, abrir una terminal **Git Bash** (menú ⌄ junto al `+` →
   *Git Bash*) si PowerShell no reconoce `bash`; el resto corre en PowerShell.
4. Ejecutar por bloques la variante B (no pegar todo de golpe: verificar GPU tras instalar,
   verificar rutas de checkpoints tras entrenar, y leer la tabla comparativa antes de
   definir `$mejor`/`$evalMejor`).
5. Los CSV de métricas (`metrics_global.csv`, `model_comparison.csv`, `robustness.csv`) y los
   `history.csv`/`config_used.json` de cada run pueden abrirse directamente en el editor.

Checklist de arranque del entrenamiento v2 (si algo falla, Ctrl+C):

```text
Device       : cuda
Save dir     : ...\cinc2020_resnet_v2\...
Aumentación activada (solo train): {...}
Monitorizando: val_f1_macro
Época 1/150
```

## Si algo falla (todas las variantes)

- `CUDA out of memory`: `batch_size` 64 → 32 en el config y relanzar.
- `Device: cpu`: parar, verificar `cuda: True` y relanzar. Nunca entrenar en CPU.
- Duda del config usado: abrir `<run>/config_used.json` y revisar `max_epochs`,
  `early_stopping_metric`, `augment`.
- `pip install -r requirements.txt` falla en `gevent`: comentarlo con `#` (solo sirve
  para despliegue productivo).
- Época lenta (>3 min): confirmar `Precargando a RAM`; cerrar apps pesadas.
- Webapp vacía en métricas: revisar `--eval-dir`/`--exp-dir`; `/health` muestra qué detectó
  (incluye `ensemble`, `ECG_MODEL_B` y temperaturas A/B).
