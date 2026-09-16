# Resultados finales — ResNet-34 CINC2020-12

## Configuración experimental

- Datos: PhysioNet/CinC Challenge 2020 (6 subconjuntos), esquema agrupado v2 de 12 clases SNOMED.
- Splits: train 33 759 / validation 6 486 / test 6 421 ventanas (10 s, 12 derivaciones, 500 Hz).
- Punto de operación: temperature scaling por clase (tuneado en validación) + umbrales por clase
  optimizados con **F0.5** (objetivo: menos falsos positivos con F1-macro alto).
- Modelos: ResNet v1 (receta base, 80 épocas, selección por `val_loss`), ResNet v2
  (aumentación + label smoothing + 150 épocas + selección por F1-macro), Ensemble (promedio v1+v2).

## Tabla principal (test, n = 6421)

| Métrica | ResNet v1 | ResNet v2 | Ensemble |
|---|---|---|---|
| F1-macro | 0.683 | **0.718** | 0.710 |
| F1-micro | 0.731 | 0.745 | **0.747** |
| Precisión-macro | 0.818 | 0.789 | **0.824** |
| Recall-macro | 0.593 | **0.673** | 0.629 |
| Especificidad-macro | 0.976 | 0.974 | **0.977** |
| AUROC-macro | 0.943 | 0.938 | **0.951** |
| AUPRC-macro | 0.784 | 0.779 | **0.805** |
| Exact-match | 0.437 | **0.459** | 0.455 |

## F1 por clase (test)

| Clase | Soporte | v1 | v2 | Ensemble |
|---|---|---|---|---|
| NSR | 3249 | 0.906 | 0.898 | 0.903 |
| AF | 568 | 0.886 | 0.901 | 0.893 |
| STach | 360 | 0.741 | 0.823 | 0.813 |
| SB | 398 | 0.668 | 0.787 | 0.658 |
| AVBlock | 421 | 0.638 | 0.718 | 0.674 |
| AxisDev | 1072 | 0.708 | 0.708 | 0.695 |
| MI | 1651 | 0.681 | 0.692 | 0.679 |
| BBB | 1527 | 0.671 | 0.658 | 0.699 |
| VEctopy | 395 | 0.660 | 0.675 | 0.671 |
| TAb | 1786 | 0.608 | 0.623 | 0.654 |
| LVH | 800 | 0.525 | 0.594 | 0.620 |
| AEctopy_Junctional | 374 | 0.510 | 0.540 | 0.562 |

## Hallazgos

1. **Receta v2: +3.45 pts F1-macro** (0.683 → 0.718). La v1 se estancó en época 37 con gap
   train/val de 0.125 (sobreajuste); la v2 (aumentación, smoothing, regularización, selección por F1)
   generaliza mejor, sobre todo en recall (+8 pts).
2. **Ablación negativa NSR-exclusivo.** Forzar "solo NSR" cuando P(NSR) ≥ 0.90 costó
   **−11.3 pts F1-macro sin ganancia de precisión** (0.605 vs 0.718). Causa: NSR describe el ritmo y
   coexiste legítimamente con clases morfológicas (NSR+LVH, NSR+BBB…). Regla descartada.
   El daño se concentró en PTB-XL (etiquetado multietiqueta completo); fue neutra en CPSC
   (etiquetado casi single-label).
3. **Ensemble: mejor discriminación y precisión.** AUROC +1.3, AUPRC +2.6 y precisión +3.5 sobre v2,
   a costa de −0.8 F1-macro y −4.4 recall. Eleva las clases morfológicas débiles (TAb, LVH, BBB)
   pero diluye SB/AVBlock donde v2 es muy superior. Bajo el objetivo F0.5 declarado, el ensemble gana.
4. **Calibración validada:** temperature scaling aporta ~+5 pts F1 (0.670 sin temps → 0.718 con temps).
5. **Robustez** (ResNet v2, F1-macro; base 0.670 sin temps): degradación guiada por recall con
   precisión estable (0.79–0.84) — el modelo se calla antes de alucinar. Robusto a perder
   1 derivación (−2.3) y escala ×1.5 (−3.1); sensible a ruido 0.10 (−21) y wander 0.20 (−23),
   niveles que exceden las magnitudes de entrenamiento.
6. **Por origen** (F1-macro): PTB-XL 0.725 (n=3221) > Georgia 0.630 > CPSC-extra 0.515 >
   CPSC 0.433 (clases ausentes hunden el macro pese a exact-match 0.642) > PTB 0.258 (n=99) >
   INCART 0.160 (n=10, no concluyente).
7. **Métrica Challenge 27 códigos: 0.2348** (F1-macro-27 0.302). Referencia interna solamente:
   no comparable al leaderboard (esquema agrupado de 12 + punto conservador).

## Modelo final recomendado

**Ensemble v1+v2** (promedio simple, temperaturas por modelo, umbrales F0.5): mejor precisión
(0.824), discriminación (AUROC 0.951) y F0.5 global, con F1-macro 0.710 (≈ v2).
ResNet v2 sola queda como mejor modelo individual en F1-macro (0.718). La webapp demuestra
el ensemble (`--model` + `--model-b` + `--temperatures`/`--temperatures-b` + `--eval-dir ensemble`).

## Artefactos

- `eval-resnet/`, `eval-resnet-v2-noexcl/`, `ensemble-noexcl/`: métricas, umbrales, predicciones, curvas.
- `exp-files/model_comparison.csv`: v1 vs v2. `comparacion_ensemble.csv`: v2 vs ensemble.
- `exp-files/robustness.csv`, `metrica-challenge/`, `errores_origen.csv`.
