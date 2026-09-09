# Experimento: robustez frente a perturbaciones ECG

Modelo: **ResNet-34**  
Checkpoint: `C:\Users\juerg\Investigacion\TIF-Biomedica\saved\cinc17_resnet\1788946337-545\0.353-0.868-019-0.234-0.922.pt`  
Dataset: `C:\Users\juerg\Investigacion\TIF-Biomedica\examples\cinc17\dev.json`  
Semilla de perturbaciones: `1234`

| Condición | Descripción | Accuracy | Macro-F1 | Weighted-F1 | Challenge-F1 | n |
|---|---|---:|---:|---:|---:|---:|
| original | ECG sin modificación | 0.8689 | 0.7653 | 0.8651 | 0.8498 | 854 |
| noise_snr_20p0db | Ruido gaussiano aditivo con SNR=20.0 dB | 0.8689 | 0.7767 | 0.8659 | 0.8451 | 854 |
| noise_snr_10p0db | Ruido gaussiano aditivo con SNR=10.0 dB | 0.8173 | 0.6983 | 0.8216 | 0.7815 | 854 |
| noise_snr_5p0db | Ruido gaussiano aditivo con SNR=5.0 dB | 0.6768 | 0.4935 | 0.6950 | 0.5761 | 854 |
| baseline_wander | Deriva de línea base senoidal: 0.33 Hz, amplitud=0.3×STD | 0.8724 | 0.7762 | 0.8683 | 0.8488 | 854 |
| amplitude_x0p5 | Escalamiento de amplitud ×0.5 | 0.8700 | 0.7554 | 0.8679 | 0.8405 | 854 |
| amplitude_x1p5 | Escalamiento de amplitud ×1.5 | 0.8618 | 0.7486 | 0.8569 | 0.8432 | 854 |
| crop_10p0s | Recorte central a 10.0 segundos | 0.7916 | 0.6615 | 0.7756 | 0.7320 | 854 |
| crop_20p0s | Recorte central a 20.0 segundos | 0.8466 | 0.7272 | 0.8391 | 0.8233 | 854 |

> Challenge-F1 es el promedio oficial sobre N/A/O. Las perturbaciones se aplican al ECG crudo y luego se usa el mismo preprocesador del entrenamiento.
