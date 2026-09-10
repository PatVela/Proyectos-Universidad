import os
import glob
import urllib.request

BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/"

def descargar_heas_faltantes(data_dir="dataset2020"):
    print("Buscando archivos .hea faltantes en el dataset local...")
    mat_files = glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True)
    
    faltantes = []
    for mat in mat_files:
        hea = mat[:-4] + ".hea"
        if not os.path.exists(hea):
            faltantes.append((mat, hea))

    if not faltantes:
        print("¡No faltan archivos .hea! Todo está completo.")
        return

    print(f"Se encontraron {len(faltantes)} archivos .hea faltantes. Descargando en sus subcarpetas...\n")

    exitos = 0
    errores = 0

    for i, (mat_path, hea_path) in enumerate(faltantes, 1):
        # Extraer la ruta relativa respetando la jerarquía de carpetas
        partes = os.path.normpath(hea_path).split(os.sep)
        if "1.0.2" in partes:
            idx_base = partes.index("1.0.2") + 1
            subpath_url = "/".join(partes[idx_base:])
        else:
            idx_base = partes.index("training") if "training" in partes else 0
            subpath_url = "/".join(partes[idx_base:])

        url = BASE_URL + subpath_url
        nombre_archivo = os.path.basename(hea_path)
        
        print(f"[{i}/{len(faltantes)}] Descargando {nombre_archivo} en {os.path.dirname(hea_path)} ... ", end="", flush=True)

        try:
            os.makedirs(os.path.dirname(hea_path), exist_ok=True)
            urllib.request.urlretrieve(url, hea_path)
            print("OK")
            exitos += 1
        except Exception as e:
            print(f"ERROR ({e})")
            errores += 1

    print(f"\nFinalizado. Descargados con éxito: {exitos} | Errores: {errores}")

if __name__ == "__main__":
    descargar_heas_faltantes("dataset2020")