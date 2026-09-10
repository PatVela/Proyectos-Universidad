import os
import glob
import urllib.request

BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/"

def descargar_faltantes(data_dir="dataset2020"):
    print("Buscando archivos .mat faltantes en la estructura local...")
    hea_files = glob.glob(os.path.join(data_dir, '**', '*.hea'), recursive=True)
    
    faltantes = []
    for hea in hea_files:
        mat = hea[:-4] + ".mat"
        if not os.path.exists(mat):
            faltantes.append((hea, mat))

    if not faltantes:
        print("¡No faltan archivos! Todos los .hea tienen su .mat.")
        return

    print(f"Se encontraron {len(faltantes)} archivos .mat faltantes. Descargando en sus subcarpetas correspondientes...\n")

    exitos = 0
    errores = 0

    for i, (hea_path, mat_path) in enumerate(faltantes, 1):
        # 1. Extraer la ruta relativa desde la carpeta '1.0.2'
        # Ejemplo: 'training/cpsc_2018/g2/A1473.mat'
        partes = os.path.normpath(mat_path).split(os.sep)
        if "1.0.2" in partes:
            idx_base = partes.index("1.0.2") + 1
            subpath_url = "/".join(partes[idx_base:])
        else:
            # Fallback en caso de que varíe la estructura
            idx_base = partes.index("training") if "training" in partes else 0
            subpath_url = "/".join(partes[idx_base:])

        url = BASE_URL + subpath_url
        nombre_archivo = os.path.basename(mat_path)
        
        print(f"[{i}/{len(faltantes)}] Guardando en {os.path.dirname(mat_path)}\\{nombre_archivo} ... ", end="", flush=True)

        try:
            # Asegura la creación de la subcarpeta local
            os.makedirs(os.path.dirname(mat_path), exist_ok=True)
            
            # Descarga el archivo directamente en la ubicación del .hea
            urllib.request.urlretrieve(url, mat_path)
            print("OK")
            exitos += 1
        except Exception as e:
            print(f"ERROR ({e})")
            errores += 1

    print(f"\nFinalizado. Exitosos: {exitos} | Errores: {errores}")

if __name__ == "__main__":
    descargar_faltantes("dataset2020")