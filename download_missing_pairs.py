import os
import re
import urllib.request

BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/training/"

def obtener_archivos_remotos(url_subcarpeta):
    try:
        req = urllib.request.Request(url_subcarpeta, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            heas = re.findall(r'href="([^"]+\.hea)"', html)
            return set(f[:-4] for f in heas)
    except Exception as e:
        print(f" Error al conectar con {url_subcarpeta}: {e}")
        return set()

def descargar_pares_faltantes(data_dir="dataset2020"):
    base_path = os.path.join(data_dir, "physionet.org", "files", "challenge-2020", "1.0.2", "training")
    
    subcarpetas_rel = [
        "cpsc_2018/g1", "cpsc_2018/g2", "cpsc_2018/g3", "cpsc_2018/g4", "cpsc_2018/g5", "cpsc_2018/g6", "cpsc_2018/g7",
        "cpsc_2018_extra/g1", "cpsc_2018_extra/g2", "cpsc_2018_extra/g3", "cpsc_2018_extra/g4",
        "georgia/g1", "georgia/g2", "georgia/g3", "georgia/g4", "georgia/g5", "georgia/g6", "georgia/g7", "georgia/g8", "georgia/g9", "georgia/g10", "georgia/g11",
        "ptb/g1",
        "ptb-xl/g1", "ptb-xl/g2", "ptb-xl/g3", "ptb-xl/g4", "ptb-xl/g5", "ptb-xl/g6", "ptb-xl/g7", "ptb-xl/g8", "ptb-xl/g9", "ptb-xl/g10", 
        "ptb-xl/g11", "ptb-xl/g12", "ptb-xl/g13", "ptb-xl/g14", "ptb-xl/g15", "ptb-xl/g16", "ptb-xl/g17", "ptb-xl/g18", "ptb-xl/g19", "ptb-xl/g20", "ptb-xl/g21", "ptb-xl/g22",
        "st_petersburg_incart/g1"
    ]

    descargas_pendientes = []

    print("--- FASE 1: Identificando archivos faltantes ---")

    for i, rel_folder in enumerate(subcarpetas_rel, 1):
        local_folder = os.path.normpath(os.path.join(base_path, rel_folder))
        os.makedirs(local_folder, exist_ok=True)
        
        print(f"[{i}/{len(subcarpetas_rel)}] Verificando en servidor: {rel_folder}...", end="", flush=True)
        
        archivos_locales = set(os.listdir(local_folder)) if os.path.exists(local_folder) else set()
        url_folder = BASE_URL + rel_folder + "/"
        
        registros_remotos = obtener_archivos_remotos(url_folder)
        faltantes_carpeta = 0
        
        for reg in registros_remotos:
            if (reg + ".hea") not in archivos_locales:
                descargas_pendientes.append((url_folder + reg + ".hea", os.path.join(local_folder, reg + ".hea")))
                faltantes_carpeta += 1
            if (reg + ".mat") not in archivos_locales:
                descargas_pendientes.append((url_folder + reg + ".mat", os.path.join(local_folder, reg + ".mat")))
                faltantes_carpeta += 1

        print(f" OK (Faltan {faltantes_carpeta} archivos)")

    print("\n--- FASE 2: Descargando archivos faltantes ---")

    if not descargas_pendientes:
        print("¡No hay nada pendiente! Tu dataset ya está 100% completo.")
        return

    print(f"Iniciando la descarga de {len(descargas_pendientes)} archivos...\n")

    exitos, errores = 0, 0
    total = len(descargas_pendientes)

    for i, (url, destino) in enumerate(descargas_pendientes, 1):
        nombre = os.path.basename(destino)
        carpeta_padre = os.path.basename(os.path.dirname(destino))
        
        print(f"[{i}/{total}] Descargando {carpeta_padre}/{nombre} ... ", end="", flush=True)

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(destino, 'wb') as out_file:
                out_file.write(resp.read())
            print("OK")
            exitos += 1
        except Exception as e:
            print(f"ERROR ({e})")
            errores += 1

    print(f"\nProceso finalizado. Descargados con éxito: {exitos} | Errores: {errores}")

if __name__ == "__main__":
    descargar_pares_faltantes()