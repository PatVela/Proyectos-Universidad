import os
import glob
import urllib.request
import re

BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/training/"

def obtener_archivos_remotos(url_subcarpeta):
    """Lee el índice HTML de una carpeta en PhysioNet y extrae los nombres de registros."""
    try:
        req = urllib.request.Request(url_subcarpeta, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            # Extraer nombres de archivos .hea del HTML
            heas = re.findall(r'href="([^"]+\.hea)"', html)
            # Eliminar la extensión .hea para tener solo el nombre base
            return set(f[:-4] for f in heas)
    except Exception as e:
        print(f"Error al leer índice remoto {url_subcarpeta}: {e}")
        return set()

def verificar_parejas_faltantes(data_dir="dataset2020"):
    base_path = os.path.join(data_dir, "physionet.org", "files", "challenge-2020", "1.0.2", "training")
    
    if not os.path.exists(base_path):
        print(f"Error: No se encontró la carpeta base '{base_path}'.")
        return

    # 1. Encontrar todas las subcarpetas locales de datos (ej. cpsc_2018/g1)
    carpetas_locales = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  # Carpetas finales que contienen los archivos
            carpetas_locales.append(root)

    print(f"Escaneando {len(carpetas_locales)} subcarpetas contra el servidor de PhysioNet...\n")

    faltan_ambos = []
    total_remotos = 0

    for i, carpeta in enumerate(carpetas_locales, 1):
        rel_folder = os.path.relpath(carpeta, base_path).replace('\\', '/')
        url_folder = BASE_URL + rel_folder + "/"
        
        print(f"[{i}/{len(carpetas_locales)}] Verificando remoto: {rel_folder} ... ", end="", flush=True)
        
        registros_remotos = obtener_archivos_remotos(url_folder)
        total_remotos += len(registros_remotos)
        
        # Archivos locales en esta subcarpeta
        archivos_locales = set(os.listdir(carpeta))
        
        locales_incompletos = 0
        for reg in registros_remotos:
            tiene_hea = (reg + ".hea") in archivos_locales
            tiene_mat = (reg + ".mat") in archivos_locales
            
            if not tiene_hea and not tiene_mat:
                faltan_ambos.append(f"{rel_folder}/{reg}")
                locales_incompletos += 1

        print(f"OK ({len(registros_remotos)} remotos | Faltan ambos: {locales_incompletos})")

    # Reporte Final
    print("\n==== RESUMEN DE INTEGRIDAD CONTRA PHYSIONET ====")
    print(f"Total de registros evaluados en servidor: {total_remotos}")
    print(f"Registros donde FALTA AMBOS (.hea y .mat): {len(faltan_ambos)}\n")

    if faltan_ambos:
        print("==== LISTA DE REGISTROS QUE FALTAN POR COMPLETO (AMBOS ARCHIVOS) ====")
        for rec in faltan_ambos:
            print(f" - {rec}")
    else:
        print("¡Excelente! No falta ningún par (.hea + .mat) en ninguna carpeta.")

if __name__ == "__main__":
    verificar_parejas_faltantes()