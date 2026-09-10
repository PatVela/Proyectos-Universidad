import os
import glob
from collections import defaultdict

def verificar_heas_faltantes(data_dir="dataset2020"):
    print(f"Escaneando carpetas en '{data_dir}'...\n")
    
    # Buscar todos los archivos .mat
    mat_files = glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True)
    
    if not mat_files:
        print(f"No se encontraron archivos .mat en '{data_dir}'.")
        return

    faltantes_por_carpeta = defaultdict(list)
    total_mat = len(mat_files)
    total_ok = 0
    total_faltantes = 0

    for mat_path in mat_files:
        hea_path = mat_path[:-4] + ".hea"
        
        if not os.path.exists(hea_path):
            carpeta = os.path.dirname(mat_path)
            archivo = os.path.basename(mat_path)
            faltantes_por_carpeta[carpeta].append(archivo)
            total_faltantes += 1
        else:
            total_ok += 1

    print(f"==== RESUMEN DEL ESCANEO ====")
    print(f"Total de señales (.mat) encontradas: {total_mat}")
    print(f"Registros OK con su .hea:             {total_ok}")
    print(f"Archivos .mat sin .hea:               {total_faltantes}\n")

    if total_faltantes == 0:
        print("¡Todo en orden! Todos los archivos .mat tienen su cabecera .hea.")
        return

    print("==== DETALLE DE CARPETAS AFECTADAS ====")
    for carpeta, archivos in faltantes_por_carpeta.items():
        print(f"\nCarpeta: {carpeta}")
        print(f"  Faltan {len(archivos)} archivos .hea")
        print(f"  Ejemplos con .mat pero sin .hea: {archivos[:5]}")

if __name__ == "__main__":
    verificar_heas_faltantes("dataset2020")