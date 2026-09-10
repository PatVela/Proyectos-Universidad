import os
import glob
from collections import defaultdict

def verificar_archivos_faltantes(data_dir="dataset2020"):
    print(f"Escaneando carpetas en '{data_dir}'...\n")
    
    # Buscar todos los archivos .hea recursivamente
    hea_files = glob.glob(os.path.join(data_dir, '**', '*.hea'), recursive=True)
    
    if not hea_files:
        print(f"No se encontraron archivos .hea en '{data_dir}'. Revisa la ruta.")
        return

    faltantes_por_carpeta = defaultdict(list)
    total_hea = len(hea_files)
    total_ok = 0
    total_faltantes = 0

    for hea_path in hea_files:
        mat_path = hea_path[:-4] + ".mat"
        
        if not os.path.exists(mat_path):
            carpeta = os.path.dirname(hea_path)
            archivo = os.path.basename(hea_path)
            faltantes_por_carpeta[carpeta].append(archivo)
            total_faltantes += 1
        else:
            total_ok += 1

    # Reporte
    print(f"==== RESUMEN DEL ESCANEO ====")
    print(f"Total de registros (.hea) encontrados: {total_hea}")
    print(f"Registros OK con su .mat:               {total_ok}")
    print(f"Registros incompletos (falta .mat):      {total_faltantes}\n")

    if total_faltantes == 0:
        print("¡Todo perfecto! Todos los archivos .hea tienen su respectivo .mat.")
        return

    print("==== DETALLE DE CARPETAS AFECTADAS ====")
    for carpeta, archivos in faltantes_por_carpeta.items():
        print(f"\nCarpeta: {carpeta}")
        print(f"  Faltan {len(archivos)} archivos .mat")
        print(f"  Ejemplos faltantes: {archivos[:5]}" if len(archivos) > 5 else f"  Faltantes: {archivos}")

# ESTA LÍNEA ES LA QUE EJECUTA LA FUNCIÓN:
verificar_archivos_faltantes("dataset2020")