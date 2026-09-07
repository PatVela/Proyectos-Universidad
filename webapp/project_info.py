# -*- coding: utf-8 -*-
"""Datos institucionales y de autoría del proyecto (EDITA AQUÍ).

Estos valores se muestran en el encabezado institucional, en la ficha de
autoría del pie de página y en el informe. Al ser un único módulo, se cambian
una sola vez sin tocar el HTML.
"""

PROJECT_INFO = {
    # ---------------------- Institución ----------------------
    "institucion": "Universidad Nacional de San Agustín de Arequipa",
    "facultad": "Facultad de Ingeniería de Producción y Servicios",
    "escuela": "Escuela Profesional de Ingeniería Electrónica",
    "titulo": "Clasificador automático de electrocardiogramas de una sola "
              "derivación mediante redes neuronales convolucionales profundas "
              "(réplica de Hannun et al., Nature Medicine 2019)",

    # ---------------------- Curso ----------------------
    "curso": "Trabajo de Investigación",
    "curso_codigo": "",               # ej. "TI-XXX" (o déjalo vacío)
    "anio": "2026",

    # ---------------------- Autoría ----------------------
    # Deja los campos vacíos si aún no los tienes; la app los omite.
    "autor": "J.P.D.V & J.J.H.N",                      # ej. "Juerg P. Velásquez"
    "autor_email": "jdiazve@unsa.edu.pe & jhuillcan@unsa.edu.pe",                # ej. "jvelasquez@unsa.edu.pe"
    "asesor": "Tu Viejo Echaiz",                     # ej. "Ing. María Gómez, M.Sc."
    "asesor_cargo": "",               # ej. "Docente del curso · UNSA"

    # ---------------------- Sobre el sistema ----------------------
    "version": "2.1",                 # versión mostrada de la aplicación
    "descripcion_corta": ("Asistente automático de lectura de ECG de una sola "
                          "derivación basado en deep learning, desarrollado "
                          "con fines de investigación académica."),
}

# Referencia bibliográfica completa (se muestra en "Referencias").
REFERENCE = {
    "cita": "Hannun AY, Rajpurkar P, Haghpanahi M, et al. Cardiologist-level "
            "arrhythmia detection and classification in ambulatory "
            "electrocardiograms using a deep neural network. Nature Medicine. "
            "2019;25(1):65-69.",
    "doi": "https://doi.org/10.1038/s41591-018-0268-3",
    "dataset": ("PhysioNet Computing in Cardiology Challenge 2017 "
                "(CinC2017): AF Classification from a Short Single Lead ECG "
                "Recording. https://physionet.org/content/challenge-2017/"),
}
