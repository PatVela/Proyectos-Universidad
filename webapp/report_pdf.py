"""Generación de PDF para resultados de la webapp CINC2020-12."""

from __future__ import annotations

from pathlib import Path


TITLE = "Reporte ECG CINC2020-12"
DISCLAIMER = "Uso académico. Las predicciones no constituyen diagnóstico médico."


def build_pdf_report(prediction: dict, output_path: str | Path) -> str:
    """Genera PDF con resultados, detalles técnicos y trazado ECG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return _build_reportlab_pdf(prediction, output_path)
    except Exception:
        return _build_minimal_pdf(prediction, output_path)


def _build_reportlab_pdf(prediction: dict, output_path: Path) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=16, textColor=colors.HexColor("#1d4ed8"), spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#1e3a8a"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#475569"))

    story = [Paragraph(TITLE, title), Paragraph(DISCLAIMER, small), Spacer(1, 4)]

    meta_rows = [
        ["Registro", prediction.get("record_name", "—")],
        ["Paciente / ID", prediction.get("patient_name") or "—"],
        ["Edad", prediction.get("patient_age") or "—"],
        ["Tipo de entrada", prediction.get("input_type", "—")],
        ["Modelo", prediction.get("model_type", "—")],
        ["Checkpoint", prediction.get("model_path", "—")],
        ["Época checkpoint", str(prediction.get("checkpoint_epoch", "—"))],
        ["Val loss checkpoint", str(prediction.get("checkpoint_val_loss", "—"))],
        ["Frecuencia original", f"{prediction.get('original_sampling_rate', '—')} Hz"],
        ["Frecuencia objetivo", f"{prediction.get('target_sampling_rate', '—')} Hz"],
        ["Forma procesada", str(prediction.get("processed_shape", "—"))],
        ["Threshold", "calibrado por clase" if prediction.get("using_class_thresholds") else str(prediction.get("threshold", "—"))],
        ["Fuente de umbrales", prediction.get("threshold_source", "fallback_global_0.5")],
        ["Fallback NSR", "aplicado" if (prediction.get("normal_fallback") or {}).get("applied") else "no aplicado"],
    ]
    story.append(Paragraph("Resumen técnico", h2))
    story.append(_table(meta_rows, col_widths=[45 * mm, 130 * mm]))

    positives = prediction.get("positive_predictions") or []
    story.append(Paragraph("Diagnósticos positivos por threshold", h2))
    if positives:
        rows = [["Clase", "Nombre", "Probabilidad", "Nota"]]
        for item in positives:
            rows.append([
                item.get("class", ""),
                item.get("display_name", ""),
                f"{item.get('probability', 0) * 100:.2f}%",
                "fallback NSR" if item.get("postprocessed") else "",
            ])
        story.append(_table(rows, header=True, col_widths=[25 * mm, 85 * mm, 30 * mm, 30 * mm]))
    else:
        if prediction.get("using_class_thresholds"):
            story.append(Paragraph("Ninguna clase superó su umbral calibrado.", body))
        else:
            story.append(Paragraph("Ninguna clase superó el threshold definido.", body))

    comparison = prediction.get("label_comparison") or {}
    if comparison.get("available"):
        story.append(Paragraph("Comparación con etiquetas reales", h2))
        source = "Header WFDB Dx" if comparison.get("source") == "wfdb_header_dx" else "Ingreso manual"
        predicted = comparison.get("predicted_classes") or []
        false_positive = comparison.get("false_positive") or []
        false_negative = comparison.get("false_negative") or []
        if comparison.get("exact_match"):
            interpretation = "Las clases predichas coinciden exactamente con las etiquetas reales disponibles."
        elif not predicted and false_negative and not false_positive:
            interpretation = "El modelo no activó ninguna clase; como existe etiqueta real, se considera falso negativo."
        elif false_positive and not false_negative:
            interpretation = "El modelo activó clases que no están en la etiqueta real; se considera falso positivo."
        else:
            interpretation = "Hay diferencias multilabel frente a la etiqueta real."
        comp_rows = [
            ["Fuente", source],
            ["Interpretación", interpretation],
            ["Reales", ", ".join(comparison.get("true_classes") or []) or "Ninguna"],
            ["Predichas", ", ".join(predicted) or "Ninguna"],
            ["Aciertos", ", ".join(comparison.get("true_positive") or []) or "Ninguna"],
            ["Falsos positivos", ", ".join(false_positive) or "Ninguna"],
            ["Falsos negativos", ", ".join(false_negative) or "Ninguna"],
            ["Coincidencia exacta", "Sí" if comparison.get("exact_match") else "No"],
            ["F1 / Jaccard", f"{comparison.get('f1', 0):.3f} / {comparison.get('jaccard', 0):.3f}"],
        ]
        story.append(_table(comp_rows, col_widths=[45 * mm, 130 * mm], font_size=8))
    else:
        story.append(Paragraph("Comparación con etiquetas reales", h2))
        story.append(Paragraph("No se proporcionaron etiquetas reales para este registro.", body))

    story.append(Paragraph("Probabilidades por clase", h2))
    rows = [["Clase", "Nombre", "SNOMED", "%", "Umbral", "Pred"]]
    for item in prediction.get("predictions", []):
        rows.append([
            item.get("class", ""),
            item.get("display_name", ""),
            ", ".join(item.get("snomed_codes", [])),
            f"{item.get('probability', 0) * 100:.2f}",
            f"{item.get('threshold', prediction.get('threshold', 0.5)) * 100:.2f}",
            "1" if item.get("prediction") else "0",
        ])
    story.append(_table(rows, header=True, col_widths=[20 * mm, 58 * mm, 53 * mm, 15 * mm, 18 * mm, 10 * mm], font_size=7))

    details = prediction.get("technical_details") or {}
    if details:
        story.append(Paragraph("Detalles metodológicos", h2))
        story.append(_table([[k, str(v)] for k, v in details.items()], col_widths=[45 * mm, 130 * mm], font_size=8))

    plot_path = prediction.get("plot_path")
    if plot_path and Path(plot_path).exists():
        story.append(Paragraph("Trazado ECG preprocesado", h2))
        img = Image(str(plot_path))
        max_w = 175 * mm
        max_h = 105 * mm
        scale = min(max_w / img.imageWidth, max_h / img.imageHeight, 1.0)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        story.append(img)

    story.append(Spacer(1, 8))
    story.append(Paragraph(DISCLAIMER, small))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=TITLE,
    )
    doc.build(story)
    return str(output_path)


def _table(rows, header: bool = False, col_widths=None, font_size: int = 8):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    table.setStyle(TableStyle(style))
    return table


def _escape_pdf_text(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_minimal_pdf(prediction: dict, output_path: Path) -> str:
    """Fallback sin reportlab: PDF mínimo de una página con texto."""
    comparison = prediction.get("label_comparison") or {}
    lines = [
        TITLE,
        DISCLAIMER,
        f"Registro: {prediction.get('record_name', '—')}",
        f"Paciente: {prediction.get('patient_name') or '—'}",
        f"Modelo: {prediction.get('model_type', '—')}",
        f"Checkpoint: {prediction.get('model_path', '—')}",
        "",
        "Comparación real vs predicho:",
        f"Disponible: {'sí' if comparison.get('available') else 'no'}",
        f"Reales: {', '.join(comparison.get('true_classes') or []) or '—'}",
        f"Predichas: {', '.join(comparison.get('predicted_classes') or []) or '—'}",
        f"Exact match: {'sí' if comparison.get('exact_match') else 'no'}",
        "",
        "Probabilidades:",
    ]
    for item in prediction.get("predictions", [])[:20]:
        lines.append(f"{item.get('class')}: {item.get('probability', 0) * 100:.2f}% pred={item.get('prediction')}")

    content = "BT /F1 9 Tf 40 790 Td "
    for i, line in enumerate(lines[:55]):
        if i > 0:
            content += "0 -13 Td "
        content += f"({_escape_pdf_text(line[:115])}) Tj "
    content += "ET"
    content_bytes = content.encode("latin-1", errors="replace")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(content_bytes)).encode() + b" >> stream\n" + content_bytes + b"\nendstream endobj\n",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj
    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    output_path.write_bytes(pdf)
    return str(output_path)
