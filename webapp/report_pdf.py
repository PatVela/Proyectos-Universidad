# -*- coding: utf-8 -*-
"""Generate a formal, well-formatted clinical ECG report as a real PDF.

The client sends the analysis + the rendered ECG PNG (already produced by the
server for the interactive view) as JSON; this module builds a clean A4 PDF
with ReportLab and returns the bytes.
"""

from __future__ import absolute_import

import base64
import io

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

try:
    # Helvetica covers Latin with diacritics; register if a serif looks better.
    pdfmetrics.registerFont(TTFont('DejaVuSerif', 'DejaVuSerif.ttf'))
    BODY_FONT = 'DejaVuSerif'
except Exception:
    BODY_FONT = 'Helvetica'

OUTER = colors.HexColor('#123b6b')
MUTED = colors.HexColor('#6b7a8d')
INK = colors.HexColor('#14202f')
LINE = colors.HexColor('#dde3ea')


def _styles():
    s = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=s['Title'], fontName=BODY_FONT,
                           fontSize=15, textColor=OUTER, spaceAfter=2,
                           alignment=TA_LEFT)
    sub = ParagraphStyle('sub', parent=s['Normal'], fontName=BODY_FONT,
                         fontSize=8.5, textColor=MUTED, spaceAfter=6)
    h2 = ParagraphStyle('h2', parent=s['Heading2'], fontName=BODY_FONT,
                        fontSize=11, textColor=OUTER, spaceBefore=6, spaceAfter=4)
    label = ParagraphStyle('label', parent=s['Normal'], fontName=BODY_FONT,
                           fontSize=9, textColor=MUTED)
    value = ParagraphStyle('value', parent=s['Normal'], fontName=BODY_FONT,
                           fontSize=9.5, textColor=INK)
    foot = ParagraphStyle('foot', parent=s['Normal'], fontName=BODY_FONT,
                          fontSize=7.5, textColor=MUTED, alignment=TA_CENTER)
    return title, sub, h2, label, value, foot


def _png_image(b64):
    """Return a ReportLab Image scaled to fit the A4 width from base64 PNG."""
    raw = base64.b64decode(b64)
    from reportlab.lib.utils import ImageReader
    reader = ImageReader(io.BytesIO(raw))
    iw, ih = reader.getSize()
    max_w = 178 * mm
    max_h = 120 * mm
    scale = min(max_w / iw, max_h / ih, 1.0)
    # Pass the raw PNG bytes so ReportLab builds the flowable correctly.
    img = Image(__import__('io').BytesIO(raw), width=iw * scale, height=ih * scale)
    img.hAlign = 'CENTER'
    return img


def build_report(data, project, reference):
    """Build a formal A4 PDF from an analysis dict. Returns PDF bytes."""
    title, sub, h2, label, value, foot = _styles()

    plot_b64 = (data.get('plot') or '').split(',')[-1]  # strip data URI prefix
    name = data.get('patient') or '—'
    age = data.get('age') or '—'
    date = data.get('date') or ''
    dominant = data.get('dominant_name') or data.get('dominant') or '—'
    pct = data.get('dominant_pct')
    conf = data.get('confidence')
    fs = data.get('fs')
    dur = data.get('duration')
    lead = data.get('lead') or ''
    ground_truth = data.get('ground_truth') or None
    classes = data.get('classes') or []

    inst = project.get('institucion', '')
    esc = project.get('escuela', '')
    curso = project.get('curso', '')
    anio = project.get('anio', '')
    autor = project.get('autor', '')
    asesor = project.get('asesor', '')
    ref = reference.get('cita', '')

    story = []
    story.append(Paragraph('Informe de Electrocardiograma (ECG)', title))
    story.append(Paragraph('{} · {}'.format(inst, esc), sub))
    story.append(Paragraph('{} {} · V{}'.format(curso, anio, project.get('version', '')), sub))
    story.append(Spacer(1, 4))
    story.append(Paragraph('Identificación de la señal', h2))

    # patient table
    rows = [
        [Paragraph('Paciente', label), Paragraph(str(name), value)],
        [Paragraph('Edad', label), Paragraph(str(age), value)],
        [Paragraph('Fecha', label), Paragraph(str(date), value)],
        [Paragraph('Ritmo predominante', label),
         Paragraph('%s%s' % (dominant, (' (%s%%)' % pct) if pct is not None else ''), value)],
        [Paragraph('Confianza', label),
         Paragraph(('%.1f%%' % conf) if conf is not None else '—', value)],
    ]
    if ground_truth:
        gt_name = ground_truth.get('name') or ground_truth.get('label') or '—'
        gt_code = ground_truth.get('label') or '—'
        gt_ok = ground_truth.get('correct')
        gt_src = ground_truth.get('source') or ''
        gt_rec = ground_truth.get('record') or ''
        gt_text = '%s (%s) · %s%s%s' % (
            gt_name, gt_code,
            'predicción correcta' if gt_ok else 'predicción distinta',
            (' · registro ' + str(gt_rec)) if gt_rec else '',
            (' · ' + str(gt_src)) if gt_src else '')
        rows.append([Paragraph('Etiqueta real', label), Paragraph(gt_text, value)])
    rows.extend([
        [Paragraph('Parámetros de registro', label),
         Paragraph('%s Hz · 1 derivación%s · duración %s s'
                   % (fs, (' (' + str(lead) + ')') if lead else '',
                      ('%.1f' % dur) if dur is not None else ''), value)],
    ])
    t = Table(rows, colWidths=[45 * mm, 130 * mm], hAlign='LEFT')
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), BODY_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), MUTED),
        ('TEXTCOLOR', (1, 0), (1, -1), INK),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, -2), 0.4, LINE),
        ('BOX', (0, 0), (-1, -1), 0.4, LINE),
    ]))
    story.append(t)

    # distribution table
    if classes:
        story.append(Paragraph('Distribución de clases', h2))
        hdrs = [Paragraph('<b>Clase</b>', label), Paragraph('<b>%</b>', label),
                Paragraph('<b>Intervalos</b>', label)]
        body = [hdrs]
        for c in classes:
            body.append([Paragraph(str(c.get('name', '')), value),
                         Paragraph(str(c.get('pct', '')), value),
                         Paragraph(str(c.get('count', '')), value)])
        dt = Table(body, colWidths=[120 * mm, 30 * mm, 25 * mm], hAlign='LEFT')
        dt.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), BODY_FONT),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eef3f8')),
            ('TEXTCOLOR', (0, 0), (-1, 0), OUTER),
            ('GRID', (0, 0), (-1, -1), 0.4, LINE),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(dt)

    # ECG image
    story.append(Paragraph('Trazado del ECG', h2))
    if plot_b64:
        try:
            story.append(_png_image(plot_b64))
        except Exception:
            story.append(Paragraph('(Sin imagen del trazado)', value))
    story.append(Spacer(1, 6))

    # signatures
    sign = Table([[Paragraph('Investigador/a · %s' % autor if autor else 'Investigador/a', foot),
                   Paragraph('Asesor/a / validador · %s' % asesor if asesor else 'Asesor/a / validador', foot)]],
                 colWidths=[89 * mm, 89 * mm], hAlign='CENTER')
    sign.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, -1), 0.5, INK),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(Spacer(1, 8))
    story.append(sign)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        'Herramienta desarrollada con fines académicos de investigación. '
        'No reemplaza la lectura de un cardiólogo ni debe usarse para '
        'diagnóstico clínico.', foot))
    story.append(Paragraph('Referencia: %s' % ref, foot))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title='Informe ECG', author=autor or 'ECG App')
    doc.build(story)
    return buf.getvalue()
