"use strict";

// ============================================================ i18n =========
const I18N = {
  es: {
    "app.title": "Clasificador de ECG",
    "nav.upload": "Cargar ECG",
    "nav.result": "Resultado",
    "nav.detail": "Detalle Técnico",
    "nav.experiments": "Experimentos",
    "model.ready": "Modelo listo",
    "model.pending": "Modelo pendiente",
    "upload.title": "Cargar electrocardiograma",
    "upload.intro": "Sube un ECG de 12 derivaciones en CSV o el par WFDB .hea + .mat. El análisis se ejecuta solo al pulsar Analizar ECG.",
    "upload.drop": "Arrastra el archivo aquí",
    "upload.dropSub": "o haz clic para seleccionarlo. Para WFDB selecciona ambos archivos del mismo registro.",
    "upload.formats": "Formatos:",
    "upload.help": "¿Qué formato usar?",
    "upload.helpCsv": "CSV: columnas de derivaciones estándar I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6. Puede iniciar con # Sampling Rate: 500 Hz.",
    "upload.helpWfdb": "WFDB: sube juntos registro.hea y registro.mat. La app leerá el header, preprocesará la señal y generará un CSV convertido descargable.",
    "upload.patient": "Paciente / ID (opcional)",
    "upload.patientPh": "Nombre o código",
    "upload.age": "Edad (opcional)",
    "upload.threshold": "Threshold multilabel",
    "upload.trueLabels": "Etiquetas reales opcionales para CSV",
    "upload.trueLabelsPh": "NSR, AF o 426783006,164889003",
    "upload.trueLabelsHint": "Si subes .hea + .mat, la comparación usa automáticamente el campo Dx del header. Para CSV puedes escribir clases o códigos SNOMED separados por coma.",
    "upload.autoModel": "La app selecciona automáticamente el mejor checkpoint disponible en la carpeta saved/ configurada.",
    "btn.analyze": "Analizar ECG",
    "btn.clear": "Limpiar",
    "btn.pdf": "Descargar informe (PDF)",
    "btn.csv": "Descargar CSV convertido",
    "btn.png": "Abrir trazado PNG",
    "result.title": "Resultado",
    "result.sub": "Predicción multilabel, trazado en papel milimetrado, probabilidades por clase y descarga de reporte.",
    "result.paper": "El trazado se muestra con cuadrícula tipo papel milimetrado ECG.",
    "empty.title": "Aún no hay resultados",
    "empty.body": "Sube un ECG y pulsa Analizar ECG.",
    "detail.title": "Detalle Técnico",
    "detail.sub": "Ficha técnica del modelo, preprocesamiento, umbrales, comparación con etiquetas reales y métricas exportadas.",
    "detail.emptyTitle": "Aún no hay detalle de predicción",
    "detail.emptyBody": "Después de analizar un ECG se completará la ficha del registro.",
    "detail.method": "Enfoque: la señal se ordena a 12 derivaciones estándar, se remuestrea a 500 Hz, se normaliza por derivación, se ajusta a 5000 muestras y se clasifica con salidas sigmoid independientes.",
    "detail.recordSheet": "Ficha del registro analizado",
    "detail.probabilities": "Probabilidades por clase",
    "detail.modelSheet": "Ficha técnica del modelo cargado automáticamente",
    "detail.schema": "Esquema de clases SNOMED-CT",
    "detail.metrics": "Métricas de evaluación",
    "detail.metricsFrom": "Resultados leídos desde",
    "detail.metricsPending": "Métricas pendientes.",
    "detail.metricsCmd": "Después de entrenar, ejecute evaluación para llenar esta sección:",
    "tech.arch": "Arquitectura",
    "tech.input": "Entrada",
    "tech.classes": "Clases",
    "tech.checkpoint": "Checkpoint",
    "tech.trained": "Entrenado",
    "tech.epoch": "Época / val_loss",
    "tech.params": "Parámetros",
    "tech.loss": "Pérdida",
    "tech.activation": "Activación",
    "table.class": "Clase",
    "table.description": "Descripción",
    "table.codes": "Códigos",
    "table.model": "Modelo",
    "metric.support": "Soporte +",
    "metric.sensitivity": "Sensibilidad",
    "metric.specificity": "Especificidad",
    "metric.precisionMacro": "Precisión macro",
    "metric.sensitivityMacro": "Sensibilidad macro",
    "exp.title": "Experimentos",
    "exp.sub": "Comparación arquitectónica y robustez con perturbaciones controladas.",
    "exp.method": "Objetivo experimental: comparar de forma justa la red residual frente a una CNN convencional equivalente usando el mismo dataset, split, preprocesamiento, entrenamiento y métricas; y medir robustez ante degradaciones controladas de la señal.",
    "exp.compare": "ResNet-34 vs CNN convencional",
    "exp.pending": "Resultado pendiente.",
    "exp.compareCmd": "Genere primero métricas para ambos modelos y luego compare:",
    "exp.robust": "Robustez frente a perturbaciones ECG",
    "exp.robustCmd": "Ejecute el experimento reproducible de robustez:",
    "exp.perturbation": "Perturbación",
    "exp.level": "Nivel",
    "footer.author": "Autor:",
    "footer.advisor": "Asesor:",
    "footer.model": "Modelo",
    "footer.disclaimer": "Herramienta de investigación — no reemplaza la lectura de un cardiólogo. Trabajo académico · UNSA.",
    "status.select": "Selecciona un CSV o el par .hea + .mat.",
    "status.ready": "listo para analizar",
    "status.loading": "Analizando la señal y ejecutando inferencia…",
    "status.done": "Análisis completado.",
    "status.error": "No se pudo analizar: ",
    "status.invalidCombo": "Sube un único CSV o exactamente el par .hea + .mat del mismo registro.",
    "status.stemMismatch": "El .hea y el .mat deben tener el mismo nombre base.",
    "notice.wfdb": "Entrada WFDB leída correctamente; se generó un CSV convertido.",
    "notice.csv": "Entrada CSV preprocesada a la forma esperada por el modelo.",
    "notice.truth": "Etiquetas reales disponibles para comparación.",
    "diag.top": "probabilidad top",
    "diag.noResult": "Sin resultado",
    "diag.noProb": "No se recibieron probabilidades.",
    "diag.noPositiveTitle": "Sin predicción positiva",
    "diag.topCandidate": "Mayor probabilidad observada",
    "diag.positives": "Predicciones positivas",
    "diag.nonePositive": "Ninguna clase superó el threshold definido",
    "diag.nonePositiveClassThresholds": "Ninguna clase superó su umbral calibrado",
    "diag.borderline": "Predicción cercana al umbral; interpretar como baja confianza.",
    "diag.fallbackApplied": "NSR añadido por regla de fallback normal porque ninguna clase superó su umbral.",
    "traffic.positive": "positiva",
    "traffic.negative": "bajo threshold",
    "summary.title": "Resumen multilabel:",
    "summary.none": "Sin positivos a threshold",
    "summary.noneClassThresholds": "Sin positivos con umbrales calibrados",
    "summary.model": "Modelo usado para este resultado",
    "summary.rule": "Regla de decisión",
    "summary.ruleClassThresholds": "probabilidad ≥ umbral calibrado por clase",
    "summary.ruleGlobalThreshold": "probabilidad ≥ threshold global",
    "gt.title": "Comparación con etiquetas reales",
    "gt.interpretation": "Interpretación",
    "gt.falseNegativeOnly": "El modelo no activó ninguna clase; como la etiqueta real sí existe, se considera falso negativo.",
    "gt.falsePositiveOnly": "El modelo activó clases que no están en la etiqueta real; se considera falso positivo.",
    "gt.mixedErrors": "Hay diferencias multilabel: algunas clases faltan y/o sobran frente al header.",
    "gt.correct": "Las clases predichas coinciden exactamente con las etiquetas reales disponibles.",
    "gt.notAvailable": "No hay etiquetas reales disponibles. En .hea + .mat se leen automáticamente desde Dx; para CSV puedes escribir clases o SNOMED antes de analizar.",
    "gt.sourceHeader": "Header WFDB Dx",
    "gt.sourceManual": "Ingreso manual",
    "gt.exact": "Coincidencia exacta multilabel",
    "gt.notExact": "Diferencias frente a las etiquetas reales",
    "gt.true": "Reales",
    "gt.pred": "Predichas",
    "gt.tp": "Aciertos",
    "gt.fp": "Falsos positivos",
    "gt.fn": "Falsos negativos",
    "gt.none": "Ninguna",
    "prob.index": "#",
    "prob.prob": "Probabilidad",
    "prob.threshold": "Umbral",
    "prob.margin": "Margen",
    "prob.state": "Estado",
    "prob.yes": "Positiva",
    "prob.no": "Negativa",
  },
  en: {
    "app.title": "ECG Classifier",
    "nav.upload": "Upload ECG",
    "nav.result": "Result",
    "nav.detail": "Technical Detail",
    "nav.experiments": "Experiments",
    "model.ready": "Model ready",
    "model.pending": "Model pending",
    "upload.title": "Upload electrocardiogram",
    "upload.intro": "Upload a 12-lead ECG as CSV or the WFDB .hea + .mat pair. Analysis starts only when you press Analyze ECG.",
    "upload.drop": "Drag the file here",
    "upload.dropSub": "or click to select it. For WFDB, select both files from the same record.",
    "upload.formats": "Formats:",
    "upload.help": "Which format should I use?",
    "upload.helpCsv": "CSV: standard lead columns I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6. It may start with # Sampling Rate: 500 Hz.",
    "upload.helpWfdb": "WFDB: upload record.hea and record.mat together. The app reads the header, preprocesses the signal and generates a downloadable converted CSV.",
    "upload.patient": "Patient / ID (optional)",
    "upload.patientPh": "Name or code",
    "upload.age": "Age (optional)",
    "upload.threshold": "Multilabel threshold",
    "upload.trueLabels": "Optional true labels for CSV",
    "upload.trueLabelsPh": "NSR, AF or 426783006,164889003",
    "upload.trueLabelsHint": "If you upload .hea + .mat, comparison uses the header Dx field automatically. For CSV, write class names or SNOMED codes separated by commas.",
    "upload.autoModel": "The app automatically selects the best available checkpoint in the configured saved/ folder.",
    "btn.analyze": "Analyze ECG",
    "btn.clear": "Clear",
    "btn.pdf": "Download report (PDF)",
    "btn.csv": "Download converted CSV",
    "btn.png": "Open PNG trace",
    "result.title": "Result",
    "result.sub": "Multilabel prediction, ECG-paper grid trace, per-class probabilities and report download.",
    "result.paper": "The trace is displayed with an ECG-paper grid.",
    "empty.title": "No results yet",
    "empty.body": "Upload an ECG and press Analyze ECG.",
    "detail.title": "Technical Detail",
    "detail.sub": "Model sheet, preprocessing, thresholds, true-label comparison and exported metrics.",
    "detail.emptyTitle": "No prediction detail yet",
    "detail.emptyBody": "After analyzing an ECG, the record sheet will be filled in.",
    "detail.method": "Method: the signal is ordered into 12 standard leads, resampled to 500 Hz, normalized per lead, fixed to 5000 samples and classified with independent sigmoid outputs.",
    "detail.recordSheet": "Analyzed record sheet",
    "detail.probabilities": "Per-class probabilities",
    "detail.modelSheet": "Automatically loaded model sheet",
    "detail.schema": "SNOMED-CT class schema",
    "detail.metrics": "Evaluation metrics",
    "detail.metricsFrom": "Results loaded from",
    "detail.metricsPending": "Metrics pending.",
    "detail.metricsCmd": "After training, run evaluation to fill this section:",
    "tech.arch": "Architecture",
    "tech.input": "Input",
    "tech.classes": "Classes",
    "tech.checkpoint": "Checkpoint",
    "tech.trained": "Trained",
    "tech.epoch": "Epoch / val_loss",
    "tech.params": "Parameters",
    "tech.loss": "Loss",
    "tech.activation": "Activation",
    "table.class": "Class",
    "table.description": "Description",
    "table.codes": "Codes",
    "table.model": "Model",
    "metric.support": "Support +",
    "metric.sensitivity": "Sensitivity",
    "metric.specificity": "Specificity",
    "metric.precisionMacro": "Macro precision",
    "metric.sensitivityMacro": "Macro sensitivity",
    "exp.title": "Experiments",
    "exp.sub": "Architecture comparison and robustness under controlled perturbations.",
    "exp.method": "Experimental goal: fairly compare the residual network against an equivalent conventional CNN using the same dataset, split, preprocessing, training setup and metrics; and measure robustness under controlled signal degradations.",
    "exp.compare": "ResNet-34 vs conventional CNN",
    "exp.pending": "Result pending.",
    "exp.compareCmd": "Generate metrics for both models first, then compare:",
    "exp.robust": "Robustness to ECG perturbations",
    "exp.robustCmd": "Run the reproducible robustness experiment:",
    "exp.perturbation": "Perturbation",
    "exp.level": "Level",
    "footer.author": "Author:",
    "footer.advisor": "Advisor:",
    "footer.model": "Model",
    "footer.disclaimer": "Research tool — it does not replace a cardiologist's reading. Academic work · UNSA.",
    "status.select": "Select a CSV or the .hea + .mat pair.",
    "status.ready": "ready to analyze",
    "status.loading": "Analyzing the signal and running inference…",
    "status.done": "Analysis completed.",
    "status.error": "Could not analyze: ",
    "status.invalidCombo": "Upload one CSV or exactly the .hea + .mat pair from the same record.",
    "status.stemMismatch": "The .hea and .mat files must have the same base name.",
    "notice.wfdb": "WFDB input read successfully; a converted CSV was generated.",
    "notice.csv": "CSV input preprocessed into the model's expected shape.",
    "notice.truth": "True labels are available for comparison.",
    "diag.top": "top probability",
    "diag.noResult": "No result",
    "diag.noProb": "No probabilities were received.",
    "diag.noPositiveTitle": "No positive prediction",
    "diag.topCandidate": "Highest observed probability",
    "diag.positives": "Positive predictions",
    "diag.nonePositive": "No class exceeded the selected threshold",
    "diag.nonePositiveClassThresholds": "No class exceeded its calibrated threshold",
    "diag.borderline": "Prediction is close to the threshold; interpret as low confidence.",
    "diag.fallbackApplied": "NSR was added by the normal-fallback rule because no class exceeded its threshold.",
    "traffic.positive": "positive",
    "traffic.negative": "below threshold",
    "summary.title": "Multilabel summary:",
    "summary.none": "No positives at threshold",
    "summary.noneClassThresholds": "No positives with calibrated thresholds",
    "summary.model": "Model used for this result",
    "summary.rule": "Decision rule",
    "summary.ruleClassThresholds": "probability ≥ class-calibrated threshold",
    "summary.ruleGlobalThreshold": "probability ≥ global threshold",
    "gt.title": "True-label comparison",
    "gt.interpretation": "Interpretation",
    "gt.falseNegativeOnly": "The model did not activate any class; because a true label exists, this is a false negative.",
    "gt.falsePositiveOnly": "The model activated classes that are not in the true label; this is a false positive.",
    "gt.mixedErrors": "There are multilabel differences: some classes are missing and/or extra relative to the header.",
    "gt.correct": "The predicted classes exactly match the available true labels.",
    "gt.notAvailable": "No true labels are available. With .hea + .mat they are read automatically from Dx; for CSV you can enter classes or SNOMED codes before analysis.",
    "gt.sourceHeader": "WFDB header Dx",
    "gt.sourceManual": "Manual input",
    "gt.exact": "Exact multilabel match",
    "gt.notExact": "Differences against true labels",
    "gt.true": "True",
    "gt.pred": "Predicted",
    "gt.tp": "True positives",
    "gt.fp": "False positives",
    "gt.fn": "False negatives",
    "gt.none": "None",
    "prob.index": "#",
    "prob.prob": "Probability",
    "prob.threshold": "Threshold",
    "prob.margin": "Margin",
    "prob.state": "State",
    "prob.yes": "Positive",
    "prob.no": "Negative",
  },
};

let LANG = "es";
try {
  const savedLang = localStorage.getItem("ecg-lang");
  if (savedLang === "en" || savedLang === "es") LANG = savedLang;
} catch (e) {}

function tr(key) {
  return (I18N[LANG] && I18N[LANG][key]) || (I18N.es && I18N.es[key]) || key;
}

function applyI18n() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = tr(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => { el.placeholder = tr(el.dataset.i18nPlaceholder); });
  const es = document.getElementById("langEs");
  const en = document.getElementById("langEn");
  if (es) es.classList.toggle("active", LANG === "es");
  if (en) en.classList.toggle("active", LANG === "en");
  refreshFileCard();
  if (lastResult) render(lastResult, false);
}

// ============================================================ state =======
const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone");
const fileInput = $("file");
const btnClassify = $("btnClassify");
const btnClear = $("btnClear");
const statusEl = $("status");
const fileCard = $("fileCard");
const visContent = $("visContent");
const visEmpty = $("visEmpty");
const detContent = $("detContent");
const detEmpty = $("detEmpty");
const skel = $("skel");
const diagnosisBox = $("diagnosisBox");
const trafficBox = $("trafficBox");
const summaryBox = $("summaryBox");
const gtBox = $("gtBox");
const plotDiv = $("plotDiv");
const technicalRows = $("technicalRows");
const probTable = $("probTable");
const btnPrint = $("btnPrint");
const btnCsv = $("btnCsv");
const btnPng = $("btnPng");
const threshold = $("threshold");
const patName = $("patName");
const patAge = $("patAge");
const trueLabels = $("trueLabels");

let currentFiles = [];
let lastResult = null;

const PALETTE = [
  "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#17becf",
  "#8c564b", "#e377c2", "#bcbd22", "#0d9488", "#64748b", "#7c3aed",
];

// ============================================================ utils =======
function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = String(s ?? "");
  return d.innerHTML;
}

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = "status " + (cls || "");
}

function setLoading(on) {
  btnClassify.disabled = on;
  btnClassify.classList.toggle("btn-loading", on);
  btnClassify.setAttribute("aria-busy", on ? "true" : "false");
  if (skel) skel.hidden = !on;
}

function fmtPct(value) {
  const n = Number(value || 0);
  return `${(n * 100).toFixed(2)}%`;
}

function shortPath(path) {
  const text = String(path || "").replace(/\\/g, "/");
  if (!text) return "—";
  const parts = text.split("/");
  return parts.slice(-3).join("/");
}

function listOrNone(values) {
  return values && values.length ? values.join(", ") : tr("gt.none");
}

function comparisonInterpretation(cmp) {
  if (!cmp || !cmp.available) return "";
  if (cmp.exact_match) return tr("gt.correct");
  const pred = cmp.predicted_classes || [];
  const fp = cmp.false_positive || [];
  const fn = cmp.false_negative || [];
  if (!pred.length && fn.length && !fp.length) return tr("gt.falseNegativeOnly");
  if (fp.length && !fn.length) return tr("gt.falsePositiveOnly");
  return tr("gt.mixedErrors");
}

function kv(label, value) {
  return `<div class="gt-kv"><span>${escapeHtml(label)}:</span> <b>${escapeHtml(value)}</b></div>`;
}

function fileExt(name) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function fileStem(name) {
  const clean = name.split(/[\\/]/).pop() || name;
  const i = clean.lastIndexOf(".");
  return (i >= 0 ? clean.slice(0, i) : clean).toLowerCase();
}

function validateFiles(files) {
  if (!files.length) return tr("status.select");
  const exts = files.map(f => fileExt(f.name));
  if (files.length === 1 && exts[0] === ".csv") return "";
  if (files.length === 2 && exts.includes(".hea") && exts.includes(".mat")) {
    const hea = files.find(f => fileExt(f.name) === ".hea");
    const mat = files.find(f => fileExt(f.name) === ".mat");
    if (fileStem(hea.name) !== fileStem(mat.name)) return tr("status.stemMismatch");
    return "";
  }
  return tr("status.invalidCombo");
}

// ============================================================ navigation ===
function goTo(sec) {
  document.querySelectorAll(".nav-link").forEach(b => b.classList.toggle("active", b.dataset.sec === sec));
  ["upload", "vis", "det", "exp"].forEach(s => {
    const el = $("sec-" + s);
    if (el) el.hidden = (s !== sec);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll(".nav-link").forEach(b => b.addEventListener("click", () => goTo(b.dataset.sec)));

// ============================================================ theme/lang ===
function updateThemeIcon() {
  const el = $("themeToggle");
  if (!el) return;
  const dark = document.documentElement.dataset.theme === "dark";
  el.innerHTML = dark
    ? '<svg class="ico"><use href="#i-sun"/></svg>'
    : '<svg class="ico"><use href="#i-moon"/></svg>';
}

try {
  const saved = localStorage.getItem("ecg-theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch (e) {}

$("themeToggle").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme;
  const nxt = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = nxt;
  try { localStorage.setItem("ecg-theme", nxt); } catch (e) {}
  updateThemeIcon();
});
$("langEs").addEventListener("click", () => { LANG = "es"; try { localStorage.setItem("ecg-lang", LANG); } catch (e) {} applyI18n(); });
$("langEn").addEventListener("click", () => { LANG = "en"; try { localStorage.setItem("ecg-lang", LANG); } catch (e) {} applyI18n(); });
updateThemeIcon();

// ============================================================ file upload ==
function refreshFileCard() {
  if (!currentFiles.length) {
    fileCard.hidden = true;
    return;
  }
  fileCard.hidden = false;
  const total = currentFiles.reduce((acc, f) => acc + (f.size || 0), 0);
  const size = total > 1048576 ? `${(total / 1048576).toFixed(2)} MB` : `${Math.max(1, Math.round(total / 1024))} KB`;
  const names = currentFiles.map(f => `<div class="fname">${escapeHtml(f.name)}</div>`).join("");
  const validation = validateFiles(currentFiles);
  const validationHtml = validation ? `<div class="hint error-text">${escapeHtml(validation)}</div>` : "";
  fileCard.innerHTML = `
    <svg class="ico"><use href="#i-file"/></svg>
    <div class="file-list">${names}<div class="hint muted">${currentFiles.length} archivo(s) · ${size} · ${tr("status.ready")}</div>${validationHtml}</div>`;
}

function handleFiles(files) {
  if (!files || !files.length) return;
  currentFiles = Array.from(files);
  refreshFileCard();
  setStatus("", "");
}

dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", e => { e.preventDefault(); dropzone.classList.remove("drag"); handleFiles(e.dataTransfer.files); });
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFiles(fileInput.files));

const fmtHelp = $("fmtHelp");
const fmtHelpBox = $("fmtHelpBox");
fmtHelp.addEventListener("click", () => { fmtHelpBox.hidden = !fmtHelpBox.hidden; });

btnClear.addEventListener("click", () => {
  currentFiles = [];
  lastResult = null;
  fileInput.value = "";
  fileCard.hidden = true;
  setStatus("", "");
  visContent.hidden = true;
  visEmpty.hidden = false;
  detContent.hidden = true;
  detEmpty.hidden = false;
  goTo("upload");
});

// ============================================================ submit ======
async function postFiles() {
  const validation = validateFiles(currentFiles);
  if (validation) {
    setStatus(validation, "err");
    refreshFileCard();
    return;
  }

  const fd = new FormData();
  currentFiles.forEach(file => fd.append("files", file, file.name));
  fd.append("patient_name", patName.value || "");
  fd.append("patient_age", patAge.value || "");
  fd.append("threshold", threshold.value || "0.5");
  fd.append("true_labels", trueLabels.value || "");

  setLoading(true);
  setStatus(tr("status.loading"), "loading");
  visEmpty.hidden = true;
  visContent.hidden = false;

  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok || data.ok === false || data.status === "error") {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    render(data);
    setStatus(tr("status.done"), "ok");
    goTo("vis");
  } catch (err) {
    console.error(err);
    setStatus(tr("status.error") + err.message, "err");
  } finally {
    setLoading(false);
  }
}

btnClassify.addEventListener("click", postFiles);

// ============================================================ render ======
function setModelNameFromResult(result) {
  const text = shortPath(result.model_path);
  const sidebar = $("modelName");
  const footer = $("modelNameFooter");
  if (sidebar) {
    sidebar.textContent = text;
    sidebar.title = result.model_path || "";
  }
  if (footer) footer.textContent = text;
}

function renderNotice(result) {
  const notice = $("notice");
  const messages = [];
  if (result.input_type === "wfdb_hea_mat") messages.push(tr("notice.wfdb"));
  if (result.input_type === "csv") messages.push(tr("notice.csv"));
  if (result.label_comparison && result.label_comparison.available) messages.push(tr("notice.truth"));
  notice.textContent = messages.join(" · ");
  notice.style.display = messages.length ? "block" : "none";
}

function renderDiagnosis(result) {
  const top = (result.top_predictions || [])[0];
  const positives = result.positive_predictions || [];
  const primary = positives[0] || null;
  const hasPositive = positives.length > 0;
  const posText = hasPositive
    ? positives.map(r => r.postprocessed
        ? `${r.class} (${fmtPct(r.probability)}; fallback NSR, threshold ${fmtPct(r.threshold ?? result.threshold)})`
        : `${r.class} (${fmtPct(r.probability)} ≥ ${fmtPct(r.threshold ?? result.threshold)})`
      ).join(", ")
    : (result.using_class_thresholds ? tr("diag.nonePositiveClassThresholds") : tr("diag.nonePositive"));
  const titleRow = primary || top;
  const cardColor = hasPositive && primary ? PALETTE[primary.index % PALETTE.length] : "#64748b";
  const title = hasPositive && primary ? primary.class : tr("diag.noPositiveTitle");
  let desc = tr("diag.noProb");
  if (hasPositive && primary) {
    const thresholdText = primary.postprocessed
      ? `fallback NSR · threshold original ${fmtPct(primary.threshold ?? result.threshold)}`
      : `threshold ${fmtPct(primary.threshold ?? result.threshold)}`;
    desc = `${escapeHtml(primary.display_name)} · ${tr("diag.top")} ${fmtPct(primary.probability)} · ${thresholdText}`;
  } else if (top) {
    desc = `${tr("diag.topCandidate")}: ${escapeHtml(top.class)} · ${fmtPct(top.probability)} < threshold ${fmtPct(top.threshold ?? result.threshold)}`;
  }
  const borderline = titleRow && (titleRow.near_threshold || Math.abs(Number(titleRow.margin || 0)) < 0.05);
  diagnosisBox.innerHTML = `
    <div class="diagnosis-card" style="border-left-color:${cardColor}">
      <div class="diagnosis-name" style="color:${cardColor}">${escapeHtml(title)}</div>
      <div class="diagnosis-desc">${desc}</div>
      <div class="diagnosis-conf">${tr("diag.positives")}: <strong>${escapeHtml(posText)}</strong></div>
      ${result.normal_fallback && result.normal_fallback.applied ? `<div class="fallback-note">${tr("diag.fallbackApplied")}</div>` : ""}
      ${borderline ? `<div class="borderline-warn">⚠ ${tr("diag.borderline")}</div>` : ""}
    </div>`;
}

function renderTraffic(result) {
  const rows = (result.predictions || []).slice(0, 8);
  trafficBox.innerHTML = rows.map(row => {
    const color = PALETTE[row.index % PALETTE.length];
    const positive = Number(row.prediction) === 1;
    const label = positive ? tr("traffic.positive") : tr("traffic.negative");
    return `<span class="traffic-item"><span class="traffic-dot" style="background:${color}"></span>${escapeHtml(row.class)} · ${label} (${fmtPct(row.probability)})</span>`;
  }).join("");
}

function renderSummary(result) {
  const positives = result.positive_predictions || [];
  summaryBox.innerHTML = `<div class="summary-label">${tr("summary.title")}</div>`;
  if (positives.length) {
    positives.forEach(row => {
      const color = PALETTE[row.index % PALETTE.length];
      const post = row.postprocessed ? " · fallback" : "";
      summaryBox.innerHTML += `<span class="chip" style="background:${color}" title="${escapeHtml(row.display_name)}">${escapeHtml(row.class)} · ${fmtPct(row.probability)}${post}</span>`;
    });
  } else {
    const noneText = result.using_class_thresholds
      ? tr("summary.noneClassThresholds")
      : `${tr("summary.none")} ${escapeHtml(result.threshold)}`;
    summaryBox.innerHTML += `<span class="chip" style="background:#64748b">${noneText}</span>`;
  }
  const rule = result.using_class_thresholds
    ? `${tr("summary.ruleClassThresholds")} (${escapeHtml(shortPath(result.threshold_source))})`
    : `${tr("summary.ruleGlobalThreshold")} ${escapeHtml(result.threshold)}`;
  summaryBox.innerHTML += `<p class="hint muted summary-hint">${tr("summary.model")}: <strong>${escapeHtml(result.model_type || "—")}</strong><br>${tr("summary.rule")}: ${rule}</p>`;
}

function renderComparison(result) {
  const cmp = result.label_comparison || null;
  if (!cmp) {
    gtBox.hidden = true;
    return;
  }
  gtBox.hidden = false;
  if (!cmp.available) {
    gtBox.className = "gt gt-warn";
    gtBox.innerHTML = `<strong>${tr("gt.title")}</strong><br>${tr("gt.notAvailable")}`;
    return;
  }
  const ok = Boolean(cmp.exact_match);
  gtBox.className = `gt ${ok ? "gt-ok" : "gt-bad"}`;
  const source = cmp.source === "wfdb_header_dx" ? tr("gt.sourceHeader") : tr("gt.sourceManual");
  const interp = comparisonInterpretation(cmp);
  gtBox.innerHTML = `
    <strong>${tr("gt.title")}: ${ok ? tr("gt.exact") : tr("gt.notExact")}</strong>
    <p class="gt-interpretation"><b>${tr("gt.interpretation")}:</b> ${escapeHtml(interp)}</p>
    <div class="gt-grid">
      ${kv(tr("gt.true"), listOrNone(cmp.true_classes))}
      ${kv(tr("gt.pred"), listOrNone(cmp.predicted_classes))}
      ${kv(tr("gt.tp"), listOrNone(cmp.true_positive))}
      ${kv(tr("gt.fp"), listOrNone(cmp.false_positive))}
      ${kv(tr("gt.fn"), listOrNone(cmp.false_negative))}
      ${kv("F1 / Jaccard", `${Number(cmp.f1).toFixed(3)} / ${Number(cmp.jaccard).toFixed(3)}`)}
    </div>
    <div class="hint muted">${source}${cmp.dx_codes && cmp.dx_codes.length ? ` · DX: ${escapeHtml(cmp.dx_codes.join(", "))}` : ""}</div>`;
}

function renderPlot(result) {
  if (result.plot_url) {
    plotDiv.innerHTML = `<a href="${escapeHtml(result.plot_url)}" target="_blank" rel="noopener"><img class="plot" src="${escapeHtml(result.plot_url)}" alt="ECG paper grid 12-lead trace"></a>`;
    btnPng.href = result.plot_url;
    btnPng.hidden = false;
  } else {
    plotDiv.innerHTML = `<p class="hint muted">No se generó imagen del trazado.</p>`;
    btnPng.hidden = true;
  }
}

function renderButtons(result) {
  if (result.pdf_url) {
    btnPrint.disabled = false;
    btnPrint.onclick = () => {
      const a = document.createElement("a");
      a.href = result.pdf_url;
      a.download = `${result.record_name || "ecg"}_reporte.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    };
  } else {
    btnPrint.disabled = true;
    btnPrint.onclick = null;
  }
  if (result.converted_csv_url) {
    btnCsv.href = result.converted_csv_url;
    btnCsv.download = `${result.record_name || "ecg"}_convertido.csv`;
    btnCsv.hidden = false;
  } else {
    btnCsv.hidden = true;
  }
}

function row(label, value) {
  return `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value ?? "—")}</td></tr>`;
}

function renderTechnical(result) {
  const details = result.technical_details || {};
  const cmp = result.label_comparison || {};
  technicalRows.innerHTML = [
    row(LANG === "es" ? "Registro" : "Record", result.record_name),
    row(LANG === "es" ? "Paciente / ID" : "Patient / ID", result.patient_name || "—"),
    row(LANG === "es" ? "Edad" : "Age", result.patient_age || "—"),
    row(LANG === "es" ? "Tipo de entrada" : "Input type", result.input_type),
    row(LANG === "es" ? "Archivos fuente" : "Source files", (result.source_files || []).map(shortPath).join(" | ")),
    row(LANG === "es" ? "Modelo" : "Model", result.model_type),
    row("Checkpoint", result.model_path),
    row(LANG === "es" ? "Época checkpoint" : "Checkpoint epoch", result.checkpoint_epoch),
    row("Val loss", result.checkpoint_val_loss),
    row("Threshold", result.using_class_thresholds ? (LANG === "es" ? "calibrado por clase" : "class-calibrated") : result.threshold),
    row(LANG === "es" ? "Fuente de umbrales" : "Threshold source", result.threshold_source || "—"),
    row(LANG === "es" ? "Shape procesado" : "Processed shape", Array.isArray(result.processed_shape) ? result.processed_shape.join(" × ") : result.processed_shape),
    row(LANG === "es" ? "Frecuencia original" : "Original sampling rate", result.original_sampling_rate ? `${result.original_sampling_rate} Hz` : "—"),
    row(LANG === "es" ? "Frecuencia objetivo" : "Target sampling rate", result.target_sampling_rate ? `${result.target_sampling_rate} Hz` : "—"),
    row(LANG === "es" ? "Ventana" : "Window", `${result.window_seconds} s · ${result.input_length} samples`),
    row(LANG === "es" ? "Derivaciones" : "Leads", Array.isArray(result.lead_names) ? result.lead_names.join(", ") : result.lead_names),
    row("DX", result.dx_codes && result.dx_codes.length ? result.dx_codes.join(", ") : "—"),
    row(LANG === "es" ? "Clases reales" : "True classes", cmp.available ? listOrNone(cmp.true_classes) : "—"),
    row(LANG === "es" ? "Clases predichas" : "Predicted classes", cmp.predicted_classes ? listOrNone(cmp.predicted_classes) : "—"),
    row(LANG === "es" ? "Coincidencia exacta" : "Exact match", cmp.available ? (cmp.exact_match ? "Sí" : "No") : "—"),
    row("Fallback NSR", result.normal_fallback && result.normal_fallback.applied ? `${LANG === "es" ? "aplicado" : "applied"} · min=${result.normal_fallback.min_nsr_probability}` : (LANG === "es" ? "no aplicado" : "not applied")),
    row(LANG === "es" ? "Preprocesamiento" : "Preprocessing", details.preprocessing),
    row(LANG === "es" ? "Estilo de trazado" : "Trace style", details.plot_style),
    row(LANG === "es" ? "Esquema" : "Schema", details.label_schema),
    row(LANG === "es" ? "Activación" : "Activation", details.activation),
    row(LANG === "es" ? "Pérdida" : "Loss", details.loss),
    row(LANG === "es" ? "Regla de decisión" : "Decision rule", details.decision_rule),
    row(LANG === "es" ? "Dispositivo" : "Device", details.device),
  ].join("");

  const rows = result.predictions || [];
  probTable.innerHTML = `
    <thead><tr><th>${tr("prob.index")}</th><th>${tr("table.class")}</th><th>${tr("table.description")}</th><th>SNOMED</th><th>${tr("prob.prob")}</th><th>${tr("prob.threshold")}</th><th>${tr("prob.margin")}</th><th>${tr("prob.state")}</th></tr></thead>
    <tbody>
      ${rows.map(r => {
        const color = PALETTE[r.index % PALETTE.length];
        const width = Math.max(0, Math.min(100, Number(r.probability || 0) * 100));
        const pos = Number(r.prediction) === 1;
        return `<tr>
          <td class="iv-idx">${escapeHtml(r.index)}</td>
          <td class="prob-name">${escapeHtml(r.class)}</td>
          <td>${escapeHtml(r.display_name || r.description || "")}</td>
          <td><code>${escapeHtml(Array.isArray(r.snomed_codes) ? r.snomed_codes.join(", ") : r.snomed_codes)}</code></td>
          <td><span class="prob-bar"><span class="prob-fill" style="width:${width}%;background:${color}"></span></span>${fmtPct(r.probability)}</td>
          <td>${fmtPct(r.threshold ?? result.threshold)}</td>
          <td>${Number(r.margin ?? 0).toFixed(3)}${r.near_threshold ? ' · ~' : ''}</td>
          <td><span class="badge ${pos ? "ok" : "no"}">${pos ? tr("prob.yes") : tr("prob.no")}</span>${r.postprocessed ? '<span class="badge post">fallback</span>' : ''}</td>
        </tr>`;
      }).join("")}
    </tbody>`;
}

function render(result, updateModel = true) {
  lastResult = result;
  if (updateModel) setModelNameFromResult(result);
  visEmpty.hidden = true;
  visContent.hidden = false;
  detEmpty.hidden = true;
  detContent.hidden = false;
  if (skel) skel.hidden = true;

  renderNotice(result);
  renderDiagnosis(result);
  renderTraffic(result);
  renderSummary(result);
  renderComparison(result);
  renderPlot(result);
  renderButtons(result);
  renderTechnical(result);
}

applyI18n();
