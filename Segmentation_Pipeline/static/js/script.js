/* ── Config ──────────────────────────────────────────────────────────────────── */
const SERVER = "http://localhost:5001";

/* ── State ───────────────────────────────────────────────────────────────────── */
let imageFile = null;
let modelFile = null;
let currentView = "result";
let resultDataURL = null;
let originalDataURL = null;

/* ── DOM refs ────────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const imageDropZone   = $("imageDropZone");
const modelDropZone   = $("modelDropZone");
const imageFileInput  = $("imageFileInput");
const modelFileInput  = $("modelFileInput");
const imageBrowseBtn  = $("imageBrowseBtn");
const modelBrowseBtn  = $("modelBrowseBtn");
const imageBadge      = $("imageBadge");
const modelBadge      = $("modelBadge");
const confSlider      = $("confSlider");
const confVal         = $("confVal");
const promptInput     = $("promptInput");
const runBtn          = $("runBtn");
const progressWrap    = $("progressWrap");
const progressBar     = $("progressBar");
const progressLabel   = $("progressLabel");
const resultsSection  = $("resultsSection");
const resultImg       = $("resultImg");
const splitOrigImg    = $("splitOrigImg");
const splitResImg     = $("splitResImg");
const singleView      = $("singleView");
const splitView       = $("splitView");
const detectionsGrid  = $("detectionsGrid");
const countBadge      = $("countBadge");
const samPromptTag    = $("samPromptTag");
const errorBanner     = $("errorBanner");
const errorText       = $("errorText");
const statusDot       = $("statusDot");
const statusText      = $("statusText");
const exportBtn       = $("exportBtn");
const serverUrlEl     = $("serverUrl");

serverUrlEl.textContent = SERVER;

/* ── Colour palette for detection cards ─────────────────────────────────────── */
const PALETTE = [
  "#3b9eff", "#22c97b", "#f5a623", "#e8514a",
  "#c97bf5", "#5ef0e8", "#f56b9e", "#b8f567",
];

/* ── Health check ────────────────────────────────────────────────────────────── */
async function checkServer() {
  try {
    const r = await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      const d = await r.json();
      statusDot.className = "status-dot online";
      statusText.textContent = `Server online · ${d.device.toUpperCase()}`;
    } else {
      throw new Error();
    }
  } catch {
    statusDot.className = "status-dot offline";
    statusText.textContent = "Server offline";
  }
}

checkServer();
setInterval(checkServer, 10_000);

/* ── Slider ──────────────────────────────────────────────────────────────────── */
confSlider.addEventListener("input", () => {
  confVal.textContent = confSlider.value + "%";
});

/* ── Drag-and-drop helpers ───────────────────────────────────────────────────── */
function setupDropZone(zone, input, accept, onFile) {
  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("dragging");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("dragging");
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
  input.addEventListener("change", () => {
    if (input.files[0]) onFile(input.files[0]);
  });
}

function setImageFile(file) {
  imageFile = file;
  imageBadge.textContent = file.name;
  imageBadge.hidden = false;
  imageDropZone.classList.add("has-file");
  updateRunBtn();
}

function setModelFile(file) {
  if (!file.name.endsWith(".pt")) {
    showError("Model file must be a .pt checkpoint.");
    return;
  }
  modelFile = file;
  modelBadge.textContent = file.name;
  modelBadge.hidden = false;
  modelDropZone.classList.add("has-file");
  updateRunBtn();
}

setupDropZone(imageDropZone, imageFileInput, "image/*", setImageFile);
setupDropZone(modelDropZone, modelFileInput, ".pt",      setModelFile);

imageBrowseBtn.addEventListener("click", () => imageFileInput.click());
modelBrowseBtn.addEventListener("click", () => modelFileInput.click());

function updateRunBtn() {
  runBtn.disabled = !(imageFile && modelFile);
}

/* ── Progress helpers ────────────────────────────────────────────────────────── */
const STAGES = [
  [10, "Uploading files…"],
  [30, "Running YOLO detection…"],
  [60, "SAM3 segmenting defects…"],
  [85, "Rendering results…"],
  [100, "Done."],
];

let stageIndex = 0;
let stageTimer = null;

function startProgress() {
  progressWrap.hidden = false;
  resultsSection.hidden = true;
  errorBanner.hidden = true;
  stageIndex = 0;
  advanceStage();
}

function advanceStage() {
  if (stageIndex >= STAGES.length) return;
  const [pct, label] = STAGES[stageIndex];
  progressBar.style.width = pct + "%";
  progressLabel.textContent = label;
  stageIndex++;
  stageTimer = setTimeout(advanceStage, stageIndex < 4 ? 1800 : 99999);
}

function finishProgress() {
  clearTimeout(stageTimer);
  progressBar.style.width = "100%";
  progressLabel.textContent = "Done.";
  setTimeout(() => { progressWrap.hidden = true; }, 800);
}

function resetProgress() {
  clearTimeout(stageTimer);
  progressBar.style.width = "0%";
  progressWrap.hidden = true;
}

/* ── Run analysis ────────────────────────────────────────────────────────────── */
runBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  hideError();
  startProgress();
  runBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("model", modelFile);
  formData.append("conf",  confSlider.value / 100);
  const prompt = promptInput.value.trim();
  if (prompt) formData.append("prompt", prompt);

  try {
    const r = await fetch(`${SERVER}/analyze`, {
      method: "POST",
      body: formData,
    });

    const data = await r.json();

    if (!r.ok || data.error) {
      throw new Error(data.error || `Server returned ${r.status}`);
    }

    finishProgress();
    renderResults(data);

  } catch (err) {
    resetProgress();
    showError(err.message);
  } finally {
    runBtn.disabled = false;
  }
}

/* ── Render results ──────────────────────────────────────────────────────────── */
function renderResults(data) {
  resultDataURL   = data.result_image;
  originalDataURL = data.original_image;

  // Set images
  resultImg.src   = resultDataURL;
  splitOrigImg.src = originalDataURL;
  splitResImg.src  = resultDataURL;

  // View state
  setView("result");

  // Count + prompt
  const n = data.detections.length;
  countBadge.textContent = n === 0 ? "0 defects" : n === 1 ? "1 defect" : `${n} defects`;
  samPromptTag.textContent = data.sam_prompt ? `SAM3 prompt: "${data.sam_prompt}"` : "";

  // Detection cards
  detectionsGrid.innerHTML = "";
  if (n === 0) {
    detectionsGrid.innerHTML =
      `<p style="color:var(--text-dim);font-size:13px;">No defects detected at this confidence threshold. Try lowering it.</p>`;
  } else {
    data.detections.forEach((det, i) => {
      detectionsGrid.appendChild(buildDetCard(det, i));
    });
  }

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function buildDetCard(det, i) {
  const colour = PALETTE[i % PALETTE.length];
  const [x1, y1, x2, y2] = det.box;
  const w = x2 - x1;
  const h = y2 - y1;

  const card = document.createElement("div");
  card.className = "det-card";
  card.innerHTML = `
    <div class="det-card-header">
      <div class="det-swatch" style="background:${colour}"></div>
      <span class="det-class">${escHtml(det.class)}</span>
      <span class="det-conf">${(det.confidence * 100).toFixed(1)}%</span>
    </div>
    <div class="det-metrics">
      <div class="det-metric"><span>Area</span><span>${det.area_px.toLocaleString()} px (${det.area_pct}%)</span></div>
      <div class="det-metric"><span>Box size</span><span>${w} × ${h} px</span></div>
      <div class="det-metric"><span>SAM3 score</span><span>${det.sam_score.toFixed(3)}</span></div>
      <div class="det-metric"><span>IoB</span><span>${(det.sam_iob * 100).toFixed(1)}%</span></div>
    </div>
    <div class="det-box-coords">[${x1}, ${y1}, ${x2}, ${y2}]</div>
  `;
  return card;
}

/* ── View toggle ─────────────────────────────────────────────────────────────── */
document.querySelectorAll(".view-btn").forEach(btn => {
  btn.addEventListener("click", () => setView(btn.dataset.view));
});

function setView(view) {
  currentView = view;
  document.querySelectorAll(".view-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.view === view)
  );
  if (view === "split") {
    singleView.hidden = true;
    splitView.hidden = false;
  } else {
    singleView.hidden = false;
    splitView.hidden = true;
    resultImg.src = view === "result" ? resultDataURL : originalDataURL;
  }
}

/* ── Split drag ──────────────────────────────────────────────────────────────── */
const splitDivider = $("splitDivider");
let splitDragging = false;

splitDivider.addEventListener("mousedown", e => {
  e.preventDefault();
  splitDragging = true;
});

document.addEventListener("mousemove", e => {
  if (!splitDragging) return;
  const rect = splitView.getBoundingClientRect();
  const ratio = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0.1), 0.9);
  const panes = splitView.querySelectorAll(".split-pane");
  panes[0].style.flex = `${ratio}`;
  panes[1].style.flex = `${1 - ratio}`;
});

document.addEventListener("mouseup", () => { splitDragging = false; });

/* ── Export ──────────────────────────────────────────────────────────────────── */
exportBtn.addEventListener("click", () => {
  if (!resultDataURL) return;
  const a = document.createElement("a");
  a.href = resultDataURL;
  a.download = `defect-inspection-${Date.now()}.png`;
  a.click();
});

/* ── Error helpers ───────────────────────────────────────────────────────────── */
function showError(msg) {
  errorText.textContent = msg;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorText.textContent = "";
}

function escHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
