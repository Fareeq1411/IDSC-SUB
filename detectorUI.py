import os
import tempfile
from uuid import uuid4

import cv2
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename

from disk_detect import DiscDetector
from resnet50_model import resnet50


APP_ROOT = os.path.join(tempfile.gettempdir(), "detectorui_flask")
UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
CROP_DIR = os.path.join(APP_ROOT, "crops")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Glaucoma Screening</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top, rgba(38, 80, 92, 0.45), transparent 35%),
        linear-gradient(180deg, #07131a 0%, #0a1820 100%);
    }

    .glass-card {
      background: rgba(14, 32, 42, 0.82);
      border: 1px solid rgba(45, 94, 112, 0.55);
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(18px);
    }

    .stage-shell {
      background:
        radial-gradient(circle at top, rgba(25, 67, 79, 0.4), transparent 38%),
        linear-gradient(180deg, #0b1720 0%, #102430 100%);
    }

    .stage-grid {
      background-image:
        linear-gradient(rgba(120, 172, 191, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(120, 172, 191, 0.05) 1px, transparent 1px);
      background-size: 28px 28px;
    }

    .scan-box {
      position: absolute;
      border: 3px solid #f5c451;
      border-radius: 16px;
      box-shadow: 0 0 0 1px rgba(245, 196, 81, 0.25), 0 0 24px rgba(245, 196, 81, 0.2);
      transition: all 0.18s ease;
    }

    .lock-box {
      position: absolute;
      border: 4px solid #54f2b4;
      border-radius: 18px;
      box-shadow: 0 0 0 1px rgba(84, 242, 180, 0.25), 0 0 32px rgba(84, 242, 180, 0.22);
    }

    .analysis-line {
      position: absolute;
      left: 18px;
      right: 18px;
      top: 0;
      height: 4px;
      border-radius: 999px;
      background: linear-gradient(90deg, transparent, rgba(109, 211, 206, 0.95), transparent);
      box-shadow: 0 0 24px rgba(109, 211, 206, 0.7);
      animation: analysisSweep 3s linear forwards;
    }

    @keyframes analysisSweep {
      from { top: 8px; }
      to { top: calc(100% - 12px); }
    }

    .frame-banner {
      background: rgba(8, 22, 30, 0.88);
      border: 1px solid rgba(35, 72, 87, 0.95);
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.24);
    }

    .frame-banner.detecting {
      background: rgba(49, 38, 8, 0.9);
      border-color: rgba(213, 161, 44, 0.95);
      color: #ffe6a1;
    }

    .frame-banner.locked {
      background: rgba(8, 39, 26, 0.9);
      border-color: rgba(57, 183, 132, 0.95);
      color: #c6ffe7;
    }

    .frame-banner.analysis {
      background: rgba(8, 32, 35, 0.9);
      border-color: rgba(55, 191, 168, 0.95);
      color: #c9fff4;
    }

    .frame-banner.safe {
      background: rgba(10, 42, 31, 0.92);
      border-color: rgba(52, 190, 137, 0.95);
      color: #cffff0;
    }

    .frame-banner.alert {
      background: rgba(53, 13, 18, 0.92);
      border-color: rgba(204, 94, 109, 0.95);
      color: #ffd1d8;
    }

    .frame-banner.error {
      background: rgba(46, 15, 22, 0.92);
      border-color: rgba(194, 99, 114, 0.95);
      color: #ffe0e5;
    }

    .result-pulse-safe {
      animation: pulseSafe 0.9s ease-in-out 6;
    }

    .result-pulse-alert {
      animation: pulseAlert 0.9s ease-in-out 6;
    }

    @keyframes pulseSafe {
      0%, 100% { box-shadow: 0 0 0 rgba(52, 190, 137, 0.1); }
      50% { box-shadow: 0 0 32px rgba(52, 190, 137, 0.28); }
    }

    @keyframes pulseAlert {
      0%, 100% { box-shadow: 0 0 0 rgba(204, 94, 109, 0.1); }
      50% { box-shadow: 0 0 36px rgba(204, 94, 109, 0.28); }
    }

    .image-fade {
      animation: imageFade 0.35s ease;
    }

    @keyframes imageFade {
      from { opacity: 0.25; transform: scale(0.985); }
      to { opacity: 1; transform: scale(1); }
    }
  </style>
</head>
<body class="min-h-screen overflow-y-auto text-slate-100">
  <div class="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-3 md:px-5 lg:px-7">
    <header class="glass-card mb-2 rounded-[22px] px-4 py-3 md:px-5">
      <div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div class="max-w-3xl">
          <p class="mb-1 text-[10px] font-semibold uppercase tracking-[0.28em] text-cyan-200/70">DFI Glaucoma Screening</p>
          <h1 class="text-[1.7rem] font-bold tracking-tight text-slate-50 md:text-[2rem]">Glaucoma Screening</h1>
          <p class="mt-1.5 max-w-2xl text-[13px] leading-5 text-slate-300">
            Upload a DFI eye scan, detect the optic disc, hold the lock visually, crop the disc region, then run glaucoma analysis with staged on-screen animation.
          </p>
        </div>
        <div class="flex flex-col gap-2 sm:flex-row">
          <input id="fileInput" type="file" accept=".png,.jpg,.jpeg,.bmp" class="hidden">
          <button id="uploadButton" class="rounded-2xl bg-emerald-400 px-6 py-2.5 text-sm font-extrabold uppercase tracking-[0.18em] text-slate-950 transition hover:bg-emerald-300">
            Upload Eye Scan
          </button>
          <button id="resetButton" class="rounded-2xl border border-slate-600/80 bg-slate-900/50 px-6 py-2.5 text-sm font-bold uppercase tracking-[0.18em] text-slate-200 transition hover:border-slate-400 hover:text-white">
            Reset View
          </button>
        </div>
      </div>
    </header>

    <section class="hidden">
      <article class="glass-card rounded-[18px] p-2.5">
        <p class="text-[11px] font-semibold uppercase tracking-[0.3em] text-cyan-200/65">Current Phase</p>
        <h2 id="phaseValue" class="mt-1 text-base font-bold text-slate-50">Waiting for image</h2>
        <p id="phaseDetail" class="mt-1 text-[13px] leading-5 text-slate-300">
          The main frame stays focused on the uploaded image and each processing stage.
        </p>
      </article>
      <article class="glass-card rounded-[18px] p-2.5">
        <p class="text-[11px] font-semibold uppercase tracking-[0.3em] text-cyan-200/65">Prediction Output</p>
        <div id="resultBadge" class="mt-1.5 rounded-2xl border border-slate-600/70 bg-slate-900/50 px-4 py-2 text-center text-base font-extrabold text-slate-100">
          Awaiting analysis
        </div>
        <p id="confidenceValue" class="mt-1.5 text-[13px] font-semibold text-slate-300">Confidence: --</p>
        <p id="coordsValue" class="mt-1 text-[13px] leading-5 text-slate-300">Optic disc coordinates: --</p>
      </article>
    </section>

    <main id="mainFrame" class="glass-card stage-shell stage-grid relative flex overflow-visible rounded-[30px] border border-cyan-900/40 p-3 md:p-4">
      <div class="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(38,97,117,0.20),transparent_35%)]"></div>
      <div class="relative z-10 flex w-full flex-col">
        <div class="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p class="text-xs font-semibold uppercase tracking-[0.3em] text-cyan-200/65">Main Processing Frame</p>
            <p class="mt-1 text-sm text-slate-300">Upload, detect, lock, crop, analyse, and show the glaucoma result inside the picture area.</p>
          </div>
          <div id="miniStatus" class="rounded-full border border-slate-700/70 bg-slate-950/55 px-4 py-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-200">
            Idle
          </div>
        </div>

        <div class="relative overflow-hidden rounded-[28px] border border-cyan-950/80 bg-[#081118]/90">
          <div id="stageViewport" class="relative p-3 md:p-4">
            <div id="emptyState" class="flex min-h-[420px] items-center justify-center">
              <div class="max-w-xl px-6 text-center">
                <p class="text-sm font-semibold uppercase tracking-[0.35em] text-cyan-200/55">No Scan Loaded</p>
                <h3 class="mt-4 text-3xl font-bold text-slate-50">Upload a DFI eye scan to begin</h3>
                <p class="mt-4 text-sm leading-7 text-slate-300">
                  The central frame will show the uploaded image, random optic-disc search animation, locked coordinates, cropped optic disc, analysis sweep, and the final glaucoma result.
                </p>
              </div>
            </div>

            <div id="visualCanvas" class="relative mx-auto hidden w-full max-w-[620px]">
              <img id="mainImage" class="image-fade block w-full rounded-[24px] object-contain" alt="DFI eye scan preview">
              <div id="overlayLayer" class="pointer-events-none absolute inset-0"></div>

              <div id="topStatusPill" class="pointer-events-none absolute left-8 top-8 hidden rounded-2xl border border-slate-700/70 bg-slate-950/75 px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-slate-100 backdrop-blur">
                Idle
              </div>

              <div id="frameBanner" class="frame-banner pointer-events-none absolute bottom-8 left-1/2 hidden w-[min(680px,calc(100%-3rem))] -translate-x-1/2 rounded-[22px] px-6 py-4 text-center">
                <div id="frameBannerTitle" class="text-2xl font-extrabold tracking-[0.08em]">Awaiting analysis</div>
                <div id="frameBannerSubtext" class="mt-2 text-sm font-semibold text-slate-200/90"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>

  <script>
    const dom = {
      fileInput: document.getElementById("fileInput"),
      uploadButton: document.getElementById("uploadButton"),
      resetButton: document.getElementById("resetButton"),
      phaseValue: document.getElementById("phaseValue"),
      phaseDetail: document.getElementById("phaseDetail"),
      resultBadge: document.getElementById("resultBadge"),
      confidenceValue: document.getElementById("confidenceValue"),
      coordsValue: document.getElementById("coordsValue"),
      miniStatus: document.getElementById("miniStatus"),
      mainFrame: document.getElementById("mainFrame"),
      visualCanvas: document.getElementById("visualCanvas"),
      mainImage: document.getElementById("mainImage"),
      emptyState: document.getElementById("emptyState"),
      overlayLayer: document.getElementById("overlayLayer"),
      topStatusPill: document.getElementById("topStatusPill"),
      frameBanner: document.getElementById("frameBanner"),
      frameBannerTitle: document.getElementById("frameBannerTitle"),
      frameBannerSubtext: document.getElementById("frameBannerSubtext")
    };

    let randomBoxTimer = null;
    let currentObjectUrl = null;
    let runToken = 0;

    const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    function setPhase(title, detail) {
      dom.phaseValue.textContent = title;
      dom.phaseDetail.textContent = detail;
    }

    function setPrediction(text, confidence = "--", coords = "--") {
      dom.resultBadge.textContent = text;
      dom.confidenceValue.textContent = `Confidence: ${confidence}`;
      dom.coordsValue.textContent = `Optic disc coordinates: ${coords}`;
    }

    function setMiniStatus(text) {
      dom.miniStatus.textContent = text;
      dom.topStatusPill.textContent = text;
      dom.topStatusPill.classList.remove("hidden");
    }

    function setFrameBanner(title, tone, subtext = "") {
      dom.frameBanner.className = "frame-banner pointer-events-none absolute bottom-8 left-1/2 w-[min(680px,calc(100%-3rem))] -translate-x-1/2 rounded-[22px] px-6 py-4 text-center";
      dom.frameBanner.classList.add(tone);
      dom.frameBanner.classList.remove("hidden");
      dom.frameBannerTitle.textContent = title;
      dom.frameBannerSubtext.textContent = subtext;
    }

    function focusMainFrame() {
      window.requestAnimationFrame(() => {
        dom.mainFrame.scrollIntoView({
          behavior: "smooth",
          block: "start"
        });
      });
    }

    function clearOverlay() {
      dom.overlayLayer.innerHTML = "";
    }

    function clearImage() {
      if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
      }
      dom.mainImage.removeAttribute("src");
      dom.mainImage.classList.add("hidden");
      dom.visualCanvas.classList.add("hidden");
      dom.emptyState.classList.remove("hidden");
      clearOverlay();
      dom.frameBanner.classList.add("hidden");
      dom.topStatusPill.classList.add("hidden");
    }

    function resetUi() {
      stopRandomBoxes();
      clearImage();
      setPhase("Waiting for image", "The main frame stays focused on the uploaded image and each processing stage.");
      setPrediction("Awaiting analysis");
      setMiniStatus("Idle");
      dom.resultBadge.className = "mt-1.5 rounded-2xl border border-slate-600/70 bg-slate-900/50 px-4 py-2 text-center text-base font-extrabold text-slate-100";
      dom.uploadButton.disabled = false;
      dom.uploadButton.classList.remove("opacity-60", "cursor-not-allowed");
    }

    function setImageSource(src) {
      dom.mainImage.classList.add("hidden");
      dom.mainImage.onload = () => {
        dom.emptyState.classList.add("hidden");
        dom.visualCanvas.classList.remove("hidden");
        dom.mainImage.classList.remove("hidden");
        focusMainFrame();
      };
      dom.mainImage.src = src;
    }

    function waitForImageLoad(src) {
      return new Promise((resolve, reject) => {
        dom.mainImage.onload = () => {
          dom.emptyState.classList.add("hidden");
          dom.visualCanvas.classList.remove("hidden");
          dom.mainImage.classList.remove("hidden");
          focusMainFrame();
          resolve();
        };
        dom.mainImage.onerror = () => reject(new Error("Image could not be loaded in browser."));
        dom.mainImage.src = src;
      });
    }

    function startRandomBoxes() {
      stopRandomBoxes();
      clearOverlay();
      randomBoxTimer = window.setInterval(() => {
        clearOverlay();
        const box = document.createElement("div");
        box.className = "scan-box";
        const width = 18 + Math.random() * 16;
        const height = 18 + Math.random() * 16;
        const left = Math.random() * (100 - width);
        const top = Math.random() * (100 - height);
        box.style.left = `${left}%`;
        box.style.top = `${top}%`;
        box.style.width = `${width}%`;
        box.style.height = `${height}%`;
        dom.overlayLayer.appendChild(box);
      }, 220);
    }

    function stopRandomBoxes() {
      if (randomBoxTimer) {
        window.clearInterval(randomBoxTimer);
        randomBoxTimer = null;
      }
    }

    function showLockBox(coords, imageWidth, imageHeight) {
      clearOverlay();
      const box = document.createElement("div");
      box.className = "lock-box";
      box.style.left = `${(coords.xmin / imageWidth) * 100}%`;
      box.style.top = `${(coords.ymin / imageHeight) * 100}%`;
      box.style.width = `${((coords.xmax - coords.xmin) / imageWidth) * 100}%`;
      box.style.height = `${((coords.ymax - coords.ymin) / imageHeight) * 100}%`;
      dom.overlayLayer.appendChild(box);
    }

    function startAnalysisSweep() {
      clearOverlay();
      const line = document.createElement("div");
      line.className = "analysis-line";
      dom.overlayLayer.appendChild(line);
    }

    function setResultBadgeTone(predictionText) {
      dom.resultBadge.className = "mt-1.5 rounded-2xl px-4 py-2 text-center text-base font-extrabold";
      if (predictionText === "GLAUCOMA DETECTED") {
        dom.resultBadge.classList.add("border", "border-rose-500/70", "bg-rose-950/70", "text-rose-100", "result-pulse-alert");
      } else {
        dom.resultBadge.classList.add("border", "border-emerald-500/70", "bg-emerald-950/60", "text-emerald-100", "result-pulse-safe");
      }
    }

    async function analyzeFile(file) {
      const form = new FormData();
      form.append("image", file);

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: form
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Processing failed.");
      }
      return data;
    }

    async function runPipeline(file) {
      runToken += 1;
      const token = runToken;

      dom.uploadButton.disabled = true;
      dom.uploadButton.classList.add("opacity-60", "cursor-not-allowed");
      dom.resultBadge.className = "mt-1.5 rounded-2xl border border-slate-600/70 bg-slate-900/50 px-4 py-2 text-center text-base font-extrabold text-slate-100";

      if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
      }
      currentObjectUrl = URL.createObjectURL(file);
      await waitForImageLoad(currentObjectUrl);

      setPhase("Scanning optic disc", `Loaded ${file.name}.`);
      setPrediction("DETECTION RUNNING", "--", "searching...");
      setMiniStatus("Scanning");
      setFrameBanner("DETECTING OPTIC DISC", "detecting");

      startRandomBoxes();

      const detectStart = performance.now();
      const backendPromise = analyzeFile(file);
      const data = await backendPromise;
      const detectElapsed = performance.now() - detectStart;
      await delay(Math.max(0, 3000 - detectElapsed));

      if (token !== runToken) {
        return;
      }

      stopRandomBoxes();
      showLockBox(data.coords, data.image_width, data.image_height);
      setPhase("Optic disc locked", "Detector output received. Holding the optic disc lock for 3 seconds before cropping.");
      setPrediction(
        "DETECTION COMPLETE",
        "--",
        `xmin=${data.coords.xmin}, xmax=${data.coords.xmax}, ymin=${data.coords.ymin}, ymax=${data.coords.ymax}`
      );
      setMiniStatus("Locked");
      setFrameBanner("OPTIC DISC LOCKED", "locked", "Holding lock before cropping");

      await delay(3000);
      if (token !== runToken) {
        return;
      }

      await waitForImageLoad(data.crop_url);
      clearOverlay();
      setPhase("Analysing cropped disc", "The crop is now displayed in the main frame while the glaucoma classifier result is prepared.");
      setPrediction("ANALYSIS RUNNING", "calculating...", `xmin=${data.coords.xmin}, xmax=${data.coords.xmax}, ymin=${data.coords.ymin}, ymax=${data.coords.ymax}`);
      setMiniStatus("Analysing");
      setFrameBanner("RUNNING GLAUCOMA ANALYSIS", "analysis", "Processing cropped optic disc");
      startAnalysisSweep();

      const analysisStart = performance.now();
      const analysisTicker = window.setInterval(() => {
        const elapsed = performance.now() - analysisStart;
        const dots = ".".repeat((Math.floor(elapsed / 400) % 4));
        dom.topStatusPill.textContent = `Analysing${dots}`;
        dom.miniStatus.textContent = `Analysing${dots}`;
        dom.frameBannerSubtext.textContent = `Processing cropped optic disc${dots}`;
      }, 220);

      await delay(3000);
      window.clearInterval(analysisTicker);

      if (token !== runToken) {
        return;
      }

      clearOverlay();
      setPhase("Analysis complete", "Model inference finished on the cropped optic disc image.");
      setPrediction(
        data.prediction_text,
        `${data.confidence_percent.toFixed(2)}%`,
        `xmin=${data.coords.xmin}, xmax=${data.coords.xmax}, ymin=${data.coords.ymin}, ymax=${data.coords.ymax}`
      );
      setMiniStatus("Complete");
      setFrameBanner(
        data.prediction_text,
        data.prediction_text === "GLAUCOMA DETECTED" ? "alert" : "safe",
          `Confidence ${data.confidence_percent.toFixed(2)}%`
      );
      setResultBadgeTone(data.prediction_text);

      dom.uploadButton.disabled = false;
      dom.uploadButton.classList.remove("opacity-60", "cursor-not-allowed");
    }

    function showError(message) {
      stopRandomBoxes();
      clearOverlay();
      setPhase("Process failed", message);
      setPrediction("PROCESS FAILED", "--", "--");
      setMiniStatus("Failed");
      setFrameBanner("PROCESS FAILED", "error", message);
      dom.resultBadge.className = "mt-1.5 rounded-2xl border border-rose-500/70 bg-rose-950/70 px-4 py-2 text-center text-base font-extrabold text-rose-100";
      dom.uploadButton.disabled = false;
      dom.uploadButton.classList.remove("opacity-60", "cursor-not-allowed");
    }

    dom.uploadButton.addEventListener("click", () => {
      dom.fileInput.click();
    });

    dom.resetButton.addEventListener("click", () => {
      runToken += 1;
      resetUi();
    });

    dom.fileInput.addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) {
        return;
      }

      try {
        await runPipeline(file);
      } catch (error) {
        showError(error.message || "Unexpected processing error.");
      } finally {
        dom.fileInput.value = "";
      }
    });

    resetUi();
  </script>
</body>
</html>
"""


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_storage_path(base_dir, original_name):
    safe_name = secure_filename(original_name) or "scan.jpg"
    stem, ext = os.path.splitext(safe_name)
    return os.path.join(base_dir, f"{stem}_{uuid4().hex}{ext.lower()}")


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "No image file was uploaded."}), 400

    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, JPEG, or BMP."}), 400

    upload_path = build_storage_path(UPLOAD_DIR, uploaded.filename)
    uploaded.save(upload_path)

    original = cv2.imread(upload_path)
    if original is None:
        return jsonify({"error": "Uploaded image could not be read."}), 400

    image_height, image_width = original.shape[:2]

    try:
        coords = DiscDetector.predict_disk_coords(upload_path)
        if not coords:
            return jsonify({"error": "No optic disc coordinates were returned for this image."}), 400

        crop = DiscDetector.crop_image(upload_path, coords)
        if crop is None or crop.size == 0:
            return jsonify({"error": "The detector returned coordinates, but the crop result was empty."}), 400

        crop_filename = f"crop_{uuid4().hex}.png"
        crop_path = os.path.join(CROP_DIR, crop_filename)
        saved = cv2.imwrite(crop_path, crop)
        if not saved:
            return jsonify({"error": "The cropped optic disc image could not be written."}), 500

        pred_label, probability = resnet50.predict_glaucoma(crop_path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    clean_coords = {
        "xmin": int(coords["xmin"]),
        "xmax": int(coords["xmax"]),
        "ymin": int(coords["ymin"]),
        "ymax": int(coords["ymax"]),
    }

    prediction_text = "GLAUCOMA DETECTED" if pred_label == 1 else "GLAUCOMA UNDETECTED"

    return jsonify(
        {
            "upload_url": f"/media/uploads/{os.path.basename(upload_path)}",
            "crop_url": f"/media/crops/{crop_filename}",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "coords": clean_coords,
            "pred_label": int(pred_label),
            "probability": float(probability),
            "confidence_percent": float(probability) * 100.0,
            "prediction_text": prediction_text,
        }
    )


@app.get("/media/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/media/crops/<path:filename>")
def cropped_file(filename):
    return send_from_directory(CROP_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
