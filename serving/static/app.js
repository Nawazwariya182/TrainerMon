// =========================================================================
// TRAINERMON CLINICAL INFERENCE APPLICATION
// Project: 5330b70a-8716-492c-aaf6-ea8dd6c9306b
// Archetype: medical_cv
// =========================================================================

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const btnChangeImage = document.getElementById("btnChangeImage");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const resultHero = document.getElementById("resultHero");
  const breakdownSection = document.getElementById("breakdownSection");
  const probList = document.getElementById("probList");
  const latencyVal = document.getElementById("latencyVal");
  const modelStatusBadge = document.getElementById("modelStatusBadge");
  const sampleChips = document.querySelectorAll(".sample-chip");

  let currentFile = null;

  function loadFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      alert("Please upload an image file (PNG, JPG, DICOM preview).");
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzonePrompt.classList.add("hidden");
      previewContainer.classList.remove("hidden");
      btnAnalyze.disabled = false;
      modelStatusBadge.textContent = "Ready for Analysis";
      modelStatusBadge.className = "status-pill status-healthy";
    };
    reader.readAsDataURL(file);
  }

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      loadFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadFile(e.dataTransfer.files[0]);
    }
  });

  btnChangeImage.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  // Quick Cohort Sample Scans
  sampleChips.forEach((chip) => {
    chip.addEventListener("click", async () => {
      const label = chip.getAttribute("data-label");
      chip.textContent = "Loading " + label + "...";
      try {
        const res = await fetch("/samples");
        const data = await res.json();
        let targetSample = null;
        if (data.samples && data.samples.length > 0) {
          targetSample = data.samples.find((s) => s.class_hint.toLowerCase().includes(label)) || data.samples[0];
        }

        if (targetSample) {
          const imgRes = await fetch("/sample_file/" + targetSample.rel_path);
          const blob = await imgRes.blob();
          const file = new File([blob], targetSample.name, { type: blob.type || "image/jpeg" });
          loadFile(file);
        } else {
          // Synthetic high-contrast test slice
          const canvas = document.createElement("canvas");
          canvas.width = 128;
          canvas.height = 128;
          const ctx = canvas.getContext("2d");
          ctx.fillStyle = "#0a0a0f";
          ctx.fillRect(0, 0, 128, 128);
          ctx.strokeStyle = "#475569";
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.arc(64, 64, 50, 0, Math.PI * 2);
          ctx.stroke();

          if (label !== "notumor") {
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(72, 55, 18, 0, Math.PI * 2);
            ctx.fill();
          }

          canvas.toBlob((blob) => {
            const file = new File([blob], label + "_sample.png", { type: "image/png" });
            loadFile(file);
          });
        }
      } catch (err) {
        console.error("Sample load error:", err);
      } finally {
        chip.textContent = label + " Sample";
      }
    });
  });

  // Run Inference
  btnAnalyze.addEventListener("click", async () => {
    if (!currentFile) return;

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = "Running Neural Assessment...";
    modelStatusBadge.textContent = "Processing...";

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || "Inference failed");
      }

      // Display Hero Prediction
      const isPositive = data.predicted_class.toLowerCase() !== "notumor";
      const badgeColor = isPositive ? "var(--accent-rose)" : "var(--accent-emerald)";
      const bgBadgeColor = isPositive ? "var(--accent-rose-bg)" : "var(--accent-emerald-bg)";

      resultHero.innerHTML = `
        <div class="result-content">
          <div class="result-badge-group">
            <span class="result-label">Diagnostic Finding</span>
            <h3 class="result-title" style="color: ${badgeColor}">${data.predicted_class}</h3>
            <span class="result-confidence">${isPositive ? "Pathology Detected" : "Normal / No Tumor"} &bull; ${data.confidence_pct} Certainty</span>
          </div>
          <div class="score-circle" style="border-color: ${badgeColor}; background-color: ${bgBadgeColor}">
            <span class="score-pct" style="color: ${badgeColor}">${data.confidence_pct}</span>
            <span class="score-sub" style="color: ${badgeColor}">CONFIDENCE</span>
          </div>
        </div>
      `;

      // Render Softmax Breakdown
      probList.innerHTML = "";
      const sortedClasses = Object.keys(data.probabilities).sort(
        (a, b) => data.probabilities[b] - data.probabilities[a]
      );

      sortedClasses.forEach((c) => {
        const prob = data.probabilities[c];
        const pctStr = (prob * 100).toFixed(1) + "%";
        const item = document.createElement("div");
        item.className = "prob-item";
        item.innerHTML = `
          <span class="prob-name">${c}</span>
          <div class="prob-bar-container">
            <div class="prob-bar-fill" style="width: ${pctStr}; background-color: ${c === data.predicted_class ? badgeColor : 'var(--text-subtle)'}"></div>
          </div>
          <span class="prob-val font-mono">${pctStr}</span>
        `;
        probList.appendChild(item);
      });

      latencyVal.textContent = data.latency_ms + " ms";
      breakdownSection.classList.remove("hidden");
      modelStatusBadge.textContent = "Assessment Complete";
      modelStatusBadge.className = "status-pill status-healthy";
    } catch (err) {
      console.error(err);
      alert("Error: " + err.message);
      modelStatusBadge.textContent = "Inference Failed";
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4M12 8h.01" />
        </svg>
        Run Neural Diagnostic Inference
      `;
    }
  });
});
