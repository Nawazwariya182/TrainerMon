// TrainerMon Client Application
document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const dropzone = document.getElementById("dropzone");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const btnChangeImage = document.getElementById("btnChangeImage");
  const btnAnalyze = document.getElementById("btnAnalyze");

  const resultsPlaceholder = document.getElementById("resultsPlaceholder");
  const resultsContent = document.getElementById("resultsContent");
  const predictedClass = document.getElementById("predictedClass");
  const confidenceVal = document.getElementById("confidenceVal");
  const probabilityBars = document.getElementById("probabilityBars");
  const latencyBadge = document.getElementById("latencyBadge");
  const metaLatency = document.getElementById("metaLatency");
  const sampleChips = document.querySelectorAll(".sample-chip");

  let currentFile = null;

  function handleFileSelect(file) {
    if (!file || !file.type.startsWith("image/")) {
      alert("Please upload a valid image file (.jpg, .jpeg, .png)");
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzonePrompt.classList.add("hidden");
      previewContainer.classList.remove("hidden");
      btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
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
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  btnChangeImage.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  // Sample Chips
  sampleChips.forEach((chip) => {
    chip.addEventListener("click", async () => {
      const label = chip.getAttribute("data-label");
      // Create a canvas mock mri slice if remote sample is loading
      const canvas = document.createElement("canvas");
      canvas.width = 224;
      canvas.height = 224;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#111827";
      ctx.fillRect(0, 0, 224, 224);
      // Draw brain contour
      ctx.fillStyle = "#374151";
      ctx.beginPath();
      ctx.ellipse(112, 112, 85, 95, 0, 0, 2 * Math.PI);
      ctx.fill();
      // Draw simulated anomaly
      ctx.fillStyle = label === "notumor" ? "#4b5563" : "#f3f4f6";
      ctx.beginPath();
      ctx.arc(112 + (label === "pituitary" ? 0 : 35), 112 + (label === "pituitary" ? 40 : -20), 22, 0, 2 * Math.PI);
      ctx.fill();

      canvas.toBlob((blob) => {
        const file = new File([blob], `sample_${label}.png`, { type: "image/png" });
        handleFileSelect(file);
      }, "image/png");
    });
  });

  // Run Inference
  btnAnalyze.addEventListener("click", async () => {
    if (!currentFile) return;

    btnAnalyze.disabled = true;
    btnAnalyze.innerHTML = '<span class="status-dot"></span> Analyzing MRI Scan...';

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

      // Render Results
      resultsPlaceholder.classList.add("hidden");
      resultsContent.classList.remove("hidden");

      predictedClass.textContent = data.predicted_class;
      confidenceVal.textContent = data.confidence_pct;
      latencyBadge.textContent = `${data.latency_ms} ms`;
      metaLatency.textContent = `${data.latency_ms} ms`;

      // Render Probability Bars
      probabilityBars.innerHTML = "";
      const probs = data.probabilities || {};
      const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

      sorted.forEach(([cname, prob], idx) => {
        const pct = (prob * 100).toFixed(1);
        const isWinner = idx === 0;
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
          <div class="prob-header">
            <span>${cname.charAt(0).toUpperCase() + cname.slice(1)}</span>
            <span>${pct}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill ${isWinner ? "winner" : ""}" style="width: ${pct}%"></div>
          </div>
        `;
        probabilityBars.appendChild(row);
      });
    } catch (err) {
      alert("Diagnostic Inference Error: " + err.message);
    } finally {
      btnAnalyze.disabled = false;
      btnAnalyze.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" /></svg> Run Neural Diagnostic Inference';
    }
  });
});
