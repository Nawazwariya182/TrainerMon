
document.addEventListener("DOMContentLoaded", () => {
  const cohortButtons = document.querySelectorAll(".cohort-btn");
  const fileInput = document.getElementById("visionFileInput");
  const btnRun = document.getElementById("btnRunVision");
  const heroName = document.getElementById("heroVisionName");
  const heroConf = document.getElementById("heroVisionConf");
  const histogram = document.getElementById("visionHistogram");

  let selectedClass = cohortButtons.length > 0 ? cohortButtons[0].getAttribute("data-class") : "Sample";

  cohortButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      cohortButtons.forEach(b => b.classList.remove("btn-active"));
      btn.classList.add("btn-active");
      selectedClass = btn.getAttribute("data-class");
      runInference(selectedClass);
    });
  });

  btnRun.addEventListener("click", () => runInference(selectedClass));

  async function runInference(sample) {
    heroConf.textContent = "Evaluating vision convolution...";
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample })
      });
      const data = await res.json();
      heroName.textContent = data.predicted_class;
      heroConf.textContent = `Confidence: ${data.confidence_pct} • Latency: ${data.latency_ms}ms`;

      histogram.innerHTML = "";
      for (const [cls, prob] of Object.entries(data.probabilities)) {
        const pct = (prob * 100).toFixed(1) + "%";
        const isWinner = cls === data.predicted_class;
        const row = document.createElement("div");
        row.className = "prob-row" + (isWinner ? " prob-row-winner" : "");
        row.innerHTML = `
          <div class="prob-meta"><span>${cls}</span><span>${pct}</span></div>
          <div class="prob-track"><div class="prob-fill" style="width: ${pct};"></div></div>
        `;
        histogram.appendChild(row);
      }
    } catch (e) {
      heroConf.textContent = "Error: " + e.message;
    }
  }
});
