(function() {
  const PANEL_ID = "okhra-panel";

  function getOrCreatePanel() {
    let panel = document.getElementById(PANEL_ID);
    if (panel) return panel;

    panel = document.createElement("div");
    panel.id = PANEL_ID;
    panel.innerHTML = `
      <div id="okhra-header">
        <span id="okhra-title">Okhra</span>
        <select id="okhra-fpr">
          <option value="0.001">0.1%</option>
          <option value="0.005">0.5%</option>
          <option value="0.01" selected>1%</option>
        </select>
        <button id="okhra-close">✕</button>
      </div>
      <div id="okhra-body">
        <div id="okhra-status">Loading model…</div>
      </div>
    `;
    document.body.appendChild(panel);

    document.getElementById("okhra-close").addEventListener("click", () => {
      panel.style.display = "none";
    });

    document.getElementById("okhra-fpr").addEventListener("change", () => {
      if (panel._lastResult) renderResult(panel._lastResult);
    });

    return panel;
  }

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  function probBg(p, alpha) {
    const hue = (1 - p) * 120;
    return `hsla(${hue},75%,40%,${alpha || 0.35})`;
  }

  function escHtml(s) {
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }

  function renderResult(data) {
    const panel = document.getElementById(PANEL_ID);
    if (!panel) return;
    panel._lastResult = data;

    if (data.error) {
      document.getElementById("okhra-body").innerHTML =
        `<div id="okhra-status" class="okhra-error">${data.error}</div>`;
      return;
    }

    const fpr = document.getElementById("okhra-fpr").value;
    const th = data.thresholds[fpr] || data.thresholds["0.01"];
    const threshold = th.threshold;
    const isAI = data.meanProb >= threshold;
    const pct = (data.meanProb * 100).toFixed(2);

    // Build chunk-coloured text: color contiguous regions per chunk
    const { displayWords, wordProbs, numChunks } = data;
    let html = "";
    for (let i = 0; i < displayWords.length; i++) {
      const w = displayWords[i];
      const p = wordProbs[i];
      const space = (i > 0) ? " " : "";
      html += `<span style="background:${probBg(p, 0.35)}">${space}${escHtml(w)}</span>`;
    }

    document.getElementById("okhra-body").innerHTML = `
      <div id="okhra-verdict" class="${isAI ? "okhra-ai" : "okhra-human"}">
        ${isAI ? "AI-Generated" : "Human-Written"}
      </div>
      <div id="okhra-bar-wrap">
        <div id="okhra-bar" style="width:${pct}%;background:${probBg(data.meanProb, 0.8)}"></div>
      </div>
      <div id="okhra-prob">${pct}% · ${numChunks} chunk${numChunks > 1 ? "s" : ""} · threshold ${(threshold * 100).toFixed(2)}%</div>
      <div id="okhra-text">${html}</div>
    `;
  }

  // ── Listen for messages from background ──────────────────

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "okhra-loading") {
      const panel = getOrCreatePanel();
      panel.style.display = "block";
      document.getElementById("okhra-body").innerHTML =
        `<div id="okhra-status">Analyzing…</div>`;
    }

    if (msg.type === "okhra-result") {
      const panel = getOrCreatePanel();
      panel.style.display = "block";
      renderResult(msg.data);
    }
  });

})();
