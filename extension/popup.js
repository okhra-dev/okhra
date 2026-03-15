const $ = (sel) => document.querySelector(sel);

let thresholdsCache = null;
let lastMeanProb    = null;

function setStatus(msg, cls) {
  const el = $("#status");
  el.textContent = msg;
  el.className = "status " + (cls || "");
}

function probBg(p, alpha) {
  const hue = (1 - p) * 120;
  return `hsla(${hue},75%,40%,${alpha || 0.35})`;
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function getThreshold() {
  if (!thresholdsCache) return { fpr: 0.01, threshold: 0.5 };
  const fpr = $("#fpr-select").value;
  const t = thresholdsCache[fpr] || thresholdsCache["0.01"];
  return { fpr: parseFloat(fpr), threshold: t.threshold };
}

function showResults(data) {
  if (data.error) {
    setStatus(data.error, "error");
    return;
  }

  thresholdsCache = data.thresholds;
  lastMeanProb    = data.meanProb;

  const { threshold, fpr } = getThreshold();
  const isAI = data.meanProb >= threshold;
  const pct  = (data.meanProb * 100).toFixed(2);

  const v = $("#verdict");
  v.textContent = isAI ? "AI-Generated" : "Human-Written";
  v.className   = isAI ? "ai" : "human";

  const bar = $("#prob-bar");
  bar.style.width      = `${pct}%`;
  bar.style.background = probBg(data.meanProb, 0.8);

  $("#prob-label").textContent =
    `${pct}% · ${data.numChunks} chunk${data.numChunks > 1 ? "s" : ""} · threshold ${(threshold * 100).toFixed(2)}%`;

  let html = "";
  for (let i = 0; i < data.displayWords.length; i++) {
    const w = data.displayWords[i];
    const p = data.wordProbs[i];
    const space = (i > 0) ? " " : "";
    html += `<span style="background:${probBg(p, 0.35)}">${space}${escapeHtml(w)}</span>`;
  }
  $("#chunk-viz").innerHTML = html;

  setStatus("Done", "ready");
  $("#results").classList.remove("hidden");
}

function updateVerdict() {
  if (lastMeanProb === null || !thresholdsCache) return;
  const { threshold, fpr } = getThreshold();
  const isAI = lastMeanProb >= threshold;
  const v = $("#verdict");
  v.textContent = isAI ? "AI-Generated" : "Human-Written";
  v.className   = isAI ? "ai" : "human";
  $("#prob-label").textContent =
    `${(lastMeanProb * 100).toFixed(2)}% · threshold ${(threshold * 100).toFixed(2)}%`;
}

async function analyze() {
  const text = $("#text-input").value.trim();
  if (!text) return;

  $("#analyze-btn").disabled = true;
  setStatus("Analyzing…");
  $("#results").classList.add("hidden");

  chrome.runtime.sendMessage(
    { type: "okhra-analyze", text },
    (response) => {
      if (chrome.runtime.lastError) {
        setStatus("Error: " + chrome.runtime.lastError.message, "error");
      } else {
        showResults(response);
      }
      $("#analyze-btn").disabled = false;
    }
  );
}

// ── Theme ─────────────────────────────────────────────────

function applyTheme(theme) {
  document.body.classList.toggle("light", theme === "light");
  chrome.storage.local.set({ theme });
}

// ── Init ──────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  // Restore theme
  const { theme } = await chrome.storage.local.get("theme");
  if (theme === "light") applyTheme("light");

  // Check if background model is ready
  chrome.runtime.sendMessage({ type: "okhra-status" }, (res) => {
    if (chrome.runtime.lastError) {
      setStatus("Background not ready — reopen in a moment", "error");
      return;
    }
    if (res && res.ready) {
      $("#analyze-btn").disabled = false;
      setStatus("Ready", "ready");
    } else {
      setStatus("Model loading…");
      // Poll until ready
      const poll = setInterval(() => {
        chrome.runtime.sendMessage({ type: "okhra-status" }, (r) => {
          if (r && r.ready) {
            clearInterval(poll);
            $("#analyze-btn").disabled = false;
            setStatus("Ready", "ready");
          }
        });
      }, 500);
    }
  });

  // Check for pending context-menu text
  const stored = await chrome.storage.local.get(["pendingText"]);
  if (stored.pendingText) {
    $("#text-input").value = stored.pendingText;
    chrome.storage.local.remove(["pendingText"]);

    // Wait for model then auto-analyze
    const waitAndAnalyze = () => {
      chrome.runtime.sendMessage({ type: "okhra-status" }, (r) => {
        if (r && r.ready) {
          analyze();
        } else {
          setTimeout(waitAndAnalyze, 300);
        }
      });
    };
    waitAndAnalyze();
  }

  // Events
  $("#analyze-btn").addEventListener("click", analyze);
  $("#text-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) analyze();
  });
  $("#fpr-select").addEventListener("change", updateVerdict);
  $("#theme-toggle").addEventListener("click", () => {
    const isLight = document.body.classList.contains("light");
    applyTheme(isLight ? "dark" : "light");
  });
});
