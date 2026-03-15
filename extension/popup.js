let session      = null;
let vocabMap     = {};
let selectWords  = new Set();
let thresholds   = {};
let modelConfig  = {};

// ── helpers ──────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);

async function loadJSON(path) {
  const url = chrome.runtime.getURL(path);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}

function setStatus(msg, cls) {
  const el = $("#status");
  el.textContent = msg;
  el.className = "status " + (cls || "");
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function probColor(p, alpha = 0.45) {
  const hue = (1 - p) * 120;           
  return `hsla(${hue},80%,42%,${alpha})`;
}

function escapeHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function getThreshold() {
  const fpr = $("#fpr-select").value;
  const t = thresholds[fpr] || thresholds["0.01"];
  return { fpr: parseFloat(fpr), threshold: t.threshold, tpr: t.tpr };
}

// ── ONNX inference ───────────────────────────────────────

async function runInference(chunks) {
  const B   = chunks.length;
  const L   = modelConfig.max_len;
  const buf = new BigInt64Array(B * L);      // zero-padded (PAD=0)

  for (let i = 0; i < B; i++)
    for (let j = 0; j < chunks[i].length; j++)
      buf[i * L + j] = BigInt(chunks[i][j]);

  const input  = new ort.Tensor("int64", buf, [B, L]);
  const output = await session.run({ input_ids: input });
  return Array.from(output.logits.data, sigmoid);
}

// ── main flow ────────────────────────────────────────────

async function analyze() {
  const text = $("#text-input").value.trim();
  if (!text || !session) return;

  $("#analyze-btn").disabled = true;
  setStatus("Analyzing…");
  $("#results").classList.add("hidden");

  try {
    const { tokens, displayWords } = tokenizeText(text, selectWords);
    const indices = tokensToIndices(tokens, vocabMap, modelConfig.unk_idx);
    const { chunks, starts } = chunkIndices(
      indices, modelConfig.max_len, modelConfig.overlap
    );
    const probs = await runInference(chunks);
    const wordProbs = mapProbsToWords(probs, starts, chunks, displayWords.length);
    const meanProb  = probs.reduce((a, b) => a + b, 0) / probs.length;

    showResults(meanProb, wordProbs, displayWords, probs.length);
  } catch (e) {
    setStatus("Error: " + e.message, "error");
    console.error(e);
  }
  $("#analyze-btn").disabled = false;
}

function showResults(meanProb, wordProbs, displayWords, nChunks) {
  const { threshold, fpr } = getThreshold();
  const isAI = meanProb >= threshold;

  // verdict
  const verdictEl = $("#verdict");
  verdictEl.textContent = isAI ? "AI-Generated" : "Human-Written";
  verdictEl.className   = isAI ? "ai" : "human";

  // probability bar
  const bar = $("#prob-bar");
  bar.style.width      = `${(meanProb * 100).toFixed(1)}%`;
  bar.style.background = probColor(meanProb, 0.9);

  $("#prob-label").textContent =
    `Mean probability ${(meanProb * 100).toFixed(2)}%` +
    ` · threshold ${(threshold * 100).toFixed(2)}% (FPR ≤ ${fpr * 100}%)`;

  // chunk-coloured text
  $("#chunk-count").textContent = ` (${nChunks} chunk${nChunks > 1 ? "s" : ""})`;

  let html = "";
  for (let i = 0; i < displayWords.length; i++) {
    const w = displayWords[i];
    const p = wordProbs[i];
    const needSpace = i > 0 && !PUNCT_SET.has(w);
    if (needSpace) html += " ";
    html += `<span class="word" style="background:${probColor(p)}"` +
            ` title="p=${(p*100).toFixed(1)}%">${escapeHtml(w)}</span>`;
  }
  $("#chunk-viz").innerHTML = html;
  setStatus("Done", "ready");
  $("#results").classList.remove("hidden");
}

function updateVerdict() {
  const vizWords = $("#chunk-viz")?.querySelectorAll(".word");
  if (!vizWords || vizWords.length === 0) return;
  const meanProb = parseFloat($("#prob-bar").style.width) / 100;
  const { threshold, fpr } = getThreshold();
  const isAI = meanProb >= threshold;
  const v = $("#verdict");
  v.textContent = isAI ? "AI-Generated" : "Human-Written";
  v.className   = isAI ? "ai" : "human";
  $("#prob-label").textContent =
    `Mean probability ${(meanProb * 100).toFixed(2)}%` +
    ` · threshold ${(threshold * 100).toFixed(2)}% (FPR ≤ ${fpr * 100}%)`;
}

// ── init ─────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  try {
    [vocabMap, selectWords, thresholds, modelConfig] = await Promise.all([
      loadJSON("model/vocab.json"),
      loadJSON("model/select_words.json").then(a => new Set(a)),
      loadJSON("model/thresholds.json"),
      loadJSON("model/model_config.json")
    ]);

    ort.env.wasm.wasmPaths  = chrome.runtime.getURL("lib/");
    ort.env.wasm.numThreads = 1;

    session = await ort.InferenceSession.create(
      chrome.runtime.getURL("model/model.onnx"),
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" }
    );

    $("#analyze-btn").disabled = false;
    setStatus("Ready", "ready");

    // auto-load context-menu selection
    const stored = await chrome.storage.local.get(["pendingText"]);
    if (stored.pendingText) {
      $("#text-input").value = stored.pendingText;
      chrome.storage.local.remove(["pendingText"]);
      chrome.action.setBadgeText({ text: "" });
      analyze();
    }
  } catch (e) {
    setStatus("Model load failed: " + e.message, "error");
    console.error(e);
  }

  $("#analyze-btn").addEventListener("click", analyze);
  $("#text-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) analyze();
  });
  $("#fpr-select").addEventListener("change", updateVerdict);
});
