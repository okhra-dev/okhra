var session = null;
var vocabMap = {};
var selectWords = null;
var thresholds = {};
var modelConfig = {};
var lastResult = null;

function $(sel) { return document.querySelector(sel); }

function setStatus(msg, cls) {
  var el = $("#status");
  el.textContent = msg;
  el.className = "status " + (cls || "");
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function getThreshold() {
  var fpr = $("#fpr-select").value;
  var t = thresholds[fpr] || thresholds["0.01"];
  return { fpr: parseFloat(fpr), threshold: t.threshold };
}


function wordColor(p, threshold) {
  if (p < threshold) return "transparent";
  var range = 1.0 - threshold;
  var t = range > 0 ? (p - threshold) / range : 1;
  if (t > 1) t = 1;
  var alpha = 0.15 + t * 0.40; 
  var isDark = document.body.classList.contains("dark");
  if (isDark) {
    return "rgba(190,50,50," + alpha.toFixed(3) + ")";
  }
  return "rgba(200,45,45," + alpha.toFixed(3) + ")";
}

function confidenceLabel(meanProb, threshold) {
  if (meanProb < threshold) {
    var dist = threshold - meanProb;
    if (dist > 0.3) return "High confidence";
    if (dist > 0.1) return "Moderate confidence";
    return "Low confidence";
  } else {
    var dist = meanProb - threshold;
    if (dist > 0.3) return "High confidence";
    if (dist > 0.1) return "Moderate confidence";
    return "Low confidence";
  }
}


function runInference(chunks) {
  var B = chunks.length;
  var L = modelConfig.max_len;
  var buf = new BigInt64Array(B * L);

  for (var i = 0; i < B; i++) {
    for (var j = 0; j < chunks[i].length; j++) {
      buf[i * L + j] = BigInt(chunks[i][j]);
    }
  }

  var input = new ort.Tensor("int64", buf, [B, L]);
  return session.run({ input_ids: input }).then(function(output) {
    var logits = output.logits.data;
    var probs = [];
    for (var k = 0; k < logits.length; k++) {
      probs.push(sigmoid(logits[k]));
    }
    return probs;
  });
}

function showResults() {
  if (!lastResult) return;
  var data = lastResult;
  var info = getThreshold();
  var isAI = data.meanProb >= info.threshold;

  var v = $("#verdict");
  v.textContent = isAI ? "AI-Generated" : "Human-Written";
  v.className = isAI ? "ai" : "human";

  $("#confidence").textContent = confidenceLabel(data.meanProb, info.threshold);

  var html = "";
  for (var i = 0; i < data.displayWords.length; i++) {
    var w = data.displayWords[i];
    var p = data.wordProbs[i];
    var bg = wordColor(p, info.threshold);
    var space = (i > 0) ? " " : "";
    html += '<span style="background:' + bg + '">' + space + escapeHtml(w) + '</span>';
  }
  $("#chunk-viz").innerHTML = html;

  setStatus("Done", "ready");
  $("#results").classList.remove("hidden");
}

// ── Analyze ───────────────────────────────────────────────

function analyze() {
  var text = $("#text-input").value.trim();
  if (!text || !session) return;

  $("#analyze-btn").disabled = true;
  setStatus("Analyzing\u2026");
  $("#results").classList.add("hidden");

  try {
    var result = tokenizeText(text, selectWords);
    var tokens = result.tokens;
    var displayWords = result.displayWords;
    var indices = tokensToIndices(tokens, vocabMap, modelConfig.unk_idx);

    if (indices.length === 0) {
      setStatus("Text too short.", "error");
      $("#analyze-btn").disabled = false;
      return;
    }

    var chunked = chunkIndices(indices, modelConfig.max_len, modelConfig.overlap);
    var chunks = chunked.chunks;
    var starts = chunked.starts;

    runInference(chunks).then(function(probs) {
      var wordProbs = mapChunkProbsToWords(probs, starts, chunks, displayWords.length);
      var sum = 0;
      for (var i = 0; i < probs.length; i++) sum += probs[i];
      var meanProb = sum / probs.length;

      lastResult = {
        meanProb: meanProb,
        wordProbs: wordProbs,
        displayWords: displayWords
      };
      showResults();
      $("#analyze-btn").disabled = false;
    }).catch(function(e) {
      setStatus("Inference error: " + e.message, "error");
      console.error(e);
      $("#analyze-btn").disabled = false;
    });
  } catch (e) {
    setStatus("Error: " + e.message, "error");
    console.error(e);
    $("#analyze-btn").disabled = false;
  }
}


function applyTheme(theme) {
  if (theme === "dark") {
    document.body.classList.add("dark");
  } else {
    document.body.classList.remove("dark");
  }
  chrome.storage.local.set({ theme: theme });
  if (lastResult) showResults();
}


function loadJSON(path) {
  return fetch(chrome.runtime.getURL(path)).then(function(res) {
    if (!res.ok) throw new Error("Failed to load " + path);
    return res.json();
  });
}

document.addEventListener("DOMContentLoaded", function() {
  chrome.storage.local.get("theme", function(data) {
    if (data.theme === "dark") applyTheme("dark");
  });

  Promise.all([
    loadJSON("model/vocab.json"),
    loadJSON("model/select_words.json"),
    loadJSON("model/thresholds.json"),
    loadJSON("model/model_config.json")
  ]).then(function(results) {
    vocabMap = results[0];
    selectWords = new Set(results[1]);
    thresholds = results[2];
    modelConfig = results[3];

    setStatus("Loading...\u2026");

    ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    return ort.InferenceSession.create(
      chrome.runtime.getURL("model/model.onnx"),
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" }
    );
  }).then(function(sess) {
    session = sess;
    $("#analyze-btn").disabled = false;
    setStatus("Ready", "ready");

    chrome.storage.local.get(["pendingText"], function(stored) {
      if (stored.pendingText) {
        $("#text-input").value = stored.pendingText;
        chrome.storage.local.remove(["pendingText"]);
        chrome.action.setBadgeText({ text: "" });
        analyze();
      }
    });
  }).catch(function(e) {
    setStatus("Load failed: " + e.message, "error");
    console.error(e);
  });

  $("#analyze-btn").addEventListener("click", analyze);
  $("#text-input").addEventListener("keydown", function(e) {
    if (e.key === "Enter" && e.ctrlKey) analyze();
  });
  $("#fpr-select").addEventListener("change", function() {
    if (lastResult) showResults();
  });
  $("#theme-toggle").addEventListener("click", function() {
    var isDark = document.body.classList.contains("dark");
    applyTheme(isDark ? "light" : "dark");
  });
});
