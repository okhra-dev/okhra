var session = null;
var vocabMap = {};
var selectWords = null;
var thresholds = {};
var modelConfig = {};
var lastMeanProb = null;

function $(sel) { return document.querySelector(sel); }

function setStatus(msg, cls) {
  var el = $("#status");
  el.textContent = msg;
  el.className = "status " + (cls || "");
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function chunkColor(p) {
  var hue = (1 - p) * 120; 
  var sat = 35 + p * 20;   
  var lit = 88 - p * 18;  
  if (document.body.classList.contains("dark")) {
    sat = 30 + p * 25;    
    lit = 22 + p * 10;  
  }
  return "hsl(" + hue + "," + sat + "%," + lit + "%)";
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function getThreshold() {
  var fpr = $("#fpr-select").value;
  var t = thresholds[fpr] || thresholds["0.01"];
  return { fpr: parseFloat(fpr), threshold: t.threshold };
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


function showResults(meanProb, wordProbs, displayWords, numChunks) {
  lastMeanProb = meanProb;
  var info = getThreshold();
  var isAI = meanProb >= info.threshold;
  var pct = (meanProb * 100).toFixed(2);

  var v = $("#verdict");
  v.textContent = isAI ? "AI-Generated" : "Human-Written";
  v.className = isAI ? "ai" : "human";

  var bar = $("#prob-bar");
  bar.style.width = pct + "%";
  bar.style.background = chunkColor(meanProb);

  $("#prob-label").textContent =
    pct + "% \u00b7 " + numChunks + " chunk" + (numChunks > 1 ? "s" : "") +
    " \u00b7 threshold " + (info.threshold * 100).toFixed(2) + "%";

  var html = "";
  for (var i = 0; i < displayWords.length; i++) {
    var w = displayWords[i];
    var p = wordProbs[i];
    var bg = chunkColor(p);
    var space = (i > 0) ? " " : "";
    html += '<span style="background:' + bg + '">' + space + escapeHtml(w) + '</span>';
  }
  $("#chunk-viz").innerHTML = html;

  setStatus("Done", "ready");
  $("#results").classList.remove("hidden");
}

function updateVerdict() {
  if (lastMeanProb === null) return;
  var info = getThreshold();
  var isAI = lastMeanProb >= info.threshold;
  var v = $("#verdict");
  v.textContent = isAI ? "AI-Generated" : "Human-Written";
  v.className = isAI ? "ai" : "human";
  $("#prob-label").textContent =
    (lastMeanProb * 100).toFixed(2) + "% \u00b7 threshold " +
    (info.threshold * 100).toFixed(2) + "%";
}


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
      setStatus("Text too short after tokenisation.", "error");
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
      showResults(meanProb, wordProbs, displayWords, probs.length);
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

    setStatus("please wait...\u2026");

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
  $("#fpr-select").addEventListener("change", updateVerdict);
  $("#theme-toggle").addEventListener("click", function() {
    var isDark = document.body.classList.contains("dark");
    applyTheme(isDark ? "light" : "dark");
  });
});
