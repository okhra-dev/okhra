let session     = null;
let vocabMap    = {};
let selectWords = new Set();
let thresholds  = {};
let modelConfig = {};
let ready       = false;
let initPromise = null;

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ── Model init (runs once) ────────────────────────────────

async function init() {
  if (ready) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const load = async (path) => {
      const res = await fetch(chrome.runtime.getURL(path));
      return res.json();
    };

    [vocabMap, selectWords, thresholds, modelConfig] = await Promise.all([
      load("model/vocab.json"),
      load("model/select_words.json").then(a => new Set(a)),
      load("model/thresholds.json"),
      load("model/model_config.json"),
    ]);

    ort.env.wasm.wasmPaths  = chrome.runtime.getURL("lib/");
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd       = true;

    session = await ort.InferenceSession.create(
      chrome.runtime.getURL("model/model.onnx"),
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" }
    );

    ready = true;
    console.log("[Okhra] Model loaded.");
  })();

  return initPromise;
}

// ── Inference ─────────────────────────────────────────────

async function runInference(text) {
  await init();

  const { tokens, displayWords } = tokenizeText(text, selectWords);
  const indices = tokensToIndices(tokens, vocabMap, modelConfig.unk_idx);

  if (indices.length === 0) {
    return { error: "Text too short after tokenisation." };
  }

  const { chunks, starts } = chunkIndices(
    indices, modelConfig.max_len, modelConfig.overlap
  );

  const B = chunks.length;
  const L = modelConfig.max_len;
  const buf = new BigInt64Array(B * L);
  for (let i = 0; i < B; i++)
    for (let j = 0; j < chunks[i].length; j++)
      buf[i * L + j] = BigInt(chunks[i][j]);

  const input  = new ort.Tensor("int64", buf, [B, L]);
  const output = await session.run({ input_ids: input });
  const probs  = Array.from(output.logits.data, sigmoid);

  const wordProbs = mapProbsToWords(probs, starts, chunks, displayWords.length);
  const meanProb  = probs.reduce((a, b) => a + b, 0) / probs.length;

  return {
    meanProb,
    chunkProbs: probs,
    wordProbs,
    displayWords,
    numChunks: B,
    thresholds,
  };
}

// ── Context menu ──────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "okhra-analyze",
    title: "Analyze with Okhra",
    contexts: ["selection"]
  });
  // Pre-load model on install
  init();
});

// Also pre-load when service worker starts
init();

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "okhra-analyze" || !info.selectionText) return;

  // Send "loading" to content script
  try {
    await chrome.tabs.sendMessage(tab.id, {
      type: "okhra-loading"
    });
  } catch (e) {
    // Content script not injected yet — inject it
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"]
    });
    await chrome.scripting.insertCSS({
      target: { tabId: tab.id },
      files: ["content.css"]
    });
    await chrome.tabs.sendMessage(tab.id, { type: "okhra-loading" });
  }

  // Run inference
  const result = await runInference(info.selectionText);

  // Send result to content script
  chrome.tabs.sendMessage(tab.id, {
    type: "okhra-result",
    data: result
  });
});

// ── Message handler (for popup) ───────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "okhra-analyze") {
    runInference(msg.text).then(sendResponse);
    return true; // async response
  }
  if (msg.type === "okhra-status") {
    sendResponse({ ready });
    return false;
  }
});
