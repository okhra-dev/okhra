var PUNCT_CHARS = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~" +
  "\u2013\u2014\u2026\u00AB\u00BB\u2039\u203A" +
  "\u201C\u201D\u2018\u2019\u201E\u201A" +
  "\u2010\u2011\u2012\u2015\u201F\u201B";
var PUNCT_SET = {};
for (var i = 0; i < PUNCT_CHARS.length; i++) {
  PUNCT_SET[PUNCT_CHARS[i]] = true;
}

function separatePunct(text) {
  var out = "";
  for (var i = 0; i < text.length; i++) {
    var ch = text[i];
    if (PUNCT_SET[ch]) {
      out += "  " + ch + "  ";
    } else {
      out += ch;
    }
  }
  return out;
}

function tokenizeText(text, selectWords) {
  var raw = separatePunct(text);
  var rawLc = separatePunct(text.toLowerCase());
  var dispTokens = raw.split(/\s+/).filter(function(t) { return t.length > 0; });
  var lcTokens = rawLc.split(/\s+/).filter(function(t) { return t.length > 0; });
  var synTokens = [];
  for (var i = 0; i < lcTokens.length; i++) {
    var t = lcTokens[i];
    if (PUNCT_SET[t] || selectWords.has(t)) {
      synTokens.push(t);
    } else {
      synTokens.push("_");
    }
  }
  return { tokens: synTokens, displayWords: dispTokens };
}

function tokensToIndices(tokens, vocabMap, unkIdx) {
  var out = [];
  for (var i = 0; i < tokens.length; i++) {
    var idx = vocabMap[tokens[i]];
    out.push(idx !== undefined ? idx : unkIdx);
  }
  return out;
}

function chunkIndices(indices, maxLen, overlap) {
  var step = maxLen - overlap;
  var chunks = [];
  var starts = [];
  if (indices.length <= maxLen) {
    chunks.push(indices.slice());
    starts.push(0);
  } else {
    for (var i = 0; i <= indices.length - maxLen; i += step) {
      chunks.push(indices.slice(i, i + maxLen));
      starts.push(i);
    }
    var tailStart = indices.length - maxLen;
    if (starts[starts.length - 1] !== tailStart) {
      chunks.push(indices.slice(tailStart));
      starts.push(tailStart);
    }
  }
  return { chunks: chunks, starts: starts };
}

function mapChunkProbsToWords(chunkProbs, starts, chunks, numWords) {
  var sum = new Float32Array(numWords);
  var count = new Float32Array(numWords);
  for (var ci = 0; ci < chunks.length; ci++) {
    for (var j = 0; j < chunks[ci].length; j++) {
      var wi = starts[ci] + j;
      if (wi < numWords) {
        sum[wi] += chunkProbs[ci];
        count[wi] += 1;
      }
    }
  }
  var out = [];
  for (var i = 0; i < numWords; i++) {
    out.push(count[i] > 0 ? sum[i] / count[i] : 0);
  }
  return out;
}
