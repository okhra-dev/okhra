const ASCII_PUNCT = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~";
const EXT_PUNCT   = "\u2013\u2014\u2026\u00AB\u00BB\u2039\u203A" +
                    "\u201C\u201D\u2018\u2019\u201E\u201A" +
                    "\u2010\u2011\u2012\u2015\u201F\u201B";
const PUNCT_SET   = new Set([...ASCII_PUNCT, ...EXT_PUNCT]);

function separatePunct(text) {
  let out = "";
  for (const ch of text) {
    out += PUNCT_SET.has(ch) ? ("  " + ch + "  ") : ch;
  }
  return out;
}

function tokenizeText(text, selectWords) {
  const lcWords  = separatePunct(text.toLowerCase()).split(/\s+/).filter(Boolean);
  const dispWords = separatePunct(text).split(/\s+/).filter(Boolean);

  const tokens = lcWords.map(t =>
    (PUNCT_SET.has(t) || selectWords.has(t)) ? t : "_"
  );
  return { tokens, displayWords: dispWords };
}

function tokensToIndices(tokens, vocabMap, unkIdx) {
  return tokens.map(t => (vocabMap[t] !== undefined ? vocabMap[t] : unkIdx));
}

function chunkIndices(indices, maxLen, overlap) {
  const step   = maxLen - overlap;
  const chunks = [];
  const starts = [];

  if (indices.length <= maxLen) {
    chunks.push(indices.slice());
    starts.push(0);
  } else {
    for (let i = 0; i <= indices.length - maxLen; i += step) {
      chunks.push(indices.slice(i, i + maxLen));
      starts.push(i);
    }
    const tailStart = indices.length - maxLen;
    if (starts[starts.length - 1] !== tailStart) {
      chunks.push(indices.slice(tailStart));
      starts.push(tailStart);
    }
  }
  return { chunks, starts };
}

function mapProbsToWords(probs, starts, chunks, numWords) {
  const sum   = new Float32Array(numWords);
  const count = new Float32Array(numWords);

  for (let ci = 0; ci < chunks.length; ci++) {
    for (let j = 0; j < chunks[ci].length; j++) {
      const wi = starts[ci] + j;
      if (wi < numWords) {
        sum[wi]   += probs[ci];
        count[wi] += 1;
      }
    }
  }
  return Array.from(sum, (s, i) => (count[i] > 0 ? s / count[i] : 0));
}
