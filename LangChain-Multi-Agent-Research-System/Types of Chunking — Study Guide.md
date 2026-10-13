# Types of Chunking — Study Guide

Oct 13, 2026 · @Sridharan

## What chunking is and why it matters

Chunking splits a long document into smaller pieces (chunks) so each one can be embedded, indexed and retrieved on its own. It is the first design decision in a Retrieval-Augmented Generation (RAG) pipeline, and it caps how good retrieval can ever be.

**Why we chunk**

- **Model limits:** embedding models and LLMs have a maximum input length (context window).
- **Retrieval precision:** a small, focused chunk produces an embedding that represents one idea; a huge chunk blurs many ideas into one vector.
- **Cost and latency:** sending only the relevant chunks to the LLM uses fewer tokens.

**The core trade-off**

- Chunks too **small** → lose context ("it" with no idea what "it" refers to).
- Chunks too **large** → noisy embeddings and irrelevant text sent to the model.

Where chunking sits in RAG:

1. Load documents
2. **Chunk** them
3. Embed each chunk into a vector
4. Store vectors in a vector database
5. At query time, embed the question and retrieve the most similar chunks
6. Pass those chunks to the LLM to generate the answer

&#91;embedded content: RAG pipeline · the 11 chunking types in 4 families\]

The chunk step feeds everything after it; the strategies below fall into four families by what they split on.

## 1. Fixed-size chunking

Split text every N characters or N tokens, usually with an **overlap** so a sentence cut at a boundary still appears whole in one of the two chunks.

- **Example:** chunk size 500 tokens, overlap 50 tokens → chunk 1 = tokens 0–499, chunk 2 = tokens 450–949.
- **Pros:** simplest, fast, predictable chunk sizes, easy to budget tokens.
- **Cons:** ignores meaning; can cut mid-sentence or mid-word; one chunk may mix two topics.
- **Use when:** you need a quick baseline or the text has little structure (logs, transcripts).

## 2. Recursive character chunking

Split using a ranked list of separators, trying the largest natural break first and falling back only when a piece is still too big. A typical order is: paragraph (`\n\n`) → line (`\n`) → sentence (` .  `) → word (space) → character.

- **Example:** LangChain's `RecursiveCharacterTextSplitter`.
- **Pros:** respects natural boundaries most of the time while still hitting a target size; a strong general-purpose default.
- **Cons:** still size-driven, not meaning-driven; separators must suit the content.
- **Use when:** general prose, articles, reports — the usual starting point for RAG.

## 3. Sentence-based chunking

Split into sentences with an NLP tokenizer (NLTK, spaCy), then group a fixed number of sentences (e.g. 3–5) per chunk.

- **Pros:** every chunk is grammatically complete.
- **Cons:** sentence lengths vary widely, so chunk sizes are uneven; one sentence may lack context.
- **Use when:** Q&A over FAQs, short factual text.

## 4. Paragraph-based chunking

Treat each paragraph as a chunk, merging short ones and splitting very long ones.

- **Pros:** paragraphs usually hold one idea, so chunks are coherent.
- **Cons:** paragraph length is unpredictable.
- **Use when:** well-written articles, essays, books.

## 5. Document-structure (format-aware) chunking

Split along the document's own structure instead of raw text.

- **Markdown / HTML:** split on headings (`#`, `##`, `<h2>`) and keep the heading path as metadata, e.g. *Guide > Install > Windows*.
- **Code:** split on functions, classes or methods using the language's syntax tree.
- **PDF / Word:** split by sections, pages, tables and figures; keep tables whole.
- **Pros:** chunks match how authors organised ideas; headings add useful metadata for filtering.
- **Cons:** needs a parser per format; messy or scanned files have weak structure.
- **Use when:** technical docs, wikis, codebases, legal contracts, manuals.

## 6. Semantic chunking

Split where the **meaning** changes. Embed each sentence, compare neighbouring sentences, and start a new chunk wherever similarity drops below a threshold (a topic shift).

- **Pros:** each chunk stays on one topic, even in text with no headings.
- **Cons:** needs an embedding call per sentence (slower, costlier); threshold needs tuning; chunk sizes vary.
- **Use when:** long unstructured text that mixes topics (transcripts, reports).

## 7. Agentic / LLM-based chunking

An LLM reads the text and decides the boundaries itself, often grouping "propositions" (small self-contained facts) into topics.

- **Pros:** closest to how a human would split; handles complex content.
- **Cons:** the slowest and most expensive option; results can vary between runs.
- **Use when:** high-value, low-volume documents where quality matters more than cost.

## 8. Hierarchical (parent–child) chunking

Create **small child chunks** for precise search and link each to a **larger parent chunk** (section or page). Search runs on children; the parent is what gets sent to the LLM.

- **Pros:** gets both precise matching and full context.
- **Cons:** more storage and more complex indexing.
- **Also called:** small-to-big retrieval, parent document retrieval.

## 9. Sliding window chunking

A fixed window moves through the text in steps smaller than the window, so every chunk heavily overlaps its neighbours.

- **Pros:** no idea is ever split without also appearing whole somewhere.
- **Cons:** many redundant chunks; more storage and duplicate results.

## 10. Late chunking

Embed the **whole document first** with a long-context embedding model, then split the token embeddings into chunks and pool each one. Every chunk's vector therefore "knows" the surrounding document.

- **Pros:** fixes lost references ("the company" still links to its name from earlier).
- **Cons:** needs a long-context embedding model.

## 11. Contextual chunking (contextual retrieval)

Before embedding, an LLM writes a short sentence of context for each chunk (e.g. "This chunk is from Acme's Q2 2025 report, revenue section") and prepends it.

- **Pros:** big gains in retrieval accuracy for chunks that are ambiguous on their own.
- **Cons:** one LLM call per chunk at indexing time (prompt caching reduces the cost).

## Comparison and how to choose

| Type | Splits by | Cost / speed | Best for |
| --- | --- | --- | --- |
| Fixed-size | Token or character count | Cheapest, fastest | Baselines, logs, unstructured text |
| Recursive | Separators, largest first | Cheap, fast | General default for prose |
| Sentence | Sentence boundaries | Cheap | FAQs, short factual text |
| Paragraph | Paragraph breaks | Cheap | Articles, essays, books |
| Document-structure | Headings, code syntax, sections | Moderate (parsers) | Docs, wikis, code, contracts |
| Semantic | Drop in meaning similarity | Higher (embeds every sentence) | Mixed-topic unstructured text |
| Agentic / LLM | LLM judgement | Highest | Small, high-value collections |
| Hierarchical | Small children + large parents | Moderate (more storage) | Precision plus full context |
| Sliding window | Overlapping fixed windows | Moderate (redundant chunks) | Text where no boundary may be lost |
| Late chunking | Pooled whole-doc token embeddings | Moderate (long-context model) | Docs full of cross-references |
| Contextual | Any split + LLM-written context | Higher at indexing | Ambiguous chunks, enterprise RAG |

**Rules of thumb**

1. Start with **recursive chunking**, about 256–512 tokens with 10–20% overlap.
2. If the content has clear structure (Markdown, code, contracts), switch to **document-structure** chunking.
3. If answers need wider context, add **hierarchical** (parent–child) retrieval.
4. If retrieval still misses, try **semantic**, **contextual** or **late** chunking.
5. Always **measure**: build a small set of test questions and compare retrieval hit rate across settings.

**Sizing tips**

- Smaller chunks (128–256 tokens) suit precise fact lookup.
- Larger chunks (512–1,024 tokens) suit summaries and reasoning across ideas.
- Store metadata with each chunk (source, page, section heading) for filtering and citations.

## Node.js examples

Plain JavaScript, no dependencies unless noted; save as `chunking.mjs` and run with Node 18+. Sizes are in characters for simplicity (roughly 4 characters per token in English).

**Fixed-size with overlap**

```javascript
export function fixedSizeChunks(text, size = 500, overlap = 50) {
  if (overlap >= size) throw new Error('overlap must be smaller than size');
  const chunks = [];
  for (let start = 0; start < text.length; start += size - overlap) {
    chunks.push(text.slice(start, start + size));
    if (start + size >= text.length) break;
  }
  return chunks;
}
```

**Recursive (largest separator first)**

```javascript
export function recursiveChunks(text, size = 500, separators = ['\n\n', '\n', '. ', ' ', '']) {
  if (text.length <= size) return [text];
  const [sep, ...rest] = separators;
  if (sep === '') return fixedSizeChunks(text, size, 0); // last resort: hard cut
  const chunks = [];
  let current = '';
  for (const part of text.split(sep)) {
    const candidate = current ? current + sep + part : part;
    if (candidate.length <= size) { current = candidate; continue; }
    if (current) chunks.push(current);
    if (part.length > size) {               // piece still too big: try the next separator
      chunks.push(...recursiveChunks(part, size, rest));
      current = '';
    } else current = part;
  }
  if (current) chunks.push(current);
  return chunks;
}
```

**Sentence-based (built-in `Intl.Segmenter`)**

```javascript
const splitSentences = (text) =>
  [...new Intl.Segmenter('en', { granularity: 'sentence' }).segment(text)]
    .map((s) => s.segment.trim())
    .filter(Boolean);

export function sentenceChunks(text, sentencesPerChunk = 3) {
  const sentences = splitSentences(text);
  const chunks = [];
  for (let i = 0; i < sentences.length; i += sentencesPerChunk)
    chunks.push(sentences.slice(i, i + sentencesPerChunk).join(' '));
  return chunks;
}
```

**Markdown-aware (heading path kept as metadata)**

```javascript
export function markdownChunks(md) {
  const chunks = [];
  const path = [];
  let buf = [];
  const flush = () => {
    const text = buf.join('\n').trim();
    if (text) chunks.push({ text, metadata: { headings: path.filter(Boolean).join(' > ') } });
    buf = [];
  };
  for (const line of md.split('\n')) {
    const m = line.match(/^(#{1,6})\s+(.*)/);
    if (m) {
      flush();
      const level = m[1].length;
      path.length = level - 1;          // drop deeper headings
      path[level - 1] = m[2].trim();
    } else buf.push(line);
  }
  flush();
  return chunks;
}
// markdownChunks('# Guide\n## Install\nsteps')
// → [{ text: 'steps', metadata: { headings: 'Guide > Install' } }]
```

**Semantic (split where meaning shifts)** — needs an embedding model; this uses the OpenAI SDK (`npm i openai`, set `OPENAI_API_KEY`), but any `embed(texts) → vectors` function works.

```javascript
import OpenAI from 'openai';
const openai = new OpenAI();

const embed = async (texts) => {
  const res = await openai.embeddings.create({ model: 'text-embedding-3-small', input: texts });
  return res.data.map((d) => d.embedding);
};

const cosine = (a, b) => {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
};

export async function semanticChunks(text, threshold = 0.75) {
  const sentences = splitSentences(text);
  if (sentences.length === 0) return [];
  const vectors = await embed(sentences);
  const chunks = [];
  let current = [sentences[0]];
  for (let i = 1; i < sentences.length; i++) {
    if (cosine(vectors[i - 1], vectors[i]) < threshold) {  // topic shift
      chunks.push(current.join(' '));
      current = [];
    }
    current.push(sentences[i]);
  }
  chunks.push(current.join(' '));
  return chunks;
}
```

**Parent–child (search small, answer with big)**

```javascript
export function parentChildChunks(text, parentSize = 2000, childSize = 400) {
  const parents = recursiveChunks(text, parentSize);
  const children = [];
  parents.forEach((parent, parentId) => {
    for (const child of recursiveChunks(parent, childSize)) children.push({ text: child, parentId });
  });
  // Embed and search `children`; send parents[hit.parentId] to the LLM.
  return { parents, children };
}
```

**Using a library: LangChain.js** (`npm i @langchain/textsplitters`)

```javascript
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
const chunks = await splitter.splitText(longText);

// Code-aware and Markdown-aware presets
const jsSplitter = RecursiveCharacterTextSplitter.fromLanguage('js', { chunkSize: 500, chunkOverlap: 0 });
const mdSplitter = RecursiveCharacterTextSplitter.fromLanguage('markdown', { chunkSize: 500, chunkOverlap: 50 });
```

## Bonus: chunking in psychology and learning

In cognitive psychology, chunking means grouping small pieces of information into larger meaningful units so working memory can hold more. George Miller (1956) suggested people hold about 7 ± 2 items; later research by Nelson Cowan puts it nearer 4 chunks.

- **Number chunking:** 9715550123 → 971-555-0123.
- **Category chunking:** a shopping list grouped into dairy, fruit, bakery.
- **Acronyms and mnemonics:** ROYGBIV for the colours of the rainbow.
- **Pattern chunking:** chess experts see board positions as familiar patterns, not 32 separate pieces.
- **Concept chunking:** learn a topic in small sections, master each, then link them — the way this guide is laid out.

## Key terms

| Term | Meaning |
| --- | --- |
| Chunk | A piece of a document indexed and retrieved on its own |
| Chunk size | Length of a chunk, in tokens or characters |
| Overlap | Text shared between neighbouring chunks to protect boundaries |
| Embedding | A vector of numbers representing a chunk's meaning |
| Vector database | Store that finds the chunks most similar to a query |
| Separator | Character pattern used as a split point (`\n\n`, ` .  `) |
| Metadata | Extra fields on a chunk: source, page, heading |

## Self-check quiz

1. Why is overlap used in fixed-size chunking?
2. In recursive chunking, which separator is tried first?
3. What signal does semantic chunking use to decide a boundary?
4. In parent–child chunking, which chunk is searched and which is sent to the LLM?
5. How does late chunking differ from normal chunking?
6. Name one drawback of agentic chunking.
7. Which strategy would you choose for a Python codebase, and why?
8. What is the main risk of chunks that are too large?

**Answers**

1. So a sentence or idea cut at a boundary still appears whole in at least one chunk.
2. The largest natural break, usually the paragraph break (`\n\n`).
3. A drop in embedding similarity between neighbouring sentences.
4. Small child chunks are searched; the larger parent chunk is sent to the LLM.
5. It embeds the whole document first, then splits and pools, so each chunk keeps document-wide context.
6. High cost and latency (also: results can vary between runs).
7. Document-structure (code-aware) chunking, splitting on functions and classes so each chunk is a complete unit of logic.
8. The embedding blurs several ideas, lowering retrieval precision and sending irrelevant text to the LLM.
