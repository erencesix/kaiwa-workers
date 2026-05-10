/**
 * Kaiwa Cloudflare Worker — v3
 *
 * Routes:
 *   POST /transcribe   Upload audio chunk → returns {jobId, status:'processing'} instantly
 *   GET  /transcribe?jobId=xxx  Poll for result → returns {status:'done', result} or {status:'processing'}
 *   POST /analyze      JSON body → streams SSE back (GPT-4o)
 *
 * How transcription works (no timeout issues):
 *   1. Browser POSTs audio → worker uploads to OpenAI Files API (fast, just a file upload)
 *   2. Worker kicks off a transcription job and returns a jobId immediately
 *   3. Browser polls GET /transcribe?jobId=xxx every 3s
 *   4. Worker checks job status and returns result when ready
 *
 * This means the worker never has to wait for Whisper — it just stores/retrieves
 * state in a KV store (Cloudflare KV, included free).
 *
 * API key lives in Cloudflare secrets — never exposed to the browser.
 */

const ALLOWED_ORIGIN = "https://kaiwa-f53.pages.dev";
const MAX_BYTES = 24 * 1024 * 1024; // 24MB — Whisper's actual limit

const ALLOWED_TYPES = new Set([
  "audio/mpeg","audio/mp3","audio/mp4","audio/m4a","audio/x-m4a",
  "audio/wav","audio/wave","audio/ogg","audio/webm","audio/flac",
  "video/mp4","video/webm","video/quicktime","application/octet-stream",
]);

// ── CORS ──────────────────────────────────────────────────────────────────────
function cors(req) {
  const origin = req?.headers?.get("origin") || "";
  const ok = origin === ALLOWED_ORIGIN
    || origin.endsWith(".pages.dev")
    || origin.endsWith(".workers.dev")
    || origin.startsWith("http://localhost");
  return {
    "Access-Control-Allow-Origin": ok ? origin : ALLOWED_ORIGIN,
    "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
}

function jsonResp(data, status = 200, req = null) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...cors(req) },
  });
}

function jsonErr(msg, status = 500, req = null) {
  return jsonResp({ error: msg }, status, req);
}

// ── Unique job ID ─────────────────────────────────────────────────────────────
function makeId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

// ── POST /transcribe ──────────────────────────────────────────────────────────
// Receives audio chunk, uploads to OpenAI Files API, starts transcription,
// stores job in KV, returns jobId immediately (well under 30s CPU limit).
async function handleTranscribeUpload(req, env, ctx) {
  const cl = parseInt(req.headers.get("content-length") || "0", 10);
  if (cl > MAX_BYTES) return jsonErr(`Chunk too large (${(cl/1048576).toFixed(1)} MB). Max 24 MB.`, 413, req);

  let fd;
  try { fd = await req.formData(); }
  catch (e) { return jsonErr("Failed to parse upload: " + e.message, 400, req); }

  const file = fd.get("file");
  if (!file || typeof file === "string") return jsonErr("No audio file received.", 400, req);

  const mime = (file.type || "").toLowerCase();
  if (mime && !ALLOWED_TYPES.has(mime)) return jsonErr(`Unsupported type: ${mime}`, 415, req);
  if (file.size > MAX_BYTES) return jsonErr(`Chunk too large (${(file.size/1048576).toFixed(1)} MB). Max 24 MB.`, 413, req);

  // Step 1: Upload to OpenAI Files API
  // This is fast — just a file upload, no transcription yet
  const uploadForm = new FormData();
  uploadForm.append("file", file);
  uploadForm.append("purpose", "assistants"); // required by Files API

  let fileId;
  try {
    const uploadRes = await fetch("https://api.openai.com/v1/files", {
      method: "POST",
      headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}` },
      body: uploadForm,
    });
    const uploadData = await uploadRes.json();
    if (!uploadRes.ok) throw new Error(uploadData.error?.message || `Upload failed (${uploadRes.status})`);
    fileId = uploadData.id;
  } catch (e) {
    return jsonErr("File upload failed: " + e.message, 502, req);
  }

  // Step 2: Start transcription job using the file ID
  // We use the standard transcriptions endpoint with the file_id
  const jobId = makeId();

  // Use ctx.waitUntil to run transcription in background AFTER response is sent
  // This is the correct Cloudflare Workers pattern for background work
  ctx.waitUntil((async () => {
    try {
      // Store job as "processing" in KV immediately
      await env.KAIWA_KV.put(`job:${jobId}`, JSON.stringify({ status: "processing" }), { expirationTtl: 3600 });

      // Download the file we just uploaded (we need to re-upload to transcriptions endpoint)
      // because OpenAI's transcription API doesn't support file_id directly in whisper-1
      const fileRes = await fetch(`https://api.openai.com/v1/files/${fileId}/content`, {
        headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}` },
      });
      if (!fileRes.ok) throw new Error(`Failed to retrieve uploaded file (${fileRes.status})`);

      const fileBlob = await fileRes.blob();
      const filename = file.name || "audio.mp3";
      const fileMime = file.type || "audio/mpeg";

      // Now transcribe
      const transcribeForm = new FormData();
      transcribeForm.append("file", new File([fileBlob], filename, { type: fileMime }));
      transcribeForm.append("model", "whisper-1");
      transcribeForm.append("response_format", "verbose_json");
      transcribeForm.append("timestamp_granularities[]", "segment");

      const transcribeRes = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}` },
        body: transcribeForm,
      });

      const raw = await transcribeRes.text();
      if (!transcribeRes.ok) {
        let msg = `Whisper error (${transcribeRes.status})`;
        try { msg = JSON.parse(raw).error?.message || msg; } catch {}
        await env.KAIWA_KV.put(`job:${jobId}`, JSON.stringify({ status: "error", error: msg }), { expirationTtl: 3600 });
        return;
      }

      const result = JSON.parse(raw);
      await env.KAIWA_KV.put(`job:${jobId}`, JSON.stringify({ status: "done", result }), { expirationTtl: 3600 });

    } catch (e) {
      await env.KAIWA_KV.put(`job:${jobId}`, JSON.stringify({ status: "error", error: e.message }), { expirationTtl: 3600 });
    } finally {
      // Clean up the uploaded file from OpenAI to avoid storage costs
      try {
        await fetch(`https://api.openai.com/v1/files/${fileId}`, {
          method: "DELETE",
          headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}` },
        });
      } catch {}
    }
  })());

  // Return jobId immediately — browser will poll for result
  return jsonResp({ jobId, status: "processing" }, 202, req);
}

// ── GET /transcribe?jobId=xxx ─────────────────────────────────────────────────
async function handleTranscribePoll(req, env) {
  const url = new URL(req.url);
  const jobId = url.searchParams.get("jobId");
  if (!jobId) return jsonErr("Missing jobId.", 400, req);

  const raw = await env.KAIWA_KV.get(`job:${jobId}`);
  if (!raw) return jsonResp({ status: "processing" }, 200, req); // not ready yet

  const job = JSON.parse(raw);
  return jsonResp(job, 200, req);
}

// ── POST /analyze ─────────────────────────────────────────────────────────────
// Streams GPT-4o response via SSE using ctx.waitUntil for background streaming
async function handleAnalyze(req, env, ctx) {
  let transcript, segments, language, detectedLanguage;
  try {
    const b = await req.json();
    transcript = b.transcript; segments = b.segments || [];
    language = b.language || "id"; detectedLanguage = b.detectedLanguage || null;
  } catch (e) { return jsonErr("Invalid request: " + e.message, 400, req); }

  if (!transcript?.trim()) return jsonErr("Transcript is empty.", 400, req);

  const outputLangLabel =
    language === "en"   ? "English" :
    language === "both" ? "Indonesian and English (write every field in BOTH languages separated by a slash, e.g. 'Rapat dimulai / Meeting started')" :
    "Indonesian (Bahasa Indonesia)";

  const srcRaw   = (detectedLanguage || "").trim().toLowerCase();
  const srcLang  = srcRaw || "the original language";
  const srcLabel = srcLang === "the original language" ? srcLang
    : srcLang.charAt(0).toUpperCase() + srcLang.slice(1);

  const originalTranscript = segments.length > 0
    ? segments.map(s => `[${s.startFormatted}] [Speaker]: ${s.text}`).join("\n")
    : transcript;

  const segBlock = segments.length > 0
    ? segments.map(s => `[${s.startFormatted}] ${s.text}`).join("\n")
    : transcript;
  const capped = segBlock.length > 12000
    ? segBlock.substring(0, 12000) + "\n[... truncated ...]"
    : segBlock;

  const systemPrompt = `You are a meeting analyst and translator. Analyze this transcript and return a JSON object.

Source language: ${srcLabel}
Output language: ${outputLangLabel}

LANGUAGE RULES — ABSOLUTE, NO EXCEPTIONS:
- Write ALL output fields in ${outputLangLabel}. Every field. No exceptions.
- summary, subPoints, keyPoints, highlights, chapter titles, chapter summaries, speaker summaries, transcript lines — ALL in output language.
- Do NOT default to English unless output language is English.
- Do NOT mix languages unless output language explicitly says "both".

CONTENT RULES:
- CHAPTERS: Time blocks by topic. Title = actual topic. No decisions here.
- KEY POINTS: Explicit decisions/commitments/action items only. WHO + WHAT.
- HIGHLIGHTS: 2-4 surprising or decisive quotes. Translate to output language. Explain why each matters.
- SUMMARY: Past-tense narrative. Specific names/numbers. No repeating key points.
- TRANSCRIPT: Full translation of every utterance. Format: "[M:SS] [Speaker]: text"

Speaker detection: real names if mentioned, else Speaker A/B/C. Consistent across all fields.
Translation: natural and contextual. Match formality level.

CRITICAL: Return ONLY valid JSON. No markdown. No code fences. Nothing outside the JSON.

{
  "speakers": [{"id":"speaker_a","label":"Speaker A","name":null,"role":null,"summary":"<output lang>"}],
  "chapters": [{"title":"<output lang>","timestamp":"0:00 - 2:30","summary":"<output lang>"}],
  "tabs": {
    "summary": [{"point":"<output lang>","subPoints":["<output lang>"]}],
    "keyPoints": [{"point":"<output lang>","subPoints":["<output lang>"]}],
    "highlights": [{"speaker":"Speaker A","quote":"<translated>","context":"<output lang>"}]
  },
  "transcripts": {
    "translated": "[0:00] [Speaker A]: <output lang>\\n[0:05] [Speaker B]: <output lang>"
  }
}`;

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();

  const send = async (obj) => {
    try { await writer.write(enc.encode("data: " + JSON.stringify(obj) + "\n\n")); }
    catch (_) {}
  };

  ctx.waitUntil((async () => {
    // Heartbeat every 5s to keep the browser connection alive
    const heartbeat = setInterval(() => send({ ping: true }), 5000);
    try {
      const gpt = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${env.OPENAI_API_KEY}` },
        body: JSON.stringify({
          model: "gpt-4o", max_tokens: 8192, temperature: 0.1, stream: true,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: `Timestamped transcript:\n${capped}` },
          ],
        }),
      });

      if (!gpt.ok) {
        const t = await gpt.text();
        let msg = `OpenAI error (${gpt.status})`;
        try { msg = JSON.parse(t).error?.message || msg; } catch {}
        await send({ error: msg }); return;
      }

      let acc = "", lineBuf = "";
      const reader = gpt.body.getReader();
      const dec = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        lineBuf += dec.decode(value, { stream: true });
        let nl;
        while ((nl = lineBuf.indexOf("\n")) !== -1) {
          const line = lineBuf.slice(0, nl).trimEnd();
          lineBuf = lineBuf.slice(nl + 1);
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") continue;
          let p; try { p = JSON.parse(raw); } catch { continue; }
          const token = p.choices?.[0]?.delta?.content;
          if (token) { acc += token; await send({ token }); }
        }
      }

      const clean = acc.replace(/^```json\s*/m,"").replace(/^```\s*/m,"").replace(/\s*```$/m,"").trim();
      const s = clean.indexOf("{"), e = clean.lastIndexOf("}");
      const jsonStr = s !== -1 && e !== -1 ? clean.slice(s, e+1) : clean;

      let result;
      try { result = JSON.parse(jsonStr); }
      catch { await send({ error: "Failed to parse analysis. Try a shorter recording. Raw: " + clean.substring(0, 200) }); return; }

      if (!result.transcripts) result.transcripts = {};
      result.transcripts.translated = result.transcripts.translated || "";
      result.transcripts.original   = originalTranscript;
      result._meta = { sourceLang: srcLabel, outputLang: language };

      await send({ done: true, result });
    } catch (e) {
      await send({ error: "Analysis failed: " + e.message });
    } finally {
      clearInterval(heartbeat);
      try { await writer.close(); } catch (_) {}
    }
  })());

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
      ...cors(req),
    },
  });
}

// ── Router ────────────────────────────────────────────────────────────────────
export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors(req) });
    }
    if (!env.OPENAI_API_KEY) return jsonErr("OPENAI_API_KEY not configured.", 500, req);
    if (!env.KAIWA_KV)       return jsonErr("KAIWA_KV binding not configured. Add KV namespace in Cloudflare dashboard.", 500, req);

    if (url.pathname === "/transcribe") {
      if (req.method === "GET")  return handleTranscribePoll(req, env);
      if (req.method === "POST") return handleTranscribeUpload(req, env, ctx);
    }
    if (url.pathname === "/analyze" && req.method === "POST") return handleAnalyze(req, env, ctx);

    return jsonErr("Not found.", 404, req);
  }
};
