/**
 * Kaiwa Cloudflare Worker — v4
 *
 * Routes:
 *   POST /transcribe   Upload audio → starts Whisper in background via ctx.waitUntil
 *                      Returns {jobId} instantly
 *   GET  /transcribe?jobId=xxx  Poll for result
 *   POST /analyze      Streams GPT-4o analysis via SSE
 *
 * ctx.waitUntil() keeps the worker alive after the response is sent,
 * allowing Whisper to finish without any timeout on the response.
 */

const ALLOWED_ORIGIN = "https://kaiwa-f53.pages.dev";
const MAX_BYTES = 24 * 1024 * 1024;

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

function makeId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

// ── POST /transcribe ──────────────────────────────────────────────────────────
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

  const jobId = makeId();

  // Store as processing immediately so polls don't return 404
  await env.KAIWA_KV.put(`job:${jobId}`, JSON.stringify({ status: "processing" }), { expirationTtl: 3600 });

  // Read file into memory BEFORE sending response — after response is sent
  // the request body may no longer be readable
  const fileBuffer = await file.arrayBuffer();
  const fileName = file.name || "audio.mp3";
  const fileMime = file.type || "audio/mpeg";

  // Run Whisper in background — ctx.waitUntil keeps worker alive until done
  ctx.waitUntil((async () => {
    try {
      const form = new FormData();
      form.append("file", new File([fileBuffer], fileName, { type: fileMime }));
      form.append("model", "whisper-1");
      form.append("response_format", "verbose_json");
      form.append("timestamp_granularities[]", "segment");

      const res = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}` },
        body: form,
      });

      const raw = await res.text();

      if (!res.ok) {
        let msg = `Whisper error (${res.status})`;
        try { msg = JSON.parse(raw).error?.message || msg; } catch {}
        await env.KAIWA_KV.put(`job:${jobId}`, JSON.stringify({ status: "error", error: msg }), { expirationTtl: 3600 });
        return;
      }

      let result;
      try { result = JSON.parse(raw); }
      catch { await env.KAIWA_KV.put(`job:${jobId}`, JSON.stringify({ status: "error", error: "Unexpected Whisper response: " + raw.substring(0, 200) }), { expirationTtl: 3600 }); return; }

      await env.KAIWA_KV.put(`job:${jobId}`, JSON.stringify({ status: "done", result }), { expirationTtl: 3600 });

    } catch (e) {
      await env.KAIWA_KV.put(`job:${jobId}`, JSON.stringify({ status: "error", error: e.message }), { expirationTtl: 3600 });
    }
  })());

  return jsonResp({ jobId, status: "processing" }, 202, req);
}

// ── GET /transcribe?jobId=xxx ─────────────────────────────────────────────────
async function handleTranscribePoll(req, env) {
  const url = new URL(req.url);
  const jobId = url.searchParams.get("jobId");
  if (!jobId) return jsonErr("Missing jobId.", 400, req);

  const raw = await env.KAIWA_KV.get(`job:${jobId}`);
  if (!raw) return jsonResp({ status: "processing" }, 200, req);

  return jsonResp(JSON.parse(raw), 200, req);
}

// ── POST /analyze ─────────────────────────────────────────────────────────────
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
    try { await writer.write(enc.encode("data: " + JSON.stringify(obj) + "\n\n")); } catch (_) {}
  };

  ctx.waitUntil((async () => {
    const hb = setInterval(() => send({ ping: true }), 5000);
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
      catch { await send({ error: "Failed to parse analysis. Raw: " + clean.substring(0,200) }); return; }

      if (!result.transcripts) result.transcripts = {};
      result.transcripts.translated = result.transcripts.translated || "";
      result.transcripts.original   = originalTranscript;
      result._meta = { sourceLang: srcLabel, outputLang: language };

      await send({ done: true, result });
    } catch (e) {
      await send({ error: "Analysis failed: " + e.message });
    } finally {
      clearInterval(hb);
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
    if (!env.KAIWA_KV)       return jsonErr("KAIWA_KV binding not configured.", 500, req);

    if (url.pathname === "/transcribe") {
      if (req.method === "GET")  return handleTranscribePoll(req, env);
      if (req.method === "POST") return handleTranscribeUpload(req, env, ctx);
    }
    if (url.pathname === "/analyze" && req.method === "POST") return handleAnalyze(req, env, ctx);

    return jsonErr("Not found.", 404, req);
  }
};
