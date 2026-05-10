/**
 * Kaiwa Cloudflare Worker
 * POST /transcribe  — multipart/form-data with "file"
 * POST /analyze     — JSON body
 *
 * Both stream SSE back to the browser.
 * API key never leaves this worker.
 */

const ALLOWED_ORIGIN = "https://kaiwa-f53.pages.dev";
const MAX_BYTES = 24 * 1024 * 1024;

const ALLOWED_TYPES = new Set([
  "audio/mpeg","audio/mp3","audio/mp4","audio/m4a","audio/x-m4a",
  "audio/wav","audio/wave","audio/ogg","audio/webm","audio/flac",
  "video/mp4","video/webm","video/quicktime","application/octet-stream",
]);

// ── CORS ──────────────────────────────────────────────────────────────────────
function corsHeaders(request) {
  const origin = request?.headers?.get("origin") || "";
  const ok = origin === ALLOWED_ORIGIN
    || origin.endsWith(".pages.dev")
    || origin.endsWith(".workers.dev")
    || origin.startsWith("http://localhost");
  return {
    "Access-Control-Allow-Origin": ok ? origin : ALLOWED_ORIGIN,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
}

function jsonErr(msg, status, req) {
  return new Response(JSON.stringify({ error: msg }), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders(req) },
  });
}

// ── SSE stream factory ────────────────────────────────────────────────────────
// Cloudflare Workers keep a TransformStream open for as long as the writer
// is not closed — no wall-clock kill during streaming.
function makeStream(req) {
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();
  const send = async (obj) => {
    try { await writer.write(enc.encode("data: " + JSON.stringify(obj) + "\n\n")); }
    catch (_) {}
  };
  const close = async () => { try { await writer.close(); } catch (_) {} };
  const response = new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
      ...corsHeaders(req),
    },
  });
  return { send, close, response };
}

function startHeartbeat(send, ms = 4000) {
  const id = setInterval(() => send({ ping: true }), ms);
  return () => clearInterval(id);
}

// ── /transcribe ───────────────────────────────────────────────────────────────
async function handleTranscribe(req, env) {
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

  const { send, close, response } = makeStream(req);

  (async () => {
    const stopPing = startHeartbeat(send);
    try {
      const out = new FormData();
      out.append("file", file);
      out.append("model", "whisper-1");
      out.append("response_format", "verbose_json");
      out.append("timestamp_granularities[]", "segment");

      const res = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}` },
        body: out,
      });

      const raw = await res.text();
      if (!res.ok) {
        let msg = `Whisper error (${res.status})`;
        try { msg = JSON.parse(raw).error?.message || msg; } catch {}
        await send({ error: msg }); return;
      }

      let result;
      try { result = JSON.parse(raw); }
      catch { await send({ error: "Whisper returned unexpected response: " + raw.substring(0,200) }); return; }

      await send({ done: true, result });
    } catch (e) {
      await send({ error: "Transcription failed: " + e.message });
    } finally {
      stopPing();
      await close();
    }
  })();

  return response;
}

// ── /analyze ──────────────────────────────────────────────────────────────────
async function handleAnalyze(req, env) {
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

  const { send, close, response } = makeStream(req);

  (async () => {
    const stopPing = startHeartbeat(send);
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
      stopPing();
      await close();
    }
  })();

  return response;
}

// ── Router ────────────────────────────────────────────────────────────────────
export default {
  async fetch(req, env) {
    const url = new URL(req.url);

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(req) });
    }
    if (req.method !== "POST") return jsonErr("Method Not Allowed", 405, req);
    if (!env.OPENAI_API_KEY)   return jsonErr("OPENAI_API_KEY not configured.", 500, req);

    if (url.pathname === "/transcribe") return handleTranscribe(req, env);
    if (url.pathname === "/analyze")    return handleAnalyze(req, env);

    return jsonErr("Not found.", 404, req);
  }
};
