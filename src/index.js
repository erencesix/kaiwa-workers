/**
 * Kaiwa Cloudflare Worker
 * Handles /transcribe and /analyze — API key never leaves this worker.
 *
 * Routes:
 *   POST /transcribe   multipart/form-data with "file" field
 *   POST /analyze      JSON body
 *
 * Both stream SSE back to the browser:
 *   data: {"ping":true}          — keepalive (ignore)
 *   data: {"done":true, ...}     — success
 *   data: {"error":"..."}        — failure
 */

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const JSON_HEADERS = {
  "Content-Type": "application/json",
  ...CORS_HEADERS,
};

const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  "X-Accel-Buffering": "no",
  ...CORS_HEADERS,
};

const MAX_BYTES = 24 * 1024 * 1024; // 24MB — Whisper's actual limit, no Netlify cap here

const ALLOWED_TYPES = new Set([
  "audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a", "audio/x-m4a",
  "audio/wav", "audio/wave", "audio/ogg", "audio/webm", "audio/flac",
  "video/mp4", "video/webm", "video/quicktime",
  "application/octet-stream",
]);

// ── Helpers ──────────────────────────────────────────────────────────────────

function jsonError(msg, status = 500) {
  return new Response(JSON.stringify({ error: msg }), { status, headers: JSON_HEADERS });
}

function sseStream(fn) {
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();

  const send = async (obj) => {
    try { await writer.write(enc.encode("data: " + JSON.stringify(obj) + "\n\n")); }
    catch (_) {}
  };

  const close = async () => {
    try { await writer.close(); } catch (_) {}
  };

  // Run handler in background — Cloudflare Workers keep the response stream
  // alive as long as there's an active TransformStream, no wall-clock kill.
  fn(send, close);

  return new Response(readable, { headers: SSE_HEADERS });
}

function heartbeat(send, intervalMs = 5000) {
  // Returns a cancel function. Sends pings so the browser knows we're alive.
  const id = setInterval(() => send({ ping: true }), intervalMs);
  return () => clearInterval(id);
}

// ── /transcribe ───────────────────────────────────────────────────────────────

async function handleTranscribe(request, env) {
  const contentLength = parseInt(request.headers.get("content-length") || "0", 10);
  if (contentLength > MAX_BYTES) {
    return jsonError(`File too large (${(contentLength / 1048576).toFixed(1)} MB). Max is 24 MB per chunk.`, 413);
  }

  let formData;
  try { formData = await request.formData(); }
  catch (e) { return jsonError("Failed to parse upload: " + e.message, 400); }

  const audioFile = formData.get("file");
  if (!audioFile || typeof audioFile === "string") {
    return jsonError("No audio file received.", 400);
  }

  const mime = (audioFile.type || "").toLowerCase();
  if (mime && !ALLOWED_TYPES.has(mime)) {
    return jsonError(`Unsupported file type: ${mime}`, 415);
  }

  if (audioFile.size > MAX_BYTES) {
    return jsonError(`File too large (${(audioFile.size / 1048576).toFixed(1)} MB). Max is 24 MB per chunk.`, 413);
  }

  return sseStream(async (send, close) => {
    const stopPing = heartbeat(send, 5000);
    try {
      const outForm = new FormData();
      outForm.append("file", audioFile);
      outForm.append("model", "whisper-1");
      outForm.append("response_format", "verbose_json");
      outForm.append("timestamp_granularities[]", "segment");

      // Cloudflare Workers have no wall-clock timeout on fetch() —
      // this will wait as long as Whisper needs, even for large files.
      const whisperRes = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}` },
        body: outForm,
      });

      const rawText = await whisperRes.text();

      if (!whisperRes.ok) {
        let errMsg = `Whisper error (${whisperRes.status})`;
        try { errMsg = JSON.parse(rawText).error?.message || errMsg; } catch {}
        await send({ error: errMsg });
        return;
      }

      let result;
      try { result = JSON.parse(rawText); }
      catch (e) { await send({ error: "Whisper returned unexpected response: " + rawText.substring(0, 200) }); return; }

      await send({ done: true, result });

    } catch (err) {
      await send({ error: "Transcription failed: " + err.message });
    } finally {
      stopPing();
      await close();
    }
  });
}

// ── /analyze ─────────────────────────────────────────────────────────────────

async function handleAnalyze(request, env) {
  let transcript, segments, language, detectedLanguage;
  try {
    const body = await request.json();
    transcript      = body.transcript;
    segments        = body.segments || [];
    language        = body.language || "id";
    detectedLanguage = body.detectedLanguage || null;
  } catch (e) {
    return jsonError("Invalid request: " + e.message, 400);
  }

  if (!transcript || !transcript.trim()) {
    return jsonError("Transcript is empty.", 400);
  }

  const outputLangLabel =
    language === "en"   ? "English" :
    language === "both" ? "Indonesian and English (write every field in BOTH languages separated by a slash, e.g. 'Rapat dimulai / Meeting started')" :
    "Indonesian (Bahasa Indonesia)";

  const sourceLangRaw   = (detectedLanguage || "").trim().toLowerCase();
  const sourceLang      = sourceLangRaw || "the original language";
  const sourceLangLabel = sourceLang === "the original language"
    ? "the original language"
    : sourceLang.charAt(0).toUpperCase() + sourceLang.slice(1);

  // Build original-language transcript from Whisper segments — verbatim, no GPT
  const originalTranscript = segments.length > 0
    ? segments.map(s => `[${s.startFormatted}] [Speaker]: ${s.text}`).join("\n")
    : transcript;

  const segmentBlock = segments.length > 0
    ? segments.map(s => `[${s.startFormatted}] ${s.text}`).join("\n")
    : transcript;
  const cappedSegments = segmentBlock.length > 12000
    ? segmentBlock.substring(0, 12000) + "\n[... truncated ...]"
    : segmentBlock;

  const systemPrompt = `You are a meeting analyst and translator. Analyze this transcript and return a JSON object.

Source language: ${sourceLangLabel}
Output language: ${outputLangLabel}

LANGUAGE RULES — ABSOLUTE, NO EXCEPTIONS:
- You MUST write ALL output fields in ${outputLangLabel}.
- This applies to EVERY field: summary points, subPoints, keyPoints, highlights quote, highlights context, chapter titles, chapter summaries, speaker summaries, transcript lines — everything.
- If output language is Indonesian, write in Indonesian. If English, write in English. If both, write "Indonesian / English" for each field.
- Do NOT default to English. Do NOT mix languages unless output language explicitly says "both".

CONTENT RULES — sections must be DISTINCT:
- CHAPTERS: Time blocks by topic. Title = actual topic discussed. No decisions here.
- KEY POINTS: Only explicit decisions, commitments, action items with WHO + WHAT.
- HIGHLIGHTS: 2-4 quotes that are surprising or decisive. Translate to output language. Explain why each matters.
- SUMMARY: Past-tense narrative with specific names/numbers. Do NOT repeat key points verbatim.
- TRANSCRIPT (translated): Full translation of every utterance in output language. Format: "[M:SS] [Speaker Name]: text"

Speaker detection: use real names if mentioned, else Speaker A/B/C. Be consistent across all fields.
Translation: natural and contextual, never literal. Match the formality level of the original.

CRITICAL: Return ONLY valid JSON. No markdown. No code fences. No text outside the JSON object.

{
  "speakers": [{"id":"speaker_a","label":"Speaker A","name":null,"role":null,"summary":"<in output language>"}],
  "chapters": [{"title":"<in output language>","timestamp":"0:00 - 2:30","summary":"<in output language>"}],
  "tabs": {
    "summary": [{"point":"<in output language>","subPoints":["<in output language>"]}],
    "keyPoints": [{"point":"<WHO> <WHAT> — in output language","subPoints":["<detail>"]}],
    "highlights": [{"speaker":"Speaker A","quote":"<translated to output language>","context":"<in output language>"}]
  },
  "transcripts": {
    "translated": "[0:00] [Speaker A]: <full translation in output language>\\n[0:05] [Speaker B]: <translation>"
  }
}`;

  return sseStream(async (send, close) => {
    const stopPing = heartbeat(send, 5000);
    try {
      const gptRes = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${env.OPENAI_API_KEY}` },
        body: JSON.stringify({
          model: "gpt-4o",
          max_tokens: 8192,
          temperature: 0.1,
          stream: true,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: `Timestamped transcript to analyze and translate:\n${cappedSegments}` },
          ],
        }),
      });

      if (!gptRes.ok) {
        const errText = await gptRes.text();
        let errMsg = `OpenAI error (${gptRes.status})`;
        try { errMsg = JSON.parse(errText).error?.message || errMsg; } catch {}
        await send({ error: errMsg });
        return;
      }

      let accumulated = "";
      let lineBuf = "";
      const reader = gptRes.body.getReader();
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
          let parsed;
          try { parsed = JSON.parse(raw); } catch { continue; }
          const token = parsed.choices?.[0]?.delta?.content;
          if (token) { accumulated += token; await send({ token }); }
        }
      }

      // Strip stray markdown fences
      const clean = accumulated
        .replace(/^```json\s*/m, "").replace(/^```\s*/m, "").replace(/\s*```$/m, "").trim();

      // Extract outermost JSON object defensively
      const jsonStart = clean.indexOf("{");
      const jsonEnd   = clean.lastIndexOf("}");
      const jsonStr   = jsonStart !== -1 && jsonEnd !== -1 ? clean.slice(jsonStart, jsonEnd + 1) : clean;

      let result;
      try { result = JSON.parse(jsonStr); }
      catch (e) {
        await send({ error: "Failed to parse analysis. Try a shorter recording. Raw: " + clean.substring(0, 200) });
        return;
      }

      // Attach original-language transcript built from Whisper segments
      if (!result.transcripts) result.transcripts = {};
      result.transcripts.translated = result.transcripts.translated || "";
      result.transcripts.original   = originalTranscript;
      result._meta = { sourceLang: sourceLangLabel, outputLang: language };

      await send({ done: true, result });

    } catch (err) {
      await send({ error: "Analysis failed: " + err.message });
    } finally {
      stopPing();
      await close();
    }
  });
}

// ── Router ────────────────────────────────────────────────────────────────────

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    if (request.method !== "POST") {
      return jsonError("Method Not Allowed", 405);
    }

    if (!env.OPENAI_API_KEY) {
      return jsonError("OPENAI_API_KEY secret not configured. Run: wrangler secret put OPENAI_API_KEY", 500);
    }

    if (url.pathname === "/transcribe") return handleTranscribe(request, env);
    if (url.pathname === "/analyze")    return handleAnalyze(request, env);

    return jsonError("Not found", 404);
  }
};
