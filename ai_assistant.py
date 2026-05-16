import os
import json
import logging
import re
from datetime import date
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
_CHAT_MODEL     = "openai/gpt-4.1-nano"
_ANALYSIS_MODEL = "openai/gpt-4.1-nano"
_MAX_TOKENS     = 600


def _strip_fences(raw: str) -> str:
    """Remove accidental markdown code fences from a model response."""
    clean = raw.strip()
    clean = re.sub(r'^```(?:json)?\s*', '', clean)
    clean = re.sub(r'\s*```$', '', clean)
    return clean.strip()


class FaceTrackAI:
    """
    FaceTrack AI assistant.  One instance per application.
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("AIPIPE_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not key:
            log.warning("No AI API key found — AI features will be disabled.")
            self._client = None
            return

        self._client = OpenAI(
            api_key=key,
            base_url="https://aipipe.org/openrouter/v1",
        )
        log.info("FaceTrack AI assistant initialised (model: %s).", _CHAT_MODEL)

    # ── Internal helper ────────────────────────────────────────────────────────

    def _call(self, system: str, user: str,
              model: str = _CHAT_MODEL,
              max_tokens: int = _MAX_TOKENS) -> str | None:
        """Single chat completion call.  Returns None on failure."""
        if self._client is None:
            return None
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_completion_tokens=max_tokens,
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.error("AI call failed: %s", e)
            return None

    def chat(self, question: str, context: dict) -> str:
        """
        Answer a free-form question about the surveillance system.

        `context` should contain serialisable data from the DB — persons,
        today's attendance logs, recent alerts, stats, etc.
        """
        system = """You are FaceTrack, an AI assistant for a face-recognition
surveillance and attendance system. You have access to real-time system data
provided in the user message as JSON. Answer concisely and factually based
only on the data provided. If the data doesn't contain the answer, say so
clearly. Format lists as bullet points. Never invent names or IDs."""

        user = f"""System data (JSON):
{json.dumps(context, indent=2, default=str)}

Question: {question}"""

        answer = self._call(system, user, max_tokens=500)
        return answer or "AI assistant is currently unavailable."

    def daily_briefing(self, stats: dict, logs: list[dict],
                       alerts: list[dict]) -> dict:
        """
        Generate a structured daily briefing.

        Returns a dict with keys:
            summary      — 2-3 sentence plain-English overview
            highlights   — list of notable events (strings)
            mood_trend   — dominant emotion pattern across all check-ins
            action_items — list of recommended actions (strings)
        """
        system = """You are an AI analyst for a security and attendance system.
Given the day's data, produce a structured briefing. Respond ONLY with valid
JSON (no markdown, no backticks) matching exactly this schema:
{
  "summary": "...",
  "highlights": ["...", "..."],
  "mood_trend": "...",
  "action_items": ["...", "..."]
}
Be concise. Base every claim on the data provided."""

        user = f"""Date: {date.today().isoformat()}

Stats:
{json.dumps(stats, indent=2, default=str)}

Today's Attendance Logs ({len(logs)} records):
{json.dumps(logs[:40], indent=2, default=str)}

Recent Alerts ({len(alerts)} active):
{json.dumps(alerts[:10], indent=2, default=str)}"""

        raw = self._call(system, user, model=_ANALYSIS_MODEL, max_tokens=600)
        if not raw:
            return _fallback_briefing()

        try:
            return json.loads(_strip_fences(raw))
        except json.JSONDecodeError:
            log.warning("Briefing JSON parse failed; returning raw as summary.")
            return {"summary": raw, "highlights": [], "mood_trend": "—",
                    "action_items": []}


    def anomaly_check(self, recent_logs: list[dict],
                      persons: list[dict],
                      alerts: list[dict]) -> list[dict]:
        """
        Analyse recent activity for anomalies.

        Returns a list of anomaly dicts:
            { "level": "critical|warning|info",
              "title": "...",
              "detail": "...",
              "employee_ids": [...] }
        """
        system = """You are a security analyst AI. Identify anomalies in the
surveillance data. Look for:
- Visits outside normal hours (before 7am or after 9pm)
- High frequency of unknown/unregistered faces
- People marked as flagged being detected
- Employees with unusually high or zero attendance
- Clusters of negative emotions (angry, sad, fear) on the same day
- Rapid consecutive short visits (< 30 seconds) by the same person

Respond ONLY with a JSON array (no markdown) of anomaly objects:
[{"level":"critical|warning|info","title":"...","detail":"...","employee_ids":[...]}]
Return an empty array [] if no anomalies are found.
Base every claim on the data. Maximum 6 anomalies."""

        user = f"""Date: {date.today().isoformat()}

Recent logs (last 48h, {len(recent_logs)} records):
{json.dumps(recent_logs[:60], indent=2, default=str)}

Enrolled persons ({len(persons)}):
{json.dumps(persons, indent=2, default=str)}

Active alerts ({len(alerts)}):
{json.dumps(alerts[:15], indent=2, default=str)}"""

        raw = self._call(system, user, model=_ANALYSIS_MODEL, max_tokens=700)
        if not raw:
            return []

        try:
            result = json.loads(_strip_fences(raw))
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            log.warning("Anomaly JSON parse failed.")
            return []


    def person_summary(self, person: dict, visit_logs: list[dict]) -> str:
        """
        Generate a concise behavioural profile for a person based on their
        visit history and emotion data.
        """
        system = """You are an AI analyst generating concise behavioural profiles
for a face-recognition attendance system. Based on the person's data and visit
history, write 2-3 sentences covering: visit frequency, typical timing,
emotional pattern, and any notable observations. Be factual and professional.
Do not mention the AI or data format."""

        user = f"""Person: {person.get('name')} ({person.get('role')}, {person.get('department')})
Employee ID: {person.get('employee_id')}
Status: {'Flagged' if person.get('status') == 'flagged' else 'Active'}
Total visits: {person.get('visit_count', 0)}
First seen: {person.get('first_seen', '—')}
Last seen: {person.get('last_seen', '—')}

Visit log (most recent {len(visit_logs)} entries):
{json.dumps(visit_logs[:20], indent=2, default=str)}"""

        result = self._call(system, user, max_tokens=200)
        return result or "Insufficient data to generate a behavioural summary."


# ── Fallback when AI is unavailable ──────────────────────────────────────────

def _fallback_briefing() -> dict:
    return {
        "summary":      "AI briefing unavailable — check AIPIPE_TOKEN in environment.",
        "highlights":   [],
        "mood_trend":   "—",
        "action_items": ["Verify AIPIPE_TOKEN is set in your .env file."],
    }

# ── Module-level singleton ─────────────────────────────────────────────────────
ai_assistant = FaceTrackAI()
