"""Production-ready Discord translator bot.

The bot supports:
* Flag reactions that DM a translation to the reacting user.
* Automatic translation between configured language channels via webhooks (isolated per server).
* Same-channel translation when messages are sent in a language different from the channel's default.
* /translate, /detect, /channel-link, /channel-unlink, /channel-groups,
  /user-auto, and /user-stop slash commands.

Configuration is loaded from environment variables (or a local .env file).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI

import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Tiny HTTP server to pass Render's free Web Service health checks
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is alive!")

def run_health_check_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# Start the health check server in a background thread
threading.Thread(target=run_health_check_server, daemon=True).start()

load_dotenv()

LOGGER = logging.getLogger("discord_translator")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.1-8b-instant"
# Discord rejects webhook usernames containing the word "discord".
WEBHOOK_NAME = "Translator Mirror"
MAX_MESSAGE_LENGTH = 2_000
MAX_EMBED_DESCRIPTION = 4_096
LLM_TIMEOUT_SECONDS = 45

SYSTEM_PROMPT = """You are a careful Discord translation engine.

Preservation rules:
- Translate only the human-readable text; do not translate URLs, custom emoji
  syntax, Markdown structure, code blocks, or Discord user/channel/role
  mentions such as <@123>, <#456>, and <@&789>.
- Preserve line breaks, Markdown formatting, punctuation, emojis, and every
  URL exactly.
- Never add commentary, explanations, labels, quotation marks, or Markdown
  code fences around the translated text.
- Return valid JSON matching the requested schema.
"""

FLAG_LANGUAGES: dict[str, str] = {
    "🇪🇸": "Spanish",
    "🇯🇵": "Japanese",
    "🇸🇦": "Arabic",
    "🇮🇩": "Indonesian",
    "🇫🇷": "French",
    "🇬🇧": "English",
    "🇺🇸": "English",
    "🇩🇪": "German",
    "🇮🇹": "Italian",
    "🇵🇹": "Portuguese",
    "🇧🇷": "Portuguese",
    "🇰🇷": "Korean",
    "🇨🇳": "Chinese",
    "🇹🇼": "Chinese",
    "🇷🇺": "Russian",
    "🇹🇷": "Turkish",
    "🇳🇱": "Dutch",
    "🇵🇱": "Polish",
    "🇸🇪": "Swedish",
    "🇳🇴": "Norwegian",
    "🇩🇰": "Danish",
    "🇫🇮": "Finnish",
    "🇬🇷": "Greek",
    "🇮🇳": "Hindi",
    "🇻🇳": "Vietnamese",
    "🇹🇭": "Thai",
    "🇮🇱": "Hebrew",
    "🇺🇦": "Ukrainian",
}

LANGUAGE_ALIASES: dict[str, str] = {
    "ar": "Arabic",
    "arabic": "Arabic",
    "zh": "Chinese",
    "chinese": "Chinese",
    "da": "Danish",
    "danish": "Danish",
    "nl": "Dutch",
    "dutch": "Dutch",
    "en": "English",
    "english": "English",
    "fi": "Finnish",
    "finnish": "Finnish",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "el": "Greek",
    "greek": "Greek",
    "he": "Hebrew",
    "hebrew": "Hebrew",
    "hi": "Hindi",
    "hindi": "Hindi",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "it": "Italian",
    "italian": "Italian",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "no": "Norwegian",
    "norwegian": "Norwegian",
    "pl": "Polish",
    "polish": "Polish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spanish": "Spanish",
    "sv": "Swedish",
    "swedish": "Swedish",
    "th": "Thai",
    "thai": "Thai",
    "tr": "Turkish",
    "turkish": "Turkish",
    "uk": "Ukrainian",
    "ukrainian": "Ukrainian",
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
}


def canonical_language(value: str) -> str:
    """Return a stable display name while allowing codes in configuration."""
    cleaned = value.strip()
    return LANGUAGE_ALIASES.get(cleaned.casefold(), cleaned.title())


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config.json"))


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def parse_gemini_keys() -> list[str]:
    raw_keys = os.getenv("GEMINI_API_KEYS", "").strip()
    if not raw_keys:
        raw_keys = os.getenv("GEMINI_API_KEY", "").strip()

    if not raw_keys:
        raise RuntimeError("Missing required environment variable: GEMINI_API_KEYS")

    keys = [key.strip().strip("'\"") for key in raw_keys.split(",") if key.strip()]

    if not keys:
        raise RuntimeError("No valid Gemini API keys found in GEMINI_API_KEYS")

    return keys


@dataclass(frozen=True)
class LinkedChannel:
    group_name: str
    channel_id: int
    guild_id: int
    language: str
    webhook_url: str


class ConfigStore:
    """Persistent channel and user-auto configuration backed by config.json."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"groups": {}, "user_auto": {}}

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {self.path}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"{self.path} must contain a JSON object")
        if not isinstance(data.get("groups", {}), dict):
            raise RuntimeError(f"{self.path}.groups must be a JSON object")
        if not isinstance(data.get("user_auto", {}), dict):
            raise RuntimeError(f"{self.path}.user_auto must be a JSON object")
        data.setdefault("groups", {})
        data.setdefault("user_auto", {})
        return data

    async def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        temporary_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.path)

    @staticmethod
    def normalize_group_name(group_name: str) -> str:
        normalized = " ".join(group_name.strip().split())
        if not normalized:
            raise ValueError("Group name cannot be empty")
        if len(normalized) > 100:
            raise ValueError("Group name must be 100 characters or fewer")
        return normalized

    def _linked_channel_from_entry(
        self,
        group_name: str,
        channel_id: str,
        entry: Any,
    ) -> LinkedChannel | None:
        if not isinstance(entry, dict):
            return None
        language = entry.get("language")
        webhook_url = entry.get("webhook_url", "")
        guild_id = entry.get("guild_id", 0)
        try:
            parsed_channel_id = int(channel_id)
            parsed_guild_id = int(guild_id) if guild_id else 0
        except (TypeError, ValueError):
            return None
        if not isinstance(language, str) or not language.strip():
            return None
        if not isinstance(webhook_url, str):
            webhook_url = ""
        return LinkedChannel(
            group_name=group_name,
            channel_id=parsed_channel_id,
            guild_id=parsed_guild_id,
            language=canonical_language(language),
            webhook_url=webhook_url,
        )

    def get_channel(self, channel_id: int) -> LinkedChannel | None:
        channel_key = str(channel_id)
        for group_name, channels in self._data["groups"].items():
            if not isinstance(channels, dict):
                continue
            linked = self._linked_channel_from_entry(
                group_name, channel_key, channels.get(channel_key)
            )
            if linked is not None:
                return linked
        return None

    def get_group(
        self, group_name: str, guild_id: int | None = None
    ) -> list[LinkedChannel]:
        channels = self._data["groups"].get(group_name, {})
        if not isinstance(channels, dict):
            return []
        result: list[LinkedChannel] = []
        for channel_id, entry in channels.items():
            linked = self._linked_channel_from_entry(group_name, channel_id, entry)
            if linked is not None:
                if (
                    guild_id is not None
                    and linked.guild_id != 0
                    and linked.guild_id != guild_id
                ):
                    continue
                result.append(linked)
        return sorted(result, key=lambda item: item.channel_id)

    def get_groups(self, guild_id: int | None = None) -> dict[str, list[LinkedChannel]]:
        res: dict[str, list[LinkedChannel]] = {}
        for group_name in sorted(self._data["groups"]):
            group_list = self.get_group(group_name, guild_id=guild_id)
            if group_list:
                res[group_name] = group_list
        return res

    async def link_channel(
        self,
        group_name: str,
        channel_id: int,
        guild_id: int,
        language: str,
        webhook_url: str,
    ) -> LinkedChannel:
        group_name = self.normalize_group_name(group_name)
        normalized_language = canonical_language(language)
        if not normalized_language:
            raise ValueError("Language cannot be empty")

        async with self._lock:
            groups = self._data["groups"]
            for existing_group in list(groups):
                channels = groups[existing_group]
                if not isinstance(channels, dict):
                    continue
                channels.pop(str(channel_id), None)
                if not channels:
                    groups.pop(existing_group, None)

            groups.setdefault(group_name, {})[str(channel_id)] = {
                "guild_id": guild_id,
                "language": normalized_language,
                "webhook_url": webhook_url,
            }
            await self._save()

        return LinkedChannel(
            group_name, channel_id, guild_id, normalized_language, webhook_url
        )

    async def unlink_channel(self, channel_id: int) -> list[LinkedChannel]:
        removed: list[LinkedChannel] = []
        async with self._lock:
            groups = self._data["groups"]
            for group_name in list(groups):
                channels = groups[group_name]
                if not isinstance(channels, dict):
                    continue
                linked = self._linked_channel_from_entry(
                    group_name, str(channel_id), channels.get(str(channel_id))
                )
                if linked is not None:
                    removed.append(linked)
                    channels.pop(str(channel_id), None)
                if not channels:
                    groups.pop(group_name, None)
            if removed:
                await self._save()
        return removed

    async def set_webhook_url(self, channel_id: int, webhook_url: str) -> None:
        async with self._lock:
            linked = self.get_channel(channel_id)
            if linked is None:
                return
            self._data["groups"][linked.group_name][str(channel_id)]["webhook_url"] = (
                webhook_url
            )
            await self._save()

    def get_user_auto_language(self, user_id: int) -> str | None:
        value = self._data["user_auto"].get(str(user_id))
        return canonical_language(value) if isinstance(value, str) else None

    async def set_user_auto_language(self, user_id: int, language: str) -> str:
        normalized_language = canonical_language(language)
        async with self._lock:
            self._data["user_auto"][str(user_id)] = normalized_language
            await self._save()
        return normalized_language

    async def remove_user_auto_language(self, user_id: int) -> bool:
        async with self._lock:
            existed = str(user_id) in self._data["user_auto"]
            self._data["user_auto"].pop(str(user_id), None)
            if existed:
                await self._save()
            return existed


def clean_json_response(value: str) -> str:
    """Remove accidental fences and isolate the JSON object."""
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Provider returned no JSON object")
    return cleaned[start : end + 1]


def parse_provider_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(clean_json_response(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Provider returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Provider response must be a JSON object")
    return parsed


def chunk_text(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split long translations at a natural boundary for Discord."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def clipped_embed_text(text: str) -> str:
    if len(text) <= MAX_EMBED_DESCRIPTION:
        return text
    return f"{text[: MAX_EMBED_DESCRIPTION - 1].rstrip()}…"


def format_reply_header_standalone(referenced_message: discord.Message) -> str:
    """Build a clickable nvu.io-style quote header for a Discord reply."""
    user_mention = f"<@{referenced_message.author.id}>"

    snippet = " ".join(referenced_message.content.replace("\n", " ").split())
    if snippet:
        snippet = re.sub(r"https?://\S+", "", snippet).strip()
        if len(snippet) > 40:
            snippet = f"{snippet[:37].rstrip()}…"

    if not snippet:
        if referenced_message.attachments or referenced_message.embeds:
            snippet = "Click to see attachment"
        else:
            snippet = "message"

    message_jump_url = referenced_message.jump_url

    return f"> {user_mention} ⇄ [**REPLY**]({message_jump_url}) *{snippet}*\n"


class TranslationProviderPool:
    """Gemini key pool with Groq fallback and structured JSON responses."""

    def __init__(self, gemini_keys: Iterable[str], groq_api_key: str):
        self._gemini_clients = [
            genai.Client(api_key=api_key) for api_key in gemini_keys
        ]
        if not self._gemini_clients and not groq_api_key:
            raise RuntimeError("Configure at least one Gemini key or a Groq key")

        self._gemini_cycle = itertools.cycle(range(len(self._gemini_clients)))
        self._groq_client = (
            AsyncOpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=LLM_TIMEOUT_SECONDS,
            )
            if groq_api_key
            else None
        )
        self.http_timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT_SECONDS)

    async def _ask_gemini(self, prompt: str, client: genai.Client) -> dict[str, Any]:
        def request() -> str:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return response.text or ""

        response_text = await asyncio.wait_for(
            asyncio.to_thread(request),
            timeout=self.http_timeout.total,
        )
        return parse_provider_json(response_text)

    async def _ask_groq(self, prompt: str) -> dict[str, Any]:
        if self._groq_client is None:
            raise RuntimeError("Groq fallback is not configured")

        response = await self._groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return parse_provider_json(content)

    async def _ask(self, prompt: str) -> dict[str, Any]:
        errors: list[str] = []
        for _ in range(len(self._gemini_clients)):
            client_index = next(self._gemini_cycle)
            try:
                return await self._ask_gemini(
                    prompt, self._gemini_clients[client_index]
                )
            except Exception as exc:
                error_name = type(exc).__name__
                errors.append(f"Gemini[{client_index}] {error_name}")
                LOGGER.warning("Gemini request failed: %s", error_name)

        if self._groq_client is not None:
            try:
                LOGGER.info("All Gemini clients failed; using Groq fallback")
                return await self._ask_groq(prompt)
            except Exception as exc:
                errors.append(f"Groq {type(exc).__name__}")
                LOGGER.exception("Groq fallback failed")

        raise RuntimeError("All translation providers failed: " + ", ".join(errors))

    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> str:
        target = canonical_language(target_language)
        source_hint = (
            f"The source language is {canonical_language(source_language)}.\n"
            if source_language
            else ""
        )
        payload = await self._ask(
            f"""{source_hint}Translate the following Discord message into {target}.
Return exactly this JSON shape: {{"translation":"..."}}

Text to translate:
{text}"""
        )
        translation = payload.get("translation")
        if not isinstance(translation, str) or not translation.strip():
            raise ValueError("Translation response did not include text")
        return translation.strip()

    async def batch_translate(
        self, text: str, target_languages: Iterable[str]
    ) -> dict[str, str]:
        targets = [canonical_language(language) for language in target_languages]
        target_json = json.dumps(targets, ensure_ascii=False)
        payload = await self._ask(
            f"""Translate the following Discord message into every language in this list:
{target_json}

Return exactly this JSON shape:
{{"translations":[{{"language":"<one requested language>","text":"<translation>"}}, ...]}}

Include one object for every reques
