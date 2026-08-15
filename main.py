"""Production-ready Discord translator bot.

Features:
* Flag reactions DM a translation to the reacting user.
* Automatic translation between configured language channels via webhooks.
* Same-channel translation when messages differ from a channel's default language.
* /translate, /detect, /channel-link, /channel-unlink, /channel-groups,
  /user-auto, and /user-stop slash commands.
* Gemini provider pool with Groq fallback.
* Resilient batching, partial-success handling, and Discord-safe message/embed splitting.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Environment / logging
# ---------------------------------------------------------------------------

load_dotenv()

LOGGER = logging.getLogger("discord_translator")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

WEBHOOK_NAME = "Translator Mirror"
MAX_MESSAGE_LENGTH = 2_000
MAX_EMBED_DESCRIPTION = 4_096
MAX_EMBED_FIELD_VALUE = 1_024
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))

GEMINI_COOLDOWN_SECONDS = float(
    os.getenv("GEMINI_COOLDOWN_SECONDS", "60")
)
GROQ_MAX_COMPLETION_TOKENS = int(
    os.getenv("GROQ_MAX_COMPLETION_TOKENS", "4096")
)
MAX_TRANSLATION_BATCH_SIZE = int(
    os.getenv("MAX_TRANSLATION_BATCH_SIZE", "3")
)

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config.json"))

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


# ---------------------------------------------------------------------------
# Render health-check server
# ---------------------------------------------------------------------------

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Bot is alive!")

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_health_check_server() -> None:
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    LOGGER.info("Health check server listening on port %s", port)
    server.serve_forever()


threading.Thread(
    target=run_health_check_server,
    daemon=True,
).start()


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

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
    "🇳🇵": "Nepali",
}

LANGUAGE_ALIASES: dict[str, str] = {
    "ar": "Arabic", "arabic": "Arabic",
    "zh": "Chinese", "chinese": "Chinese",
    "da": "Danish", "danish": "Danish",
    "nl": "Dutch", "dutch": "Dutch",
    "en": "English", "english": "English",
    "fi": "Finnish", "finnish": "Finnish",
    "fr": "French", "french": "French",
    "de": "German", "german": "German",
    "el": "Greek", "greek": "Greek",
    "he": "Hebrew", "hebrew": "Hebrew",
    "hi": "Hindi", "hindi": "Hindi",
    "id": "Indonesian", "indonesian": "Indonesian",
    "it": "Italian", "italian": "Italian",
    "ja": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "korean": "Korean",
    "ne": "Nepali", "nepali": "Nepali",
    "no": "Norwegian", "norwegian": "Norwegian",
    "pl": "Polish", "polish": "Polish",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "ru": "Russian", "russian": "Russian",
    "es": "Spanish", "spanish": "Spanish",
    "sv": "Swedish", "swedish": "Swedish",
    "th": "Thai", "thai": "Thai",
    "tr": "Turkish", "turkish": "Turkish",
    "uk": "Ukrainian", "ukrainian": "Ukrainian",
    "vi": "Vietnamese", "vietnamese": "Vietnamese",
}


def canonical_language(value: str) -> str:
    cleaned = value.strip()
    return LANGUAGE_ALIASES.get(cleaned.casefold(), cleaned.title())


def all_language_choices() -> list[app_commands.Choice[str]]:
    values = sorted(set(LANGUAGE_ALIASES.values()))
    return [app_commands.Choice(name=x, value=x) for x in values[:25]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
        return []

    return [
        key.strip().strip("'\"")
        for key in raw_keys.split(",")
        if key.strip()
    ]


@dataclass(frozen=True)
class LinkedChannel:
    group_name: str
    channel_id: int
    guild_id: int
    language: str
    webhook_url: str


class ConfigStore:
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

        data.setdefault("groups", {})
        data.setdefault("user_auto", {})

        if not isinstance(data["groups"], dict):
            raise RuntimeError(f"{self.path}.groups must be a JSON object")
        if not isinstance(data["user_auto"], dict):
            raise RuntimeError(f"{self.path}.user_auto must be a JSON object")

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
                group_name,
                channel_key,
                channels.get(channel_key),
            )
            if linked is not None:
                return linked

        return None

    def get_group(
        self,
        group_name: str,
        guild_id: int | None = None,
    ) -> list[LinkedChannel]:
        channels = self._data["groups"].get(group_name, {})
        if not isinstance(channels, dict):
            return []

        result: list[LinkedChannel] = []

        for channel_id, entry in channels.items():
            linked = self._linked_channel_from_entry(
                group_name,
                channel_id,
                entry,
            )
            if linked is None:
                continue

            if (
                guild_id is not None
                and linked.guild_id != 0
                and linked.guild_id != guild_id
            ):
                continue

            result.append(linked)

        return sorted(result, key=lambda item: item.channel_id)

    def get_groups(
        self,
        guild_id: int | None = None,
    ) -> dict[str, list[LinkedChannel]]:
        result: dict[str, list[LinkedChannel]] = {}

        for group_name in sorted(self._data["groups"]):
            group_channels = self.get_group(
                group_name,
                guild_id=guild_id,
            )
            if group_channels:
                result[group_name] = group_channels

        return result

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
            group_name,
            channel_id,
            guild_id,
            normalized_language,
            webhook_url,
        )

    async def unlink_channel(
        self,
        channel_id: int,
    ) -> list[LinkedChannel]:
        removed: list[LinkedChannel] = []

        async with self._lock:
            groups = self._data["groups"]

            for group_name in list(groups):
                channels = groups[group_name]
                if not isinstance(channels, dict):
                    continue

                linked = self._linked_channel_from_entry(
                    group_name,
                    str(channel_id),
                    channels.get(str(channel_id)),
                )

                if linked is not None:
                    removed.append(linked)
                    channels.pop(str(channel_id), None)

                if not channels:
                    groups.pop(group_name, None)

            if removed:
                await self._save()

        return removed

    async def set_webhook_url(
        self,
        channel_id: int,
        webhook_url: str,
    ) -> None:
        async with self._lock:
            linked = self.get_channel(channel_id)
            if linked is None:
                return

            self._data["groups"][linked.group_name][
                str(channel_id)
            ]["webhook_url"] = webhook_url

            await self._save()

    def get_user_auto_language(
        self,
        user_id: int,
    ) -> str | None:
        value = self._data["user_auto"].get(str(user_id))
        return canonical_language(value) if isinstance(value, str) else None

    async def set_user_auto_language(
        self,
        user_id: int,
        language: str,
    ) -> str:
        normalized_language = canonical_language(language)

        async with self._lock:
            self._data["user_auto"][str(user_id)] = normalized_language
            await self._save()

        return normalized_language

    async def remove_user_auto_language(
        self,
        user_id: int,
    ) -> bool:
        async with self._lock:
            existed = str(user_id) in self._data["user_auto"]
            self._data["user_auto"].pop(str(user_id), None)

            if existed:
                await self._save()

            return existed


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def clean_json_response(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
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


def chunk_text(
    text: str,
    limit: int = MAX_MESSAGE_LENGTH,
) -> list[str]:
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


def batched(
    items: list[str],
    size: int,
) -> Iterable[list[str]]:
    size = max(1, size)

    for index in range(0, len(items), size):
        yield items[index : index + size]


def translation_batch_size(text: str) -> int:
    if len(text) > 3000:
        return 1
    if len(text) > 1500:
        return min(2, MAX_TRANSLATION_BATCH_SIZE)
    return min(3, MAX_TRANSLATION_BATCH_SIZE)


def is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    message = str(exc).upper()

    return (
        status_code == 429
        or code == 429
        or "429" in message
        or "RESOURCE_EXHAUSTED" in message
        or "TOO MANY REQUESTS" in message
    )


def format_reply_header_standalone(
    referenced_message: discord.Message,
) -> str:
    user_mention = f"<@{referenced_message.author.id}>"

    snippet = " ".join(
        referenced_message.content.replace("\n", " ").split()
    )

    if snippet:
        snippet = re.sub(r"https?://\S+", "", snippet).strip()
        if len(snippet) > 40:
            snippet = f"{snippet[:37].rstrip()}…"

    if not snippet:
        if referenced_message.attachments or referenced_message.embeds:
            snippet = "Click to see attachment"
        else:
            snippet = "message"

    return (
        f"> {user_mention} ⇄ "
        f"[**REPLY**]({referenced_message.jump_url}) "
        f"*{snippet}*\n"
    )


# ---------------------------------------------------------------------------
# Translation providers
# ---------------------------------------------------------------------------

class TranslationProviderPool:
    def __init__(
        self,
        gemini_keys: Iterable[str],
        groq_api_key: str,
    ):
        self._gemini_clients = [
            genai.Client(api_key=api_key)
            for api_key in gemini_keys
        ]

        if not self._gemini_clients and not groq_api_key:
            raise RuntimeError(
                "Configure at least one Gemini key or a Groq key"
            )

        self._gemini_cycle = itertools.cycle(
            range(len(self._gemini_clients))
        )
        self._gemini_blocked_until: list[float] = [
            0.0 for _ in self._gemini_clients
        ]

        self._groq_client = (
            AsyncOpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=2,
            )
            if groq_api_key
            else None
        )

        self.http_timeout = aiohttp.ClientTimeout(
            total=LLM_TIMEOUT_SECONDS
        )

    async def _ask_gemini(
        self,
        prompt: str,
        client: genai.Client,
    ) -> dict[str, Any]:
        def request() -> str:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            return response.text or ""

        response_text = await asyncio.wait_for(
            asyncio.to_thread(request),
            timeout=self.http_timeout.total,
        )

        return parse_provider_json(response_text)

    async def _ask_groq(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        if self._groq_client is None:
            raise RuntimeError("Groq fallback is not configured")

        response = await self._groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS,
        )

        content = response.choices[0].message.content or ""
        return parse_provider_json(content)

    async def _ask(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        errors: list[str] = []

        if self._gemini_clients:
            checked_clients = 0

            while checked_clients < len(self._gemini_clients):
                client_index = next(self._gemini_cycle)
                checked_clients += 1

                if (
                    time.monotonic()
                    < self._gemini_blocked_until[client_index]
                ):
                    continue

                try:
                    return await self._ask_gemini(
                        prompt,
                        self._gemini_clients[client_index],
                    )
                except Exception as exc:
                    error_name = type(exc).__name__
                    errors.append(
                        f"Gemini[{client_index}] {error_name}"
                    )

                    if is_rate_limit_error(exc):
                        self._gemini_blocked_until[
                            client_index
                        ] = (
                            time.monotonic()
                            + GEMINI_COOLDOWN_SECONDS
                        )
                        LOGGER.warning(
                            "Gemini[%s] rate limited; cooling down for %.0f seconds",
                            client_index,
                            GEMINI_COOLDOWN_SECONDS,
                        )
                    else:
                        LOGGER.warning(
                            "Gemini[%s] request failed: %s",
                            client_index,
                            error_name,
                        )

        if self._groq_client is not None:
            try:
                LOGGER.info(
                    "Gemini unavailable; using Groq fallback"
                )
                return await self._ask_groq(prompt)
            except Exception as exc:
                errors.append(f"Groq {type(exc).__name__}")
                LOGGER.exception("Groq fallback failed")

        raise RuntimeError(
            "All translation providers failed: "
            + ", ".join(errors)
        )

    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: str | None = None,
    ) -> str:
        target = canonical_language(target_language)

        source_hint = (
            f"The source language is "
            f"{canonical_language(source_language)}.\n"
            if source_language
            else ""
        )

        payload = await self._ask(
            f"""{source_hint}Translate the following Discord message into {target}.

Return ONLY this JSON object:
{{"translation":"<translated text>"}}

Do not omit the translation field.

Text to translate:
{text}"""
        )

        translation = payload.get("translation")

        if (
            not isinstance(translation, str)
            or not translation.strip()
        ):
            raise ValueError(
                "Translation response did not include text"
            )

        return translation.strip()

    async def detect_language(
        self,
        text: str,
    ) -> str:
        payload = await self._ask(
            f"""Detect the primary human language of this Discord message.

Return ONLY:
{{"language":"<English language name>"}}

Text:
{text}"""
        )

        language = payload.get("language")

        if not isinstance(language, str) or not language.strip():
            raise ValueError(
                "Language detection response omitted language"
            )

        return canonical_language(language)

    async def _batch_translate_once(
        self,
        text: str,
        target_languages: list[str],
    ) -> dict[str, str]:
        if not target_languages:
            return {}

        targets = [
            canonical_language(language)
            for language in target_languages
        ]

        target_json = json.dumps(
            targets,
            ensure_ascii=False,
        )

        payload = await self._ask(
            f"""Translate the following Discord message into EVERY language in this list:

{target_json}

Return ONLY a JSON object in exactly this structure:

{{
  "translations": [
    {{
      "language": "<requested language>",
      "text": "<translated text>"
    }}
  ]
}}

Requirements:
- Include exactly one translation object for every requested language.
- The "language" value must exactly match one requested language name.
- Do not omit any requested language.
- Do not add unrequested languages.
- Do not include explanations or commentary.

Text to translate:
{text}"""
        )

        translations = payload.get("translations")

        if not isinstance(translations, list):
            raise ValueError(
                "Batch response did not contain a translations list"
            )

        requested_lookup = {
            language.casefold(): language
            for language in targets
        }

        result: dict[str, str] = {}

        for item in translations:
            if not isinstance(item, dict):
                continue

            language = item.get("language")
            translated_text = item.get("text")

            if not isinstance(language, str):
                continue
            if not isinstance(translated_text, str):
                continue
            if not translated_text.strip():
                continue

            normalized_language = canonical_language(language)
            requested_language = requested_lookup.get(
                normalized_language.casefold()
            )

            if requested_language is None:
                LOGGER.warning(
                    "Provider returned unexpected language: %s",
                    language,
                )
                continue

            result[requested_language] = translated_text.strip()

        return result

    async def batch_translate(
        self,
        text: str,
        target_languages: Iterable[str],
    ) -> dict[str, str]:
        targets = list(
            dict.fromkeys(
                canonical_language(language)
                for language in target_languages
                if str(language).strip()
            )
        )

        if not targets:
            return {}

        result: dict[str, str] = {}
        batch_size = translation_batch_size(text)

        LOGGER.info(
            "Translating into %s languages using batches of %s",
            len(targets),
            batch_size,
        )

        for language_batch in batched(
            targets,
            batch_size,
        ):
            batch_result: dict[str, str] = {}

            try:
                batch_result = await self._batch_translate_once(
                    text,
                    language_batch,
                )
                result.update(batch_result)

            except Exception as exc:
                LOGGER.warning(
                    "Translation batch failed for %s: %s",
                    ", ".join(language_batch),
                    type(exc).__name__,
                )

            missing = [
                language
                for language in language_batch
                if language not in batch_result
            ]

            if missing:
                LOGGER.warning(
                    "Batch omitted %s; retrying individually",
                    ", ".join(missing),
                )

            for language in missing:
                try:
                    result[language] = await self.translate(
                        text,
                        language,
                    )
                except Exception as exc:
                    LOGGER.error(
                        "Translation permanently failed for %s: %s",
                        language,
                        type(exc).__name__,
                    )

        successful = len(result)

        if successful != len(targets):
            still_missing = [
                language
                for language in targets
                if language not in result
            ]
            LOGGER.warning(
                "Translation completed partially: %s/%s languages succeeded. "
                "Still missing: %s",
                successful,
                len(targets),
                ", ".join(still_missing),
            )
        else:
            LOGGER.info(
                "Translation completed successfully: %s/%s languages",
                successful,
                len(targets),
            )

        return result


# ---------------------------------------------------------------------------
# Discord client
# ---------------------------------------------------------------------------

class TranslatorBot(discord.Client):
    def __init__(
        self,
        config: ConfigStore,
        provider_pool: TranslationProviderPool,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.guilds = True
        intents.members = True

        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)
        self.config = config
        self.provider_pool = provider_pool
        self._synced = False

    async def setup_hook(self) -> None:
        register_commands(self)

    async def on_ready(self) -> None:
        if not self._synced:
            try:
                await self.tree.sync()
                self._synced = True
                LOGGER.info("Slash commands synced")
            except Exception:
                LOGGER.exception("Failed syncing slash commands")

        LOGGER.info(
            "Logged in as %s (%s)",
            self.user,
            self.user.id if self.user else "unknown",
        )

    async def on_message(
        self,
        message: discord.Message,
    ) -> None:
        if message.author.bot:
            return

        if message.webhook_id is not None:
            return

        if not message.content.strip():
            return

        linked = self.config.get_channel(message.channel.id)

        # Same-channel translation for messages not matching channel language.
        if linked is not None:
            try:
                detected = await self.provider_pool.detect_language(
                    message.content
                )
            except Exception:
                detected = None
                LOGGER.exception(
                    "Language detection failed for message %s",
                    message.id,
                )

            if (
                detected
                and canonical_language(detected)
                != canonical_language(linked.language)
            ):
                try:
                    same_channel_translation = (
                        await self.provider_pool.translate(
                            message.content,
                            linked.language,
                            source_language=detected,
                        )
                    )
                    await self.send_same_channel_translation(
                        message,
                        linked.language,
                        same_channel_translation,
                    )
                except Exception:
                    LOGGER.exception(
                        "Same-channel translation failed for message %s",
                        message.id,
                    )

            try:
                await self.mirror_message(message, linked)
            except Exception:
                LOGGER.exception(
                    "Failed to mirror message %s",
                    message.id,
                )

        # Optional personal auto-translation.
        auto_language = self.config.get_user_auto_language(
            message.author.id
        )
        if auto_language:
            try:
                translated = await self.provider_pool.translate(
                    message.content,
                    auto_language,
                )
                await self.safe_dm_translation(
                    message.author,
                    message,
                    auto_language,
                    translated,
                )
            except Exception:
                LOGGER.exception(
                    "User auto translation failed for %s",
                    message.author.id,
                )

    async def on_raw_reaction_add(
        self,
        payload: discord.RawReactionActionEvent,
    ) -> None:
        if self.user and payload.user_id == self.user.id:
            return

        target_language = FLAG_LANGUAGES.get(str(payload.emoji))
        if target_language is None:
            return

        channel = self.get_channel(payload.channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(payload.channel_id)
            except Exception:
                return

        if not isinstance(
            channel,
            (discord.TextChannel, discord.Thread),
        ):
            return

        try:
            message = await channel.fetch_message(payload.message_id)
        except Exception:
            LOGGER.exception(
                "Could not fetch reacted message %s",
                payload.message_id,
            )
            return

        if not message.content.strip():
            return

        user = self.get_user(payload.user_id)
        if user is None:
            try:
                user = await self.fetch_user(payload.user_id)
            except Exception:
                return

        try:
            translated = await self.provider_pool.translate(
                message.content,
                target_language,
            )
            await self.safe_dm_translation(
                user,
                message,
                target_language,
                translated,
            )
        except Exception:
            LOGGER.exception(
                "Flag reaction translation failed for message %s",
                message.id,
            )

    async def safe_dm_translation(
        self,
        user: discord.User | discord.Member,
        source_message: discord.Message,
        language: str,
        translated_text: str,
    ) -> None:
        parts = chunk_text(translated_text, 1800)

        for index, part in enumerate(parts, start=1):
            header = (
                f"**{language} translation**"
                if len(parts) == 1
                else f"**{language} translation ({index}/{len(parts)})**"
            )

            await user.send(
                f"{header}\n"
                f"[Original message]({source_message.jump_url})\n\n"
                f"{part}"
            )

    async def send_same_channel_translation(
        self,
        source_message: discord.Message,
        language: str,
        translated_text: str,
    ) -> None:
        parts = chunk_text(translated_text, 1800)

        for index, part in enumerate(parts, start=1):
            prefix = (
                f"🌐 **{language}:** "
                if index == 1
                else ""
            )
            await source_message.channel.send(
                prefix + part,
                reference=(
                    source_message
                    if index == 1
                    else None
                ),
                mention_author=False,
            )

    async def ensure_webhook(
        self,
        linked: LinkedChannel,
    ) -> str:
        channel = self.get_channel(linked.channel_id)

        if channel is None:
            channel = await self.fetch_channel(linked.channel_id)

        if not isinstance(channel, discord.TextChannel):
            raise RuntimeError(
                "Linked destination must be a text channel"
            )

        if linked.webhook_url:
            try:
                webhook = discord.Webhook.from_url(
                    linked.webhook_url,
                    client=self,
                )
                await webhook.fetch()
                return linked.webhook_url
            except Exception:
                LOGGER.warning(
                    "Stored webhook invalid for channel %s; recreating",
                    linked.channel_id,
                )

        webhooks = await channel.webhooks()

        for webhook in webhooks:
            if (
                webhook.name == WEBHOOK_NAME
                and webhook.token
            ):
                url = webhook.url
                await self.config.set_webhook_url(
                    linked.channel_id,
                    url,
                )
                return url

        webhook = await channel.create_webhook(
            name=WEBHOOK_NAME,
            reason="Discord translator channel mirror",
        )

        await self.config.set_webhook_url(
            linked.channel_id,
            webhook.url,
        )

        return webhook.url

    async def send_mirrored_translation(
        self,
        source_message: discord.Message,
        target: LinkedChannel,
        translated_text: str,
    ) -> None:
        webhook_url = await self.ensure_webhook(target)

        webhook = discord.Webhook.from_url(
            webhook_url,
            client=self,
        )

        avatar_url = (
            source_message.author.display_avatar.url
            if source_message.author.display_avatar
            else None
        )

        reply_header = ""

        if source_message.reference:
            resolved = source_message.reference.resolved
            if isinstance(resolved, discord.Message):
                reply_header = format_reply_header_standalone(
                    resolved
                )

        content = reply_header + translated_text
        parts = chunk_text(content, MAX_MESSAGE_LENGTH)

        for part in parts:
            await webhook.send(
                content=part,
                username=source_message.author.display_name[:80],
                avatar_url=avatar_url,
                allowed_mentions=discord.AllowedMentions.none(),
                wait=True,
            )

    async def mirror_message(
        self,
        message: discord.Message,
        linked: LinkedChannel | None = None,
    ) -> None:
        linked = linked or self.config.get_channel(
            message.channel.id
        )

        if linked is None:
            return

        group = self.config.get_group(
            linked.group_name,
            guild_id=message.guild.id if message.guild else None,
        )

        targets = [
            target
            for target in group
            if target.channel_id != message.channel.id
        ]

        if not targets:
            return

        languages_to_translate = list(
            dict.fromkeys(
                target.language
                for target in targets
                if canonical_language(target.language)
                != canonical_language(linked.language)
            )
        )

        translations: dict[str, str] = {}

        if languages_to_translate:
            translations = await self.provider_pool.batch_translate(
                message.content,
                languages_to_translate,
            )

        # If two linked channels happen to use the same language,
        # forward the source text as-is instead of translating.
        translations[
            canonical_language(linked.language)
        ] = message.content

        for target in targets:
            target_language = canonical_language(
                target.language
            )

            translated_text = translations.get(
                target_language
            )

            if not translated_text:
                LOGGER.warning(
                    "Skipping mirror to channel %s because "
                    "%s translation failed",
                    target.channel_id,
                    target.language,
                )
                continue

            try:
                await self.send_mirrored_translation(
                    message,
                    target,
                    translated_text,
                )
            except Exception:
                LOGGER.exception(
                    "Failed sending translation to channel %s",
                    target.channel_id,
                )


# ---------------------------------------------------------------------------
# Slash-command helpers
# ---------------------------------------------------------------------------

async def send_translation_embeds(
    interaction: discord.Interaction,
    translated_text: str,
    language: str,
) -> None:
    parts = chunk_text(
        translated_text,
        limit=4000,
    )

    total = len(parts)

    for index, part in enumerate(parts, start=1):
        if total == 1:
            title = f"Translation — {language}"
        else:
            title = (
                f"Translation — {language} "
                f"({index}/{total})"
            )

        embed = discord.Embed(
            title=title,
            description=part,
        )

        await interaction.followup.send(
            embed=embed,
            ephemeral=True,
        )


def require_guild(
    interaction: discord.Interaction,
) -> discord.Guild:
    if interaction.guild is None:
        raise app_commands.CheckFailure(
            "This command can only be used in a server."
        )
    return interaction.guild


def register_commands(bot: TranslatorBot) -> None:
    @bot.tree.command(
        name="translate",
        description="Translate text into another language.",
    )
    @app_commands.describe(
        text="Text to translate",
        language="Target language, e.g. English or Spanish",
    )
    async def translate_command(
        interaction: discord.Interaction,
        text: str,
        language: str,
    ) -> None:
        await interaction.response.defer(
            ephemeral=True,
            thinking=True,
        )

        try:
            target = canonical_language(language)
            translated = await bot.provider_pool.translate(
                text,
                target,
            )
            await send_translation_embeds(
                interaction,
                translated,
                target,
            )
        except Exception as exc:
            LOGGER.exception("Slash translation failed")
            await interaction.followup.send(
                f"Translation failed: `{type(exc).__name__}`",
                ephemeral=True,
            )

    @bot.tree.command(
        name="detect",
        description="Detect the language of some text.",
    )
    @app_commands.describe(
        text="Text whose language should be detected",
    )
    async def detect_command(
        interaction: discord.Interaction,
        text: str,
    ) -> None:
        await interaction.response.defer(
            ephemeral=True,
            thinking=True,
        )

        try:
            detected = await bot.provider_pool.detect_language(
                text
            )
            await interaction.followup.send(
                f"Detected language: **{detected}**",
                ephemeral=True,
            )
        except Exception as exc:
            LOGGER.exception("Language detection failed")
            await interaction.followup.send(
                f"Detection failed: `{type(exc).__name__}`",
                ephemeral=True,
            )

    @bot.tree.command(
        name="channel-link",
        description="Link this channel to a translation group.",
    )
    @app_commands.default_permissions(manage_channels=True)
    @app_commands.describe(
        group="Name shared by channels that mirror each other",
        language="Default language for this channel",
    )
    async def channel_link_command(
        interaction: discord.Interaction,
        group: str,
        language: str,
    ) -> None:
        guild = require_guild(interaction)

        if not isinstance(
            interaction.channel,
            discord.TextChannel,
        ):
            await interaction.response.send_message(
                "Run this command inside a normal text channel.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(
            ephemeral=True,
            thinking=True,
        )

        channel = interaction.channel
        target_language = canonical_language(language)

        try:
            # Create or reuse the webhook immediately so permission
            # problems are reported at configuration time.
            webhooks = await channel.webhooks()
            webhook = next(
                (
                    item
                    for item in webhooks
                    if item.name == WEBHOOK_NAME
                    and item.token
                ),
                None,
            )

            if webhook is None:
                webhook = await channel.create_webhook(
                    name=WEBHOOK_NAME,
                    reason="Discord translator channel link",
                )

            linked = await bot.config.link_channel(
                group_name=group,
                channel_id=channel.id,
                guild_id=guild.id,
                language=target_language,
                webhook_url=webhook.url,
            )

            await interaction.followup.send(
                f"Linked {channel.mention} to **{linked.group_name}** "
                f"as **{linked.language}**.",
                ephemeral=True,
            )
        except Exception as exc:
            LOGGER.exception("Channel link failed")
            await interaction.followup.send(
                f"Could not link channel: `{type(exc).__name__}`",
                ephemeral=True,
            )

    @bot.tree.command(
        name="channel-unlink",
        description="Remove this channel from its translation group.",
    )
    @app_commands.default_permissions(manage_channels=True)
    async def channel_unlink_command(
        interaction: discord.Interaction,
    ) -> None:
        require_guild(interaction)

        if interaction.channel is None:
            await interaction.response.send_message(
                "No channel is available.",
                ephemeral=True,
            )
            return

        removed = await bot.config.unlink_channel(
            interaction.channel.id
        )

        if not removed:
            await interaction.response.send_message(
                "This channel is not linked.",
                ephemeral=True,
            )
            return

        await interaction.response.send_message(
            "Channel unlinked from its translation group.",
            ephemeral=True,
        )

    @bot.tree.command(
        name="channel-groups",
        description="Show translation channel groups in this server.",
    )
    async def channel_groups_command(
        interaction: discord.Interaction,
    ) -> None:
        guild = require_guild(interaction)
        groups = bot.config.get_groups(guild_id=guild.id)

        if not groups:
            await interaction.response.send_message(
                "No translation channel groups are configured.",
                ephemeral=True,
            )
            return

        lines: list[str] = []

        for group_name, channels in groups.items():
            lines.append(f"**{group_name}**")
            for linked in channels:
                lines.append(
                    f"• <#{linked.channel_id}> — {linked.language}"
                )

        text = "\n".join(lines)
        chunks = chunk_text(text, 1900)

        await interaction.response.send_message(
            chunks[0],
            ephemeral=True,
        )

        for chunk in chunks[1:]:
            await interaction.followup.send(
                chunk,
                ephemeral=True,
            )

    @bot.tree.command(
        name="user-auto",
        description="Automatically DM your messages translated into a language.",
    )
    @app_commands.describe(
        language="Language you want automatic translations in",
    )
    async def user_auto_command(
        interaction: discord.Interaction,
        language: str,
    ) -> None:
        target = await bot.config.set_user_auto_language(
            interaction.user.id,
            language,
        )

        await interaction.response.send_message(
            f"Automatic translation enabled: **{target}**.",
            ephemeral=True,
        )

    @bot.tree.command(
        name="user-stop",
        description="Stop your automatic DM translations.",
    )
    async def user_stop_command(
        interaction: discord.Interaction,
    ) -> None:
        removed = await bot.config.remove_user_auto_language(
            interaction.user.id
        )

        if removed:
            message = "Automatic translation disabled."
        else:
            message = "Automatic translation was not enabled."

        await interaction.response.send_message(
            message,
            ephemeral=True,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    discord_token = required_env("DISCORD_BOT_TOKEN")
    gemini_keys = parse_gemini_keys()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not gemini_keys and not groq_api_key:
        raise RuntimeError(
            "Configure GEMINI_API_KEYS/GEMINI_API_KEY "
            "or GROQ_API_KEY."
        )

    config = ConfigStore(CONFIG_PATH)

    providers = TranslationProviderPool(
        gemini_keys=gemini_keys,
        groq_api_key=groq_api_key,
    )

    bot = TranslatorBot(
        config=config,
        provider_pool=providers,
    )

    bot.run(
        discord_token,
        log_handler=None,
    )


if __name__ == "__main__":
    main()
