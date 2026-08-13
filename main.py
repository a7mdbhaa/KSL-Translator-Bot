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

Include one object for every requested language, using the requested language
names exactly. Do not omit a language.

Text to translate:
{text}"""
        )
        raw_translations = payload.get("translations")
        if not isinstance(raw_translations, list):
            raise ValueError("Batch response did not include translations")

        translations: dict[str, str] = {}
        for item in raw_translations:
            if not isinstance(item, dict):
                continue
            language = item.get("language")
            translated_text = item.get("text")
            if isinstance(language, str) and isinstance(translated_text, str):
                translations[canonical_language(language)] = translated_text.strip()

        missing = [language for language in targets if language not in translations]
        if missing:
            raise ValueError("Batch response omitted languages: " + ", ".join(missing))
        return {language: translations[language] for language in targets}

    async def detect(self, text: str) -> str:
        payload = await self._ask(
            f"""Identify the primary language of the following text.
Return exactly this JSON shape: {{"language":"<language name>"}}

Text to identify:
{text}"""
        )
        language = payload.get("language")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Detection response did not include a language")
        return canonical_language(language)

    async def close(self) -> None:
        if self._groq_client is not None:
            await self._groq_client.close()


class WebhookManager:
    """Find or create bot-owned webhooks and cache them by channel ID."""

    def __init__(self, bot: discord.Client, config_store: ConfigStore):
        self.bot = bot
        self.config_store = config_store
        self._cache: dict[int, discord.Webhook] = {}
        self._lock = asyncio.Lock()

    async def get_for_channel(
        self,
        channel: discord.TextChannel,
        persisted_url: str | None = None,
        *,
        persist: bool = True,
    ) -> discord.Webhook:
        if channel.id in self._cache:
            return self._cache[channel.id]

        async with self._lock:
            if channel.id in self._cache:
                return self._cache[channel.id]

            if persisted_url:
                try:
                    persisted_webhook = discord.Webhook.from_url(
                        persisted_url, client=self.bot
                    )
                    webhook = await persisted_webhook.fetch()
                    self._cache[channel.id] = webhook
                    return webhook
                except discord.HTTPException:
                    LOGGER.info(
                        "Persisted webhook for channel %s is unavailable; repairing it",
                        channel.id,
                    )

            webhooks = await channel.webhooks()
            bot_user_id = self.bot.user.id if self.bot.user else None
            webhook = next(
                (
                    item
                    for item in webhooks
                    if item.name == WEBHOOK_NAME
                    and (item.user is None or item.user.id == bot_user_id)
                ),
                None,
            )
            if webhook is None:
                webhook = await channel.create_webhook(
                    name=WEBHOOK_NAME,
                    reason="Create the Discord Translator mirror webhook",
                )
            self._cache[channel.id] = webhook
            if persist:
                await self.config_store.set_webhook_url(channel.id, str(webhook.url))
            return webhook


def translation_embed(
    *,
    translation: str,
    target_language: str,
    original_message: discord.Message | None = None,
    original_text: str | None = None,
) -> discord.Embed:
    author = original_message.author if original_message else None
    embed = discord.Embed(
        title=f"Translation · {target_language}",
        description=clipped_embed_text(translation),
        color=discord.Color.blurple(),
    )
    if author is not None:
        embed.set_author(
            name=f"{author.display_name} ({author.name})",
            icon_url=author.display_avatar.url,
        )
        embed.set_footer(text=f"Requested from #{original_message.channel.name}")
    if original_text:
        embed.add_field(
            name="Original",
            value=clipped_embed_text(original_text),
            inline=False,
        )
    return embed


class TranslatorBot(discord.Client):
    def __init__(
        self,
        provider_pool: TranslationProviderPool,
        config_store: ConfigStore,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.guilds = True
        intents.members = False
        super().__init__(intents=intents)
        self.provider_pool = provider_pool
        self.config_store = config_store
        self.tree = app_commands.CommandTree(self)
        self.webhooks = WebhookManager(self, config_store)
        self._synced = False

    async def setup_hook(self) -> None:
        if not self._synced:
            synced_commands = await self.tree.sync()
            self._synced = True
            LOGGER.info("Synced %d slash commands", len(synced_commands))

    async def close(self) -> None:
        await self.provider_pool.close()
        await super().close()

    async def on_ready(self) -> None:
        LOGGER.info(
            "Logged in as %s (guilds=%d)",
            self.user,
            len(self.guilds),
        )

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.webhook_id:
            return

        try:
            await self.mirror_message(message)
            await self.auto_translate_user_message(message)
        except discord.Forbidden:
            LOGGER.warning(
                "Missing Discord permission while mirroring message %s",
                message.id,
            )
        except Exception:
            LOGGER.exception("Failed to mirror message %s", message.id)

    async def get_reply_header_for_message(self, message: discord.Message) -> str:
        """Resolve referenced message and return formatted reply header without translating it."""
        reference = message.reference
        if reference is None or reference.message_id is None:
            return ""

        referenced_message = reference.cached_message
        if referenced_message is None:
            try:
                referenced_message = await message.channel.fetch_message(
                    reference.message_id
                )
            except (discord.Forbidden, discord.HTTPException):
                LOGGER.info(
                    "Unable to resolve reply reference %s for message %s",
                    reference.message_id,
                    message.id,
                )
                return ""

        return format_reply_header_standalone(referenced_message)

    async def mirror_message(self, message: discord.Message) -> None:
        if not message.guild:
            return

        raw_content = message.content
        if not raw_content.strip():
            return

        registration = self.config_store.get_channel(message.channel.id)
        if registration is None:
            return

        # 1. Detect source language of the incoming message
        source_language = await self.provider_pool.detect(raw_content)
        normalized_source = canonical_language(source_language)
        channel_language = canonical_language(registration.language)

        # 2. Collect ALL channels in the group
        all_group_channels = self.config_store.get_group(
            registration.group_name, guild_id=message.guild.id
        )

        target_channels: list[tuple[LinkedChannel, bool]] = []
        languages_to_translate: set[str] = set()

        # Check source channel first: Does it need same-channel translation?
        if normalized_source != channel_language:
            target_channels.append((registration, True))
            languages_to_translate.add(channel_language)

        # Check other channels in the group
        for linked in all_group_channels:
            if linked.channel_id == registration.channel_id:
                continue

            target_lang = canonical_language(linked.language)
            if target_lang == normalized_source:
                # Same language -> mirror original text directly
                target_channels.append((linked, False))
            else:
                # Different language -> requires translation
                target_channels.append((linked, True))
                languages_to_translate.add(target_lang)

        if not target_channels:
            return

        # 3. Batch translate only for languages that need translation
        translations: dict[str, str] = {}
        if languages_to_translate:
            translations = await self.provider_pool.batch_translate(
                raw_content, list(languages_to_translate)
            )

        # 4. Fetch untranslated reply header
        reply_header = await self.get_reply_header_for_message(message)

        author_name = f"{message.author.display_name} ({source_language})"
        avatar_url = str(message.author.display_avatar.url)

        # 5. Dispatch messages via webhooks
        for linked, needs_translation in target_channels:
            target_lang = canonical_language(linked.language)

            if needs_translation:
                if target_lang not in translations:
                    continue
                body_text = translations[target_lang]
            else:
                body_text = raw_content

            target_channel = self.get_channel(linked.channel_id)
            if not isinstance(target_channel, discord.TextChannel):
                try:
                    fetched_channel = await self.fetch_channel(linked.channel_id)
                except discord.HTTPException:
                    LOGGER.warning(
                        "Unable to fetch target channel %s", linked.channel_id
                    )
                    continue
                if not isinstance(fetched_channel, discord.TextChannel):
                    continue
                target_channel = fetched_channel

            if target_channel.guild.id != message.guild.id:
                continue

            webhook = await self.webhooks.get_for_channel(
                target_channel, persisted_url=linked.webhook_url or None
            )

            final_text = f"{reply_header}{body_text}"

            for chunk in chunk_text(final_text):
                await webhook.send(
                    content=chunk,
                    username=author_name[:80],
                    avatar_url=avatar_url,
                    wait=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )

    async def auto_translate_user_message(self, message: discord.Message) -> None:
        target_language = self.config_store.get_user_auto_language(message.author.id)
        if target_language is None or not message.content.strip():
            return

        source_language = await self.provider_pool.detect(message.content)
        if canonical_language(source_language) == canonical_language(target_language):
            return

        translation = await self.provider_pool.translate(
            message.content,
            target_language,
            source_language=source_language,
        )
        if not isinstance(message.channel, discord.TextChannel):
            await message.author.send(
                embed=translation_embed(
                    translation=translation,
                    target_language=target_language,
                    original_message=message,
                    original_text=message.content,
                )
            )
            return

        webhook = await self.webhooks.get_for_channel(message.channel, persist=False)
        for chunk in chunk_text(translation):
            await webhook.send(
                content=chunk,
                username=f"{message.author.display_name} ({source_language})"[:80],
                avatar_url=str(message.author.display_avatar.url),
                wait=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    async def on_raw_reaction_add(
        self, payload: discord.RawReactionActionEvent
    ) -> None:
        target_language = FLAG_LANGUAGES.get(payload.emoji.name or "")
        if target_language is None or (
            self.user is not None and payload.user_id == self.user.id
        ):
            return

        channel = self.get_channel(payload.channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(payload.channel_id)
            except discord.HTTPException:
                LOGGER.warning("Unable to fetch reacted channel %s", payload.channel_id)
                return
        if not isinstance(channel, discord.TextChannel):
            return

        try:
            message = await channel.fetch_message(payload.message_id)
        except discord.HTTPException:
            LOGGER.warning("Unable to fetch reacted message %s", payload.message_id)
            return

        if message.author.bot or message.webhook_id or not message.content.strip():
            return

        try:
            translation = await self.provider_pool.translate(
                message.content, target_language
            )
            user = self.get_user(payload.user_id) or await self.fetch_user(
                payload.user_id
            )
            embed = translation_embed(
                translation=translation,
                target_language=target_language,
                original_message=message,
                original_text=message.content,
            )
            await user.send(embed=embed)
        except discord.Forbidden:
            LOGGER.info(
                "Could not DM user %s for flag translation; DMs may be closed",
                payload.user_id,
            )
        except Exception:
            LOGGER.exception("Failed to translate reaction on message %s", message.id)
        finally:
            await self.remove_reaction_if_allowed(message, payload)

    async def remove_reaction_if_allowed(
        self,
        message: discord.Message,
        payload: discord.RawReactionActionEvent,
    ) -> None:
        if not message.guild or not isinstance(message.channel, discord.TextChannel):
            return
        guild_me = message.guild.me
        if guild_me is None:
            return
        permissions = message.channel.permissions_for(guild_me)
        if not permissions.manage_messages:
            return
        try:
            user = self.get_user(payload.user_id) or await self.fetch_user(
                payload.user_id
            )
            await message.remove_reaction(payload.emoji, user)
        except (discord.Forbidden, discord.HTTPException):
            LOGGER.debug("Unable to remove flag reaction from message %s", message.id)


def register_commands(bot: TranslatorBot) -> None:
    async def send_ephemeral(
        interaction: discord.Interaction,
        content: str,
        *,
        embed: discord.Embed | None = None,
    ) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(content, embed=embed, ephemeral=True)
        else:
            await interaction.response.send_message(
                content, embed=embed, ephemeral=True
            )

    async def command_error(
        interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        if isinstance(error, app_commands.errors.MissingPermissions):
            await send_ephemeral(
                interaction,
                "You need Manage Server permission to change translator configuration.",
            )
            return
        if isinstance(error, app_commands.errors.NoPrivateMessage):
            await send_ephemeral(
                interaction,
                "This command can only be used inside a server.",
            )
            return
        LOGGER.exception("Unhandled slash command error", exc_info=error)
        await send_ephemeral(
            interaction,
            "The command could not be completed. Please try again shortly.",
        )

    bot.tree.on_error = command_error

    @bot.tree.command(
        name="channel-link",
        description="Link a channel to a dynamic translation group",
    )
    @app_commands.guild_only()
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(
        group="Name of the translation group",
        language="Language assigned to this channel",
        channel="Channel to link; defaults to the current channel",
    )
    async def channel_link_command(
        interaction: discord.Interaction,
        group: str,
        language: str,
        channel: discord.TextChannel | None = None,
    ) -> None:
        target_channel = channel or interaction.channel
        if (
            not isinstance(target_channel, discord.TextChannel)
            or not target_channel.guild
        ):
            await send_ephemeral(
                interaction,
                "Choose a text channel or run this command in a text channel.",
            )
            return

        try:
            group_name = bot.config_store.normalize_group_name(group)
            normalized_language = canonical_language(language)
            existing = bot.config_store.get_channel(target_channel.id)
            webhook = await bot.webhooks.get_for_channel(
                target_channel,
                persisted_url=existing.webhook_url if existing else None,
                persist=False,
            )
            linked = await bot.config_store.link_channel(
                group_name,
                target_channel.id,
                target_channel.guild.id,
                normalized_language,
                str(webhook.url),
            )
            embed = discord.Embed(
                title="Channel linked",
                description=(
                    f"{target_channel.mention} is now part of **{linked.group_name}**."
                ),
                color=discord.Color.green(),
            )
            embed.add_field(name="Language", value=linked.language)
            embed.add_field(name="Webhook", value="Active")
            await send_ephemeral(interaction, "", embed=embed)
        except (ValueError, discord.Forbidden, discord.HTTPException):
            LOGGER.exception("Channel link failed")
            await send_ephemeral(
                interaction,
                "I could not link that channel. Check the group/language values "
                "and make sure I have Manage Webhooks permission there.",
            )

    @bot.tree.command(
        name="channel-unlink",
        description="Remove a channel from its translation group",
    )
    @app_commands.guild_only()
    @app_commands.checks.has_permissions(manage_guild=True)
    @app_commands.describe(
        channel="Channel to unlink; defaults to the current channel",
    )
    async def channel_unlink_command(
        interaction: discord.Interaction,
        channel: discord.TextChannel | None = None,
    ) -> None:
        target_channel = channel or interaction.channel
        if not isinstance(target_channel, discord.TextChannel):
            await send_ephemeral(
                interaction,
                "Choose a text channel or run this command in a text channel.",
            )
            return

        removed = await bot.config_store.unlink_channel(target_channel.id)
        if not removed:
            await send_ephemeral(
                interaction,
                f"{target_channel.mention} is not linked to a translation group.",
            )
            return

        groups = ", ".join(sorted({item.group_name for item in removed}))
        embed = discord.Embed(
            title="Channel unlinked",
            description=f"{target_channel.mention} was removed from **{groups}**.",
            color=discord.Color.orange(),
        )
        await send_ephemeral(interaction, "", embed=embed)

    @bot.tree.command(
        name="channel-groups",
        description="List active translation channel groups for this server",
    )
    @app_commands.guild_only()
    async def channel_groups_command(interaction: discord.Interaction) -> None:
        guild_id = interaction.guild_id if interaction.guild else None
        groups = bot.config_store.get_groups(guild_id=guild_id)
        embed = discord.Embed(
            title="Active channel groups",
            description="Dynamic translator routing for this server.",
            color=discord.Color.blurple(),
        )
        if not groups:
            embed.description = "No channels are linked in this server yet. Use `/channel-link` to begin."
        else:
            for group_name, linked_channels in groups.items():
                lines = [
                    f"<#{item.channel_id}> · {item.language} · "
                    f"webhook {'active' if item.webhook_url else 'missing'}"
                    for item in linked_channels
                ]
                value = "\n".join(lines)
                embed.add_field(
                    name=group_name,
                    value=value[:1_024],
                    inline=False,
                )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @bot.tree.command(
        name="translate",
        description="Translate text into a language",
    )
    @app_commands.describe(
        text="The text to translate",
        to="Target language, such as Spanish or ja",
        from_="Optional source language hint",
    )
    @app_commands.rename(from_="from")
    async def translate_command(
        interaction: discord.Interaction,
        text: str,
        to: str,
        from_: str | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            target_language = canonical_language(to)
            translation = await bot.provider_pool.translate(
                text, target_language, source_language=from_
            )
            embed = translation_embed(
                translation=translation,
                target_language=target_language,
                original_text=text,
            )
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception:
            LOGGER.exception("Slash translation failed")
            await interaction.followup.send(
                "Translation is temporarily unavailable. Please try again shortly.",
                ephemeral=True,
            )

    @bot.tree.command(
        name="user-auto",
        description="Automatically translate one user's messages",
    )
    @app_commands.guild_only()
    @app_commands.describe(
        language="Language to translate this user's messages into",
        user="User to configure; defaults to yourself",
    )
    async def user_auto_command(
        interaction: discord.Interaction,
        language: str,
        user: discord.Member | None = None,
    ) -> None:
        target_user = user or interaction.user
        actor_is_manager = (
            isinstance(interaction.user, discord.Member)
            and interaction.user.guild_permissions.manage_guild
        )
        if target_user.id != interaction.user.id and not actor_is_manager:
            await send_ephemeral(
                interaction,
                "You can only enable auto-translation for yourself unless you "
                "have Manage Server permission.",
            )
            return

        normalized_language = await bot.config_store.set_user_auto_language(
            target_user.id, language
        )
        embed = discord.Embed(
            title="User auto-translation enabled",
            description=(
                f"{target_user.mention} will be translated into "
                f"**{normalized_language}** in text channels."
            ),
            color=discord.Color.green(),
        )
        await send_ephemeral(interaction, "", embed=embed)

    @bot.tree.command(
        name="user-stop",
        description="Disable automatic translation for a user",
    )
    @app_commands.guild_only()
    @app_commands.describe(
        user="User to configure; defaults to yourself",
    )
    async def user_stop_command(
        interaction: discord.Interaction,
        user: discord.Member | None = None,
    ) -> None:
        target_user = user or interaction.user
        actor_is_manager = (
            isinstance(interaction.user, discord.Member)
            and interaction.user.guild_permissions.manage_guild
        )
        if target_user.id != interaction.user.id and not actor_is_manager:
            await send_ephemeral(
                interaction,
                "You can only disable auto-translation for yourself unless you "
                "have Manage Server permission.",
            )
            return

        removed = await bot.config_store.remove_user_auto_language(target_user.id)
        if removed:
            message = f"Auto-translation disabled for {target_user.mention}."
        else:
            message = f"No auto-translation was configured for {target_user.mention}."
        await send_ephemeral(interaction, message)

    @bot.tree.command(
        name="detect",
        description="Detect the primary language of text",
    )
    @app_commands.describe(text="The text whose language should be detected")
    async def detect_command(
        interaction: discord.Interaction,
        text: str,
    ) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            language = await bot.provider_pool.detect(text)
            embed = discord.Embed(
                title="Language detected",
                description=f"**{language}**",
                color=discord.Color.teal(),
            )
            embed.add_field(name="Text", value=clipped_embed_text(text))
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception:
            LOGGER.exception("Slash detection failed")
            await interaction.followup.send(
                "Language detection is temporarily unavailable. Please try again shortly.",
                ephemeral=True,
            )
    @bot.tree.command(
        name="status", description="Check the status of Storm Translator."
    )
    async def status_command(interaction: discord.Interaction) -> None:
        messages = [
            "Online, active, and keeping our channels connected in memory of Storm ❤️",
            "Good dogs leave paw prints on our hearts forever. Ready to translate!",
            "Translating across channels to keep everyone together—just like a good companion 🐾",
            "Storm Translator is online and watching over all server conversations ✨",
            "Keeping all our language channels linked in honor of Storm 💙",
            "Always here, bridging languages and bringing people closer in Storm's memory 🐶",
            "A loyal companion for all our server conversations, active and ready!",
            "Forever part of our community—Storm Translator is online and connected 🌟",
            "Running smoothly and keeping every channel in sync for everyone ❤️",
        ]
        selected_message = random.choice(messages)
        embed = discord.Embed(
            title="🐾 Storm Translator",
            description=selected_message,
            color=discord.Color.blue(),
        )
        embed.set_footer(text="Translating messages across all server channels.")
        await interaction.response.send_message(embed=embed)






def build_bot() -> TranslatorBot:
    token = required_env("DISCORD_BOT_TOKEN")
    gemini_keys = parse_gemini_keys()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    config_store = ConfigStore(CONFIG_PATH)
    provider_pool = TranslationProviderPool(gemini_keys, groq_api_key)
    bot = TranslatorBot(provider_pool, config_store)
    register_commands(bot)
    bot._translator_token = token  # type: ignore[attr-defined]
    return bot


def main() -> None:
    bot = build_bot()
    token = bot._translator_token  # type: ignore[attr-defined]
    try:
        bot.run(token, log_handler=None)
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested")


if __name__ == "__main__":
    main()
